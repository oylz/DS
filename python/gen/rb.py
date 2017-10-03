
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Rb(object):
    #========begin====residual_block==========================
    def _batch_norm_fn(self, x, scope=None):
        if scope is None:
            scope = tf.get_variable_scope().name + "/bn"
        return slim.batch_norm(x, scope=scope)
    
    def create_inner_block(self, incoming, scope):
        n = incoming.get_shape().as_list()[-1]
        stride = 1
        if self.increase_dim:
            n *= 2
            stride = 2
    
        incoming = slim.conv2d(
            incoming, n, [3, 3], stride, activation_fn=self.nonlinearity, padding="SAME",
            normalizer_fn=self._batch_norm_fn, weights_initializer=self.weights_initializer,
            biases_initializer=self.bias_initializer, weights_regularizer=self.regularizer,
            scope=scope + "/1")
        if self.summarize_activations:
            tf.summary.histogram(incoming.name + "/activations", incoming)
    
        incoming = slim.dropout(incoming, keep_prob=0.6)
    
        incoming = slim.conv2d(
            incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
            normalizer_fn=None, weights_initializer=self.weights_initializer,
            biases_initializer=self.bias_initializer, weights_regularizer=self.regularizer,
            scope=scope + "/2")
        return incoming
        
    def create_link(self, incoming, scope):
        if self.is_first:
            network = incoming
        else:
            network = self._batch_norm_fn(incoming, scope=scope + "/bn")
            network = self.nonlinearity(network)
            if self.summarize_activations:
                tf.summary.histogram(scope+"/activations", network)
    
        pre_block_network = network
        post_block_network = self.create_inner_block(pre_block_network, scope)
    
        incoming_dim = pre_block_network.get_shape().as_list()[-1]
        outgoing_dim = post_block_network.get_shape().as_list()[-1]
        if incoming_dim != outgoing_dim:
            assert outgoing_dim == 2 * incoming_dim, \
                "%d != %d" % (outgoing_dim, 2 * incoming)
            projection = slim.conv2d(
                incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
                scope=scope+"/projection", weights_initializer=self.weights_initializer,
                biases_initializer=None, weights_regularizer=self.regularizer)
            network = projection + post_block_network
        else:
            network = incoming + post_block_network
        return network
    
    
    
    def residual_block(self, incoming, scope, nonlinearity=tf.nn.elu,
                       weights_initializer=tf.truncated_normal_initializer(1e3),
                       bias_initializer=tf.zeros_initializer(), regularizer=None,
                       increase_dim=False, is_first=False,
                       summarize_activations=True):
        self.nonlinearity = nonlinearity
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.regularizer = regularizer
        self.increase_dim = increase_dim
        self.summarize_activations = summarize_activations
        self.is_first = is_first
            
        return self.create_link(incoming, scope)
    #========end====residual_block==========================
