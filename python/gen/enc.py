import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gen.nw as nw

import time

class Enc(object):
    def _preprocess(self, image, is_training=False, enable_more_augmentation=True):
        image = image[:, :, ::-1]  # BGR to RGB
        if is_training:
            image = tf.image.random_flip_left_right(image)
            if enable_more_augmentation:
                image = tf.image.random_brightness(image, max_delta=50)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        return image
    
    
    def _run_in_batches(self, f, data_dict, out, batch_size):
        data_len = len(out)
        num_batches = int(data_len / batch_size)
    
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
            out[s:e] = f(batch_data_dict)
        if e < len(out):
            batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
            out[e:] = f(batch_data_dict)
    
    
    
    def preEncode(self, image_shape, batch_size=32, 
                   checkpoint_path=None,
                             loss_mode="cosine"):
        self.batch_size = batch_size
        self.image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)
    
        preprocessed_image_var = tf.map_fn(
            lambda x: self._preprocess(x, is_training=False),
            tf.cast(self.image_var, tf.float32))
    
        l2_normalize = loss_mode == "cosine"
        self.nw = nw.Nw()
        self.feature_var, _ = self.nw.factory_fn(
            preprocessed_image_var, l2_normalize=l2_normalize, reuse=None)
        self.feature_dim = self.feature_var.get_shape().as_list()[-1]
    
        self.session = tf.Session()
        if checkpoint_path is not None:
            slim.get_or_create_global_step()
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path, slim.get_variables_to_restore())
            self.session.run(init_assign_op, feed_dict=init_feed_dict)
            
    def encode(self, data_x):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        enct1 = int(round(time.time() * 1000))
        self._run_in_batches(
            lambda x: self.session.run(self.feature_var, feed_dict=x),
            {self.image_var: data_x}, out, self.batch_size)
        enct2 = int(round(time.time() * 1000))
        print("one frame cost time:t2-t1:%s" % (str(enct2-enct1)))        
        return out
    
    
    
    
    
