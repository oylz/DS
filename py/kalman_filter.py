# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
from sklearn.utils.linear_assignment_ import linear_assignment

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """
    def LinearAssignmentForCpp(self, cost_matrixi, rows, cols):
        cost_matrix = np.reshape(cost_matrixi, (rows, cols)).astype(np.float)
        indices = linear_assignment(cost_matrix)
        return indices
            
    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat_ = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat_[i, ndim + i] = dt
        self._update_mat_ = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position_ = 1. / 20
        self._std_weight_velocity_ = 1. / 160

    def initiate(self, measurementi):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        measurement = np.reshape(measurementi, (4,)).astype(np.float)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
    
        std = [
            2 * self._std_weight_position_ * measurement[3],
            2 * self._std_weight_position_ * measurement[3],
            1e-2,
            2 * self._std_weight_position_ * measurement[3],
            10 * self._std_weight_velocity_ * measurement[3],
            10 * self._std_weight_velocity_ * measurement[3],
            1e-5,
            10 * self._std_weight_velocity_ * measurement[3]]
        covariance = np.diag(np.square(std))
        print("[-4--]begin mean:")
        print(mean)
        print("[-4--]end mean")
        print("[-4--]begin covariance:")
        print(covariance)
        print("[-4--]end covariance")
        return mean, covariance

    def predict(self, meani, covariancei):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        mean = np.reshape(meani, (8,)).astype(np.float)
        covariance = np.reshape(covariancei, (8, 8)).astype(np.float)
        
        std_pos = [
            self._std_weight_position_ * mean[3],
            self._std_weight_position_ * mean[3],
            1e-2,
            self._std_weight_position_ * mean[3]]
        std_vel = [
            self._std_weight_velocity_ * mean[3],
            self._std_weight_velocity_ * mean[3],
            1e-5,
            self._std_weight_velocity_ * mean[3]]
        
        print("[-3--]begin square")
        print(np.square(np.r_[std_pos, std_vel]))
        print("[-3--]end square")
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat_, mean)
        print("[-3--]begin self._motion_mat_")
        print(self._motion_mat_)
        print("[-3--]end self._motion_mat_")
        print("[-3--]begin covariance")
        print(covariance)
        print("[-3--]end covariance")
    
        print("[-3--]begin motion_cov:") 
        print(motion_cov)
        print("[-3--]end motion_cov:") 

        covariance = np.linalg.multi_dot((
            self._motion_mat_, covariance, self._motion_mat_.T)) + motion_cov
        print("[-3--]begin covariance result")
        print(covariance)
        print("[-3--]end covariance result")
        
        return mean, covariance

    def update(self, meani, covariancei, measurementi):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        mean = np.reshape(meani, (8,)).astype(np.float)
        covariance = np.reshape(covariancei, (8, 8)).astype(np.float)
        measurement = np.reshape(measurementi, (4,)).astype(np.float)
        
        projected_mean, projected_cov = self._project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        """print("[-1--]begin projected_cov")
        print(projected_cov)
        print("[-1--]bend projected_cov")
        print("[-1--]bbegin (chol_factor, lower)")
        print((chol_factor, lower))
        print("[-1--]bend (chol_factor, lower)")"""
        ddd = np.dot(covariance, self._update_mat_.T)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat_.T).T,
            check_finite=False).T
        print("[-1--]bbegin ddd")
        print(ddd)
        print("[-1--]bend ddd")
        print("[-1--]bbegin kalman_gain")
        print(kalman_gain)
        print("[-1--]bend kalman_gain")
        print("[-1--]begin measurement")
        print(measurement)
        print("[-1--]end measurement")
        print("[-1--]begin projected_mean")
        print(projected_mean)
        print("[-1--]end projectd_mean")
        innovation = measurement - projected_mean
        print("[-1--]begin innovation")
        print(innovation)
        print("[-1--]end innovation")
        print("[-1--]begin projected_cov")
        print(projected_cov)
        print("[-1--]end projectd_cov")

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, meani, covariancei, measurementsi,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean = np.reshape(meani, (8,)).astype(np.float)
        covariance = np.reshape(covariancei, (8, 8)).astype(np.float)
        measurements = np.reshape(measurementsi, (-1, 4)).astype(np.float)
        print("[-2--]begin mean")
        print(mean)
        print("[-2--]end mean")
        print("[-2--]begin covariance")
        print(covariance)
        print("[-2--]end covariance")
        print("[-2--]begin measurements")
        print(measurements)
        print("[-2--]end measurements")
      
        mean, covariance = self._project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        print("[-2--]begin mean1")
        print(mean)
        print("[-2--]end mean1")
        print("[-2--]begin covariance1")
        print(covariance)
        print("[-2--]end covariance1")

        cholesky_factor = np.linalg.cholesky(covariance)
        """print("[-2--]begin covariance:")
        print(covariance)
        print("[-2--]bend covariance:")"""
        print("[-2--]bbegin cholesky_factor")
        print(cholesky_factor)
        print("[-2--]bend cholesky_factor")
        """print("[-2--]measurements:\n")
        print(measurements)
        print("[-2--]mean:\n")
        print(mean)"""
        d = measurements - mean
        print("[-2--]bbegin d")
        print d
        print("[-2--]bend d")
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        print("[-2--]begin z")
        print(z)
        print("[-2--]end z")
        squared_maha = np.sum(z * z, axis=0)
        print("[-2--]begin squared_maha")
        print(squared_maha)
        print("[-2--]end squared_maha")
        return squared_maha
    
    def _project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position_ * mean[3],
            self._std_weight_position_ * mean[3],
            1e-1,
            self._std_weight_position_ * mean[3]]
        print("[-0--]begin mtmp")
        print(np.square(std))
        print("[-0--]end mtmp")


        innovation_cov = np.diag(np.square(std))
        print("[-0--]begin innovation_cov")
        print(innovation_cov)
        print("[-0--]end innovation_cov")

        mean = np.dot(self._update_mat_, mean)
        print("[-0--]begin var")
        print(covariance)
        print("[-0--]end var")

        print("[-0--]begin _update_mat_")
        print(self._update_mat_)
        print("[-0--]end _update_mat_")


        covariance = np.linalg.multi_dot((
            self._update_mat_, covariance, self._update_mat_.T))
        print("[-0--]begin var1")
        print(covariance)
        print("[-0--]end var1")


        return mean, covariance + innovation_cov
    
