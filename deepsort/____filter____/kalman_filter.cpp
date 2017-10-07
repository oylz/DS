#include "kalman_filter.h"

#include <memory>
#include <iostream>

KalmanFilter1::KalmanFilter1(const Eigen::VectorXd &x0,
                           const Eigen::MatrixXd &covariance0,
                           LinearMotionModel::ConstPtr motion_model,
                           LinearMeasurementModel::ConstPtr measurement_model)
{
  type_ = "KalmanFilter1";

  x_ = x0;
  mean_ = x_;
  covariance_ = covariance0;
  motion_model_ = motion_model;
  measurement_model_ = measurement_model;
}

void KalmanFilter1::predict(const Eigen::VectorXd &u)
{
  LinearMotionModel::ConstPtr linear_motion_model = std::static_pointer_cast<const LinearMotionModel>(motion_model_);

  const Eigen::MatrixXd A = linear_motion_model->getA();
  const Eigen::MatrixXd A_transpose = A.transpose();
  const Eigen::MatrixXd B = linear_motion_model->getB();

  mean_ = A * mean_ + B * u;
  covariance_ = A * covariance_ * A_transpose + linear_motion_model->calculateRt();

  x_ = mean_;
}

void KalmanFilter1::correct(const Eigen::VectorXd &z)
{
  LinearMeasurementModel::ConstPtr linear_measurement_model = std::static_pointer_cast<const LinearMeasurementModel>(measurement_model_);

  const Eigen::MatrixXd C = linear_measurement_model->getC();
  const Eigen::MatrixXd C_transpose = C.transpose();
  const Eigen::MatrixXd Q = linear_measurement_model->getQt();

  const Eigen::MatrixXd kalman_gain = covariance_ * C_transpose * (C * covariance_ * C_transpose + Q).inverse();

  mean_ += kalman_gain * (z - C * mean_);
  covariance_ = (Eigen::MatrixXd::Identity(mean_.rows(), mean_.rows()) - kalman_gain * C) * covariance_;

  x_ = mean_;
}
