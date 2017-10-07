#include "information_filter.h"

#include <memory>
#include <iostream>

InformationFilter::InformationFilter(const Eigen::VectorXd &x0,
                                     const Eigen::MatrixXd &covariance0,
                                     LinearMotionModel::ConstPtr motion_model,
                                     LinearMeasurementModel::ConstPtr measurement_model)
{
  type_ = "InformationFilter";

  x_ = x0;
  covariance_ = covariance0;
  information_matrix_ = covariance0.inverse();
  information_vector_ = information_matrix_ * x_;
  motion_model_ = motion_model;
  measurement_model_ = measurement_model;
}

void InformationFilter::predict(const Eigen::VectorXd &u)
{
  LinearMotionModel::ConstPtr linear_motion_model = std::static_pointer_cast<const LinearMotionModel>(motion_model_);

  const Eigen::MatrixXd A = linear_motion_model->getA();
  const Eigen::MatrixXd A_transpose = A.transpose();
  const Eigen::MatrixXd B = linear_motion_model->getB();
  const Eigen::MatrixXd information_matrix_inv = information_matrix_.inverse();

  information_matrix_ = (A_transpose * information_matrix_inv * A + linear_motion_model->calculateRt()).inverse();
  information_vector_ = information_matrix_ * (A * information_matrix_inv * information_vector_ + B * u);

  covariance_ = information_matrix_.inverse();
  x_ = covariance_ * information_vector_;
}

void InformationFilter::correct(const Eigen::VectorXd &z)
{
  LinearMeasurementModel::ConstPtr linear_measurement_model = std::static_pointer_cast<const LinearMeasurementModel>(measurement_model_);

  const Eigen::MatrixXd C = linear_measurement_model->getC();
  const Eigen::MatrixXd C_transpose = C.transpose();
  const Eigen::MatrixXd Q_inverse = (linear_measurement_model->getQt()).inverse();

  information_matrix_ += C_transpose * Q_inverse * C;
  information_vector_ += C_transpose * Q_inverse * z;

  covariance_ = information_matrix_.inverse();
  x_ = covariance_ * information_vector_;
}
