#ifndef INCLUDE_INFORMATION_FILTER_H_
#define INCLUDE_INFORMATION_FILTER_H_

#include "./filter.h"
#include "./linear_motion_model.h"
#include "./linear_measurement_model.h"

class InformationFilter : public Filter
{
 public:
  InformationFilter(const Eigen::VectorXd &x0,
                    const Eigen::MatrixXd &covariance0,
                    LinearMotionModel::ConstPtr motion_model,
                    LinearMeasurementModel::ConstPtr measurement_model);

  ~InformationFilter() {}

  virtual void predict(const Eigen::VectorXd &u);
  virtual void correct(const Eigen::VectorXd &z);

  const Eigen::MatrixXd &getCovariance() const
  {
    return covariance_;
  }

 protected:
  Eigen::MatrixXd information_matrix_;
  Eigen::MatrixXd covariance_;
  Eigen::VectorXd information_vector_;
};

#endif  // INCLUDE_INFORMATION_FILTER_H_
