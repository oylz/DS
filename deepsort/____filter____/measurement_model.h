#ifndef INCLUDE_MEASUREMENT_MODEL_H_
#define INCLUDE_MEASUREMENT_MODEL_H_
#include <Eigen>
//#include <Eigen/Dense>
#include <string>
#include <memory>

class MeasurementModel
{
 public:
  typedef std::shared_ptr<MeasurementModel> Ptr;
  typedef std::shared_ptr<const MeasurementModel> ConstPtr;

  virtual ~MeasurementModel()
  {
  }

  // TODO(markcsie): for non-linear model
  // virtual Eigen::VectorXd h(const Eigen::VectorXd &x) = 0;

  const std::string &getType() const
  {
    return type_;
  }

  const Eigen::MatrixXd &getQt() const
  {
    return Qt_;
  }

  const Eigen::MatrixXd &getQtInv() const
  {
    return Qt_inv_;
  }

 protected:
  std::string type_;
  Eigen::MatrixXd Qt_;
  Eigen::MatrixXd Qt_inv_;
};

#endif  // INCLUDE_MEASUREMENT_MODEL_H_
