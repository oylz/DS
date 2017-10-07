#ifndef INCLUDE_MOTION_MODEL_H_
#define INCLUDE_MOTION_MODEL_H_
#include <Eigen>
//#include <Eigen/Dense>
#include <string>
#include <memory>

class MotionModel
{
 public:
  typedef std::shared_ptr<MotionModel> Ptr;
  typedef std::shared_ptr<const MotionModel> ConstPtr;


  virtual ~MotionModel()
  {
  }

  // TODO(markcsie): for non-linear model
  // virtual Eigen::VectorXd g(const Eigen::VectorXd &u, const Eigen::VectorXd &x) const = 0;

  virtual const Eigen::MatrixXd &calculateRt() const
  {
    return Rt_;
  }

  const std::string &getType() const
  {
    return type_;
  }

 protected:
  Eigen::MatrixXd Rt_;
  std::string type_;
};

#endif  // INCLUDE_MOTION_MODEL_H_
