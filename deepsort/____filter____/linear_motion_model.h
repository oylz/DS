#ifndef INCLUDE_LINEAR_MOTION_MODEL_H_
#define INCLUDE_LINEAR_MOTION_MODEL_H_
#include <Eigen>
//#include <Eigen/Dense>
#include <memory>
#include "./motion_model.h"

class LinearMotionModel : public MotionModel
{
 public:
  typedef std::shared_ptr<LinearMotionModel> Ptr;
  typedef std::shared_ptr<const LinearMotionModel> ConstPtr;

  LinearMotionModel(const Eigen::MatrixXd &A,
                    const Eigen::MatrixXd &B,
                    const Eigen::MatrixXd &Rt) : A_(A), B_(B)
  {
    Rt_ = Rt;
  }

  virtual ~LinearMotionModel()
  {
  }

  const Eigen::MatrixXd &getA() const
  {
    return A_;
  }

  const Eigen::MatrixXd &getB() const
  {
    return B_;
  }

 protected:
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
};

#endif  // INCLUDE_LINEAR_MOTION_MODEL_H_
