#ifndef INCLUDE_LINEAR_MEASUREMENT_MODEL_H_
#define INCLUDE_LINEAR_MEASUREMENT_MODEL_H_

#include <memory>
#include "./measurement_model.h"

class LinearMeasurementModel : public MeasurementModel
{
 public:
  typedef std::shared_ptr<LinearMeasurementModel> Ptr;
  typedef std::shared_ptr<const LinearMeasurementModel> ConstPtr;

  LinearMeasurementModel(const Eigen::MatrixXd &C, const Eigen::MatrixXd &Qt) : C_(C)
  {
    Qt_ = Qt;
    Qt_inv_ = Qt.inverse();
  }

  virtual ~LinearMeasurementModel()
  {
  }

  const Eigen::MatrixXd &getC() const
  {
    return C_;
  }

 protected:
  Eigen::MatrixXd C_;
};

#endif  // INCLUDE_LINEAR_MEASUREMENT_MODEL_H_
