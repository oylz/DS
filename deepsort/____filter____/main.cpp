#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "../include/filter.h"
#include "../include/linear_motion_model.h"
#include "../include/linear_measurement_model.h"
#include "../include/kalman_filter.h"
#include "../include/information_filter.h"

struct data
{
  double real_x;
  double u;
  double z;
};

int main()
{
  // read data
  std::vector<data> data_vector;
  std::string line;
  std::ifstream input_file("../data/noisy.data");
  while (std::getline(input_file, line))
  {
    std::istringstream iss(line);
    data d;
    iss >> d.z;
    iss >> d.real_x;
    iss >> d.u;
    data_vector.push_back(d);
  }

  Eigen::MatrixXd A(1, 1);
  A << 0.5;
  Eigen::MatrixXd B(1, 1);
  B << 1.5;
  Eigen::MatrixXd Rt(1, 1);
  Rt << 0.01;
  auto linear_motion_model = LinearMotionModel::ConstPtr(new LinearMotionModel(A, B, Rt));

  Eigen::MatrixXd C(1, 1);
  C << 1;
  Eigen::MatrixXd Qt(1, 1);
  Qt << 1;
  auto linear_measurement_model = LinearMeasurementModel::ConstPtr(new LinearMeasurementModel(C, Qt));

  Eigen::VectorXd x0(1);
  x0 << 0;
  Eigen::MatrixXd covariance0(1, 1);
  covariance0 << 1;

  KalmanFilter kalman_filter(x0, covariance0, linear_motion_model, linear_measurement_model);
  InformationFilter information_filter(x0, covariance0, linear_motion_model, linear_measurement_model);

  std::ofstream kf_output_file("../data/kf_output.data");
  std::ofstream if_output_file("../data/if_output.data");

  for (size_t i = 0; i < data_vector.size(); ++i)
  {
    Eigen::VectorXd z(1);
    z << data_vector[i].z;
    kalman_filter.correct(z);
    information_filter.correct(z);

    if (i > 0)
    {
      Eigen::VectorXd u(1);
      u << data_vector[i-1].u;
      kalman_filter.predict(u);
      information_filter.predict(u);
    }

    kf_output_file << (kalman_filter.getX())(0, 0) << " " << data_vector[i].real_x << " " << data_vector[i].z << std::endl;
    if_output_file << (information_filter.getX())(0, 0) << " " << data_vector[i].real_x << " " << data_vector[i].z << std::endl;
  }
  kf_output_file.close();
  if_output_file.close();

  return 0;
}
