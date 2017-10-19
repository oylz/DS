#ifndef _DETECTIONH_
#define _DETECTIONH_
#include <vector>
#include <Eigen>

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DSBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DSBOXS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, -1, 128, Eigen::RowMajor> FEATURESS;
typedef std::vector<int> IDS;
typedef Eigen::Matrix<float, 1, 2, Eigen::RowMajor> PT2;
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> VAR;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> NMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> NVAR;

struct Detection {
	DSBOX tlwh_;
	float confidence_;
	FEATURE feature_;
	int oriPos_ = -1;
	Detection(const DSBOX &tlwh, float confidence, const FEATURE &feature) {
		tlwh_ = tlwh;
		confidence_ = confidence;
		//std::cout << feature;
		feature_ = feature;
	}
	DSBOX to_tlbr() const{
		DSBOX ret = tlwh_;
		ret(0, 2) += ret(0, 0);
		ret(0, 3) += ret(0, 1);
		return ret;
	}
	DSBOX to_xyah() const{
		DSBOX ret = tlwh_;
		ret(0, 0) += ret(0, 2) / 2;
		ret(0, 1) += ret(0, 3) / 2;
		ret(0, 2) /= ret(0, 3);
		return ret;
	}
};
#endif
