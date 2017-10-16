#ifndef _FEATUREGETTERH_
#define _FEATUREGETTERH_


#include <opencv2/opencv.hpp>
#include <Eigen>
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FFEATURE;





class FeatureGetter {
private:
	static FeatureGetter *self_;
public:
	static FeatureGetter *Instance() {
		if (self_ == NULL) {
			self_ = new FeatureGetter();
		}
		return self_;
	}
	bool Init(); 
	bool Get(const cv::Mat &img, const std::vector<cv::Rect> &rcs,
		std::vector<FFEATURE> &fts);

public:
	~FeatureGetter() {
	}

};

#endif
