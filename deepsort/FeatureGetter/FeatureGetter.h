#ifndef _FEATUREGETTERH_
#define _FEATUREGETTERH_

#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen>
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FFEATURE;





class FeatureGetter {
private:
	static boost::shared_ptr<FeatureGetter> self_;
public:
	static boost::shared_ptr<FeatureGetter> Instance() {
		if (self_.get() == NULL) {
			self_.reset(new FeatureGetter());
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
