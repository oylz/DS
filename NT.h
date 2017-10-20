#ifndef _NTH_
#define _NTH_
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include "NTN.h"
using namespace cv;

struct DSResult{
	cv::Rect rc_;
	int oriPos_;
	DSResult(){
		rc_ = cv::Rect(0, 0, 0, 0);
		oriPos_ = -1;
	}
};

class TTracker;
typedef boost::shared_ptr<TTracker> TTrackerP;

class FDSSTTracker;
typedef boost::shared_ptr<FDSSTTracker> FDSSTTrackerP;


class NT{
public:
	NT();
	~NT();
	bool Init();

	// for framebuffer
	std::map<int, DSResult> UpdateAndGet(const cv::Mat &frame, 
		const std::vector<cv::Rect> &rcs, 
		int num,
		std::vector<cv::Rect> &outRcs, 
		const std::vector<int> &oriPos=std::vector<int>(0));
private:
	void UpdateFDSST(const Mat &frame, std::vector<cv::Rect> &rcs);	
	NewAndDelete UpdateDS(const cv::Mat &frame, 
		const std::vector<cv::Rect> &rcs, 
		int num,
		const std::vector<int> &oriPos);
private:
	cv::Rect ToOriRect(const cv::Rect &rc){
		cv::Rect re;
		float x = ((float)rc.x)/scale_;
		float y = ((float)rc.y)/scale_;
		float w = ((float)rc.width)/scale_;
		float h = ((float)rc.height)/scale_;
		re.x = x;
		re.y = y;
		re.width = w;
		re.height = h;
		return re;
	}
	cv::Rect ToScaleRect(const cv::Rect &rc){
		cv::Rect re;
		float x = ((float)rc.x)*scale_;
		float y = ((float)rc.y)*scale_;
		float w = ((float)rc.width)*scale_;
		float h = ((float)rc.height)*scale_;
		re.x = x;
		re.y = y;
		re.width = w;
		re.height = h;
		return re;
	}
private:
	TTrackerP tt_;
	std::map<int, FDSSTTrackerP> fdssts_;
	float scale_ = 0.25;
};
#endif


