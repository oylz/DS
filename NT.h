#ifndef _NTH_
#define _NTH_
#include <opencv2/opencv.hpp>

class TTracker;
class NT{
public:
	NT();
	~NT();
	bool Init();
	void Update(const cv::Mat &frame, const std::vector<cv::Rect> &rcs, int num);
	std::map<int, cv::Rect> Get();
private:
	TTracker *tt_;
};
#endif
