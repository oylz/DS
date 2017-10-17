#include "NT.h"
#include "deepsort/FeatureGetter/FeatureGetter.h"
#include "./deepsort/tracker.h"
#include "StrCommon.h"

NearestNeighborDistanceMetric *NearestNeighborDistanceMetric::self_ = NULL;
KF *KF::self_ = NULL;

#define UBC



void ExtractFeature(const cv::Mat &in, 
	const std::vector<cv::Rect> &rcsin,
	std::vector<FEATURE> &fts) {
	int maxw = 0;
	int maxh = 0;
	int count = rcsin.size(); 
#ifdef UBC
	int BC = 8;
	if(count < BC)count=BC;
#endif
	std::vector<cv::Mat> faces;
	cv::Rect lr;
	for (int i = 0; i < count; i++) {
		cv::Rect rc;
		if(i < rcsin.size()){
			rc = rcsin[i];
			lr = rc;
		}
		else{
			rc = lr;
		}
		faces.push_back(in(rc).clone());
		int w = rc.width;
		int h = rc.height;
		if (w > maxw) {
			maxw = w;
		}
		if (h > maxh) {
			maxh = h;
		}
	}
	maxw += 10;
	maxh += 10;

	cv::Mat frame(maxh, maxw*count, CV_8UC3);
	std::vector<cv::Rect> rcs;
	for (int i = 0; i < count; i++) {
		cv::Mat &face = faces[i];
		cv::Rect rc = cv::Rect(i*maxw + 5, 5, face.cols, face.rows);
		rcs.push_back(rc);
		cv::Mat tmp = frame(rc);
		face.copyTo(tmp);
	}
	std::vector<FEATURE> newfts;
	FeatureGetter::Instance()->Get(frame, rcs, newfts);
	for(int i = 0; i < rcsin.size(); i++){
		fts.push_back(newfts[i]);
	}
}

NT::NT(){
	tt_ = new TTracker();
}

NT::~NT(){
	if(tt_){
		delete tt_;
	}
}
bool NT::Init(){
	if(!FeatureGetter::Instance()->Init()){
		return false;
	}
	KF::Instance()->Init();
#ifdef UBC
		Mat frame = cv::imread("../xyz/img1/000001.jpg");
		std::vector<Detection> dets;
		std::vector<FEATURE> fts;
		std::vector<cv::Rect> rcs;
		srand((unsigned)time(NULL));
		int width = frame.cols;
		int height = frame.rows;
					int x = rand()%width;
					int y = rand()%height;
					int w = 100;
					int h = 100;
					//std::cout << x << "," << y  << "," << w  << "," << h << "\n";
					if(x+w > width){
						w = width - x;
					}
					if(y+h > height){
						h = height - y;
					}
					cv::Rect rc(x, y, w, h);	
					rcs.push_back(rc);
		ExtractFeature(frame, rcs, fts);
#endif
	NearestNeighborDistanceMetric::Instance()->Init(0.2, 100);

	return true;
}
void NT::Update(const cv::Mat &frame, const std::vector<cv::Rect> &rcs, int num){
		int64_t tm1 = gtm();
		std::vector<Detection> dets;
		std::vector<FEATURE> fts;
		if(rcs.size() > 0){
			ExtractFeature(frame, rcs, fts);
		}
		int64_t tm2 = gtm();
		for (int i = 0; i < rcs.size(); i++){	
			DSBOX box;
			cv::Rect rc = rcs[i];
			box(0) = rc.x;
			box(1) = rc.y;
			box(2) = rc.width;
			box(3) = rc.height;
			Detection det(box, 1, fts[i]);
			dets.push_back(det);
		}
   	 	tt_->update(dets);
		int64_t tm3 = gtm();
		std::string tail = "";
		if(tm3-tm1 > 30000){
			tail = "****";
		}

		std::cout << num << "----rcs.size():" << rcs.size() << "[tm1:" << tm1 << ",tm2:" << tm2 << "("<< (tm2 - tm1) << ")"<< ",tm3:"
			<< tm3 << "(" << (tm3-tm1) << ")]" << tail.c_str() << "\n";

}


std::map<int, cv::Rect> NT::Get(){
	std::map<int, cv::Rect> map;
	std::vector<KalmanTracker*> &kalmanTrackers =
	tt_->kalmanTrackers_;
	
    	for (const auto& track : kalmanTrackers){
		if (!track->is_confirmed() || track->time_since_update_ > 0) {
			continue;
		}
		DSBOX box = track->to_tlwh();
		cv::Rect rc;
		rc.x = box(0);
		rc.y = box(1);
		rc.width = box(2);
		rc.height = box(3);
		int id = (int)track->track_id;
		map.insert(std::make_pair(id, rc));
    	}
	return map;
}




