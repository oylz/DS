#include "NT.h"
#include "deepsort/FeatureGetter/FeatureGetter.h"
#include "./deepsort/tracker.h"
#include "StrCommon.h"
#include "fdsst/fdssttracker.hpp"



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
	tt_ = TTrackerP(new TTracker(0.7, 30, 1));
}

NT::~NT(){
}
bool NT::Init(){
	if(!FeatureGetter::Instance()->Init()){
		return false;
	}
	KF::Instance()->Init();
#ifdef UBC
		Mat frame = cv::imread("/home/xyz/code1/xyz/img1/000001.jpg");
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
NewAndDelete NT::UpdateDS(const cv::Mat &frame, const std::vector<cv::Rect> &rcs, int num, const std::vector<int> &oriPos){
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
			//printf("oriPos.size():%d\n", oriPos.size());
			if(i < (int)oriPos.size()-1){
				det.oriPos_ = oriPos[i];
			}
			dets.push_back(det);
		}
   	 	NewAndDelete nad = tt_->update(dets);
		int64_t tm3 = gtm();
		std::string tail = "";
		if(tm3-tm1 > 30000){
			tail = "****";
		}

		std::cout << num << "----rcs.size():" << rcs.size() << "[tm1:" << tm1 << ",tm2:" << tm2 << "("<< (tm2 - tm1) << ")"<< ",tm3:"
			<< tm3 << "(" << (tm3-tm1) << ")]" << tail.c_str() << "\n";
		return nad;
}



// for framebuffer
void NT::UpdateFDSST(const Mat &frame, std::vector<cv::Rect> &rcs){
	std::map<int, FDSSTTrackerP>::iterator it;
	std::vector<int> lostIds;
	for(it = fdssts_.begin(); it != fdssts_.end(); ++it){
		//FDSSTTrackerP fdsst = it->second;
		cv::Rect rc = it->second->update(frame);
		int ww = frame.cols;
		int hh = frame.rows;
		int min = 8;
                if(rc.x<0 || rc.y<0 ||
                        (rc.x+rc.width)>ww ||
                        (rc.y+rc.height)>hh ||
                        rc.width<=min || rc.height<=min){
			lostIds.push_back(it->first);
			continue;
		}
		rcs.push_back(rc);
	}
	// remove
	for(int id:lostIds){
		fdssts_.erase(id);
	}
}
std::map<int, DSResult> NT::UpdateAndGet(const cv::Mat &frame, 
	const std::vector<cv::Rect> &rcsin, 
	int num,
	std::vector<cv::Rect> &outRcs,
	const std::vector<int> &oriPos){
	std::vector<cv::Rect> rcs = rcsin;
        
	//{	
	Mat ff;
        cvtColor(frame, ff, cv::COLOR_RGB2GRAY);
	//}
	if(!rcsin.empty()){
		fdssts_.clear();
	}	
	else{
		UpdateFDSST(ff, rcs);
	}
	outRcs = rcs;
	NewAndDelete nad = UpdateDS(frame, rcs, num, oriPos);


	std::map<int, DSResult> map;
	std::vector<KalmanTracker*> &kalmanTrackers =
			tt_->kalmanTrackers_;
	
    	for (const auto& track : kalmanTrackers){
		int id = (int)track->track_id;
		printf("trackid:%d, is_confirmed:%d, time_since_update:%d\n", id, track->is_confirmed(), track->time_since_update_);
		//if (!track->is_confirmed() || track->time_since_update_ > 0) {
		//	continue;
		//}
		if(track->time_since_update_ > 0){
			continue;
		}
		DSBOX box = track->to_tlwh();
		cv::Rect rc;
		rc.x = box(0);
		rc.y = box(1);
		rc.width = box(2);
		rc.height = box(3);
		int oriPos= track->oriPos_;
		DSResult tr;
		tr.rc_ = rc;
		tr.oriPos_ = oriPos;
		if(!rcsin.empty()){
			FDSSTTrackerP fdsst(new FDSSTTracker());
			printf("id:%d, oriPos:%d, rcsin.size():%d, rcs.size():%d\n", id, oriPos, rcsin.size(), rcs.size());
			/*if(rc.x<0)rc.x = 0;
			if(rc.y<0)rc.y = 0;
			int ww = frame.cols;
			int hh = frame.rows;
			if(rc.x+rc.width>ww)rc.width=ww-rc.x;
			if(rc.y+rc.height>hh)rc.height=hh-rc.y;

			if(rc.x>=0 && rc.y>=0 && rc.width>5 && rc.height>5){
			*/
			fdsst->init(rc, ff);
			fdssts_.insert(std::make_pair(id, fdsst));
			
		}
		if (!track->is_confirmed() || track->time_since_update_ > 0) {
			continue;
		}

		map.insert(std::make_pair(id, tr));
    	}
	return map;
}





