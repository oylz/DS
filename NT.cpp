#include "NT.h"

//#define UDL
#ifdef UDL
	//#define UBC
	#include "deepsort/FeatureGetter/FeatureGetter.h"
#endif

#include "./deepsort/tracker.h"
#include "StrCommon.h"
#include "fdsst/fdssttracker.hpp"
#include "fdsst/fhog.h"
#include <boost/thread/mutex.hpp>


boost::shared_ptr<NearestNeighborDistanceMetric> NearestNeighborDistanceMetric::self_;
boost::shared_ptr<KF> KF::self_;

#define UHOG


void ExtractFeatureHog(const cv::Mat &in, 
	const std::vector<cv::Rect> &rcsin,
	std::vector<FEATURE> &fts){
	cv::Mat frame;
	cvtColor(in, frame, cv::COLOR_RGB2GRAY);
	for(int i = 0; i < rcsin.size(); i++){
		Mat nnn = frame(rcsin[i]);
		resize(nnn, nnn, Size(32, 32));
		int len = 0;
		float *hog = HOGXYZ(nnn, len);
		if(hog==NULL || len!=128){
			printf("hog(%d) is null or len(%d)!=128,exit!\n", hog==NULL, len);
			exit(0);
		}
		FEATURE ft;
		for(int j = 0; j < len; j++){
			ft(j) = hog[j];
		}
		delete []hog;
		fts.push_back(ft);
	}
}
#ifdef UDL
void ExtractFeature(const cv::Mat &in, 
	const std::vector<cv::Rect> &rcsin,
	std::vector<FEATURE> &fts) {
	int maxw = 0;
	int maxh = 0;
	int count = rcsin.size(); 
#ifdef UBC
	int BC = 1;
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
#endif

NT::NT(){
	tt_ = TTrackerP(new TTracker(0.7, 30, 1));
}

NT::~NT(){
}
bool NT::Init(){
#ifdef UDL
	if(!FeatureGetter::Instance()->Init()){
		return false;
	}
#endif
	if(0){// just a test
		Mat frame = cv::imread("/home/xyz/code1/xyz/img1/000001.jpg");
		Mat nnn;
       		cvtColor(frame, nnn, cv::COLOR_RGB2GRAY);
		resize(nnn, nnn, Size(32, 32));
		Mat a = fhog(nnn, 4, 9, 0.2f, false);
		std::cout << "a:cols:" << a.cols << "a:rows:" << a.rows << "\njust a test, exit\n";
		exit(0);
	}

	KF::Instance()->Init();
#ifdef UDL
	#ifdef UBC
		Mat frame = cv::imread("/home/xyz/code1/xyz/img1/000001.jpg");
		std::vector<Detection> dets;
		std::vector<FEATURE> fts;
		std::vector<cv::Rect> rcs;
		srand((unsigned)time(NULL));
		int width = frame.cols;
		int height = frame.rows;
		//for(int i = 0; i < 30; i++){
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
		//}
		ExtractFeature(frame, rcs, fts);
	#endif
#endif
	NearestNeighborDistanceMetric::Instance()->Init(0.2, 100);

	return true;
}
NewAndDelete NT::UpdateDS(const cv::Mat &frame, const std::vector<cv::Rect> &rcs, int num, const std::vector<int> &oriPos){
		int64_t tm1 = gtm();
		std::vector<Detection> dets;
		std::vector<FEATURE> fts;
		if(rcs.size() > 0){
#ifdef UHOG
			ExtractFeatureHog(frame, rcs, fts);
#else
			ExtractFeature(frame, rcs, fts);
#endif
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


struct RRS{
	void Push(const cv::Rect &rc){
		boost::mutex::scoped_lock lock(mutex_);
		rcs_.push_back(rc);
	}
	void Get(std::vector<cv::Rect> &rcs){
		rcs = rcs_;
	}
private:
	std::vector<cv::Rect> rcs_;
	boost::mutex mutex_;
};
struct FFS{
public:
	void Push(int id, const FDSSTTrackerP &ff){
		boost::mutex::scoped_lock lock(mutex_);
		ffs_.push_back(std::make_pair(id, ff));
	}
	void Get(std::vector<std::pair<int, FDSSTTrackerP> > &ffs){
		ffs = ffs_;
	}
private:
	std::vector<std::pair<int, FDSSTTrackerP > > ffs_;
	boost::mutex mutex_;
};

// for framebuffer
void NT::UpdateFDSST(const Mat &frame, std::vector<cv::Rect> &rcs){
	std::map<int, FDSSTTrackerP>::iterator it;
	std::vector<int> lostIds;
	RRS rrs;
	std::vector<FDSSTTrackerP> ffs;
	for(it = fdssts_.begin(); it != fdssts_.end(); ++it){
		FDSSTTrackerP fdsst = it->second;
		ffs.push_back(fdsst);
	}
	#pragma omp parallel for
	for(int i = 0; i < ffs.size(); i++){
		cv::Rect rc = ffs[i]->update(frame);
		rrs.Push(rc);
	}
	//
	std::vector<cv::Rect> rrcs;
	rrs.Get(rrcs);
	for(int i = 0; i < rrcs.size(); i++){
		cv::Rect rc = rrcs[i];
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
		rcs.push_back(ToOriRect(rc));
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
	Mat ffMat;
        cvtColor(frame, ffMat, cv::COLOR_RGB2GRAY);
	resize(ffMat, ffMat, Size(ffMat.cols*scale_, ffMat.rows*scale_));
	std::cout << "NT::UpdateAndGet1\n";
	//}
	if(!rcsin.empty()){
		fdssts_.clear();
	}	
	else{
		UpdateFDSST(ffMat, rcs);
	}
	std::cout << "NT::UpdateAndGet1.5\n";
	outRcs = rcs;
	NewAndDelete nad = UpdateDS(frame, rcs, num, oriPos);


	std::map<int, DSResult> map;
	std::vector<KalmanTracker> &kalmanTrackers =
			tt_->kalmanTrackers_;

	std::cout << "NT::UpdateAndGet2\n";
	std::vector<std::pair<int, cv::Rect> > idrcs;	
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
			idrcs.push_back(std::make_pair(id, rc));	
			printf("id:%d, rc:(%d, %d, %d, %d), oriPos:%d, rcsin.size():%d, rcs.size():%d\n", 
					id, rc.x, rc.y, rc.width, rc.height,
					oriPos, rcsin.size(), rcs.size());
			
		}
		if (!track->is_confirmed() || track->time_since_update_ > 0) {
			continue;
		}

		map.insert(std::make_pair(id, tr));
    	}
	std::cout << "NT::UpdateAndGet3\n";
	FFS ffs;
	#pragma omp parallel for
	for(int i = 0; i < idrcs.size(); i++){
		std::pair<int, cv::Rect> pa = idrcs[i];
		int id = pa.first;
		cv::Rect rc = pa.second;
		printf("id:%d, rc:(%d, %d, %d, %d)\n", 
				id, rc.x, rc.y, rc.width, rc.height);

		FDSSTTrackerP fdsst(new FDSSTTracker());
		fdsst->init(ToScaleRect(rc), ffMat);
		ffs.Push(id, fdsst);
	}
	std::cout << "NT::UpdateAndGet4\n";
	std::vector<std::pair<int, FDSSTTrackerP> > pps;
	ffs.Get(pps);
	for(int i = 0; i < pps.size(); i++){
		std::pair<int, FDSSTTrackerP> pa = pps[i];
		fdssts_.insert(pa);
	}
	return map;
}





