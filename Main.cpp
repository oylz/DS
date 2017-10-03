#include "./Tracker/Ctracker.h"
#include "StrCommon.h"
#include "Tracker/ds/LossMgr.h"

LossMgr *LossMgr::self_ = NULL;

CTracker *_tt = NULL;
void DrawTrack(cv::Mat frame,
                   const KalmanTracker& track
                   ){
	CvScalar clr = cvScalar(0, 255, 0);
	cv::Rect rc = track.GetLastRect();
		//printf("lastRc(%d, %d, %d, %d)\n", rc.x, rc.y, rc.width, rc.height);
        cv::rectangle(frame, rc, clr);
	std::string disp = toStr((int)track.m_trackID);
	cv::putText(frame, 
		disp, 
		cvPoint(rc.x, rc.y), 
		CV_FONT_HERSHEY_SIMPLEX, 
		0.6, 
		cv::Scalar(0, 0, 255));

}
void DrawData(cv::Mat frame){
		KalmanTrackers &kalmanTrackers =
			_tt->GetKalmanTrackersFor_MouseExample_MonitorDetect_FaceDetector_PedestrianDetector();

        for (const auto& track : kalmanTrackers){
            /*if (!track->IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    ){
			continue;
		}
		*/
            DrawTrack(frame, *track);
        }
}


void ReadFileContent(const std::string &file, std::string &content){
	FILE *fl = fopen(file.c_str(), "rb");
	if(fl == NULL){
		return;
	}
	fseek(fl, 0, SEEK_END);
	int len = ftell(fl);
	if(len <= 0){
		return;
	}
	fseek(fl, 0, SEEK_SET);
	char *buf = new char[len+1];
	memset(buf, 0, len+1);
	fread(buf, 1, len, fl);
	content = std::string(buf);
	delete []buf;
	fclose(fl);
}

std::map<int, std::vector<cv::Rect>> _rcMap;
void ReadRcFileTotal(const std::string &file) {
	std::string content = "";
	ReadFileContent(file, content);


	std::vector<std::string> lines;
	splitStr(content, "\n", lines);
	std::vector<cv::Rect> rcs;
	int num = -1;
	int tmpNum = -1;
	for (int i = 0; i < lines.size(); i++) {
		std::vector<std::string> cols;
		splitStr(lines[i], ",", cols);
		if (cols.size() < 6) {
			continue;
		}
		tmpNum = toInt(trim(cols[0]));
		if (num!=-1 && tmpNum!=num) {
			_rcMap.insert(std::make_pair(num, rcs));
			rcs.clear();
			num = tmpNum;
		}
		if (num == -1) {
			num = tmpNum;
		}
		cv::Rect rc;
		rc.x = toInt(trim(cols[2]));
		rc.y = toInt(trim(cols[3]));
		rc.width = toInt(trim(cols[4]));
		rc.height = toInt(trim(cols[5]));
		rcs.push_back(rc);
	}
	if (!rcs.empty()) {
		_rcMap.insert(std::make_pair(tmpNum, rcs));
	}
}
std::string _rcFile = "";
std::string _imgDir;
VideoWriter *_vw = NULL;
bool _isShow = false;
int _imgCount = 0;
std::vector<cv::Rect> _lastRcs;

void CB(Mat &frame, int num){
	if (_vw == NULL) {
		_vw = new VideoWriter("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(frame.cols, frame.rows));
	}

	if (_rcMap.empty()) {
		ReadRcFileTotal(_rcFile);
	}
	std::vector<cv::Rect> rcs;
	std::map<int, std::vector<cv::Rect>>::iterator it = _rcMap.find(num);
	if (it != _rcMap.end()) {
		rcs = it->second;
	}
	Mat mm;
	{
		mm = frame.clone();
		for(int i = 0; i < rcs.size(); i++){
			cv::Rect rc = rcs[i];
			cv::rectangle(mm, rc, cvScalar(0, 255, 0));
		}
	}


        std::vector<Point_t> centers;
        regions_t regions;
        for (auto rect : rcs)
        {
            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }
	//Mat grayFrame;
	//cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    _tt->Update(centers, regions, frame);
	DrawData(frame);
	(*_vw) << frame;
	if(_isShow){
		std::string disp = "frame";
		resize(mm, mm, Size(mm.cols/2, mm.rows/2));
		resize(frame, frame, Size(frame.cols/2, frame.rows/2));
		imshow("mm", mm);
		imshow(disp, frame);
		waitKey(1);
	}
}

void Go() {
	std::string root = _imgDir;
	for (int i = 1; i < _imgCount; i++) {
		std::string path = root;
		path += to6dStr(i);
		path += ".jpg";
		cv::Mat mat = imread(path);
		CB(mat, i);
		printf("finish %d frame\n", i);
	}

}

int main(int argc, char **argv){
	if (argc < 2) {
		printf("usage:\n./tt showornot(0/1)\n");
		return 0;
	}
	_isShow = toInt(argv[1]);
	LossMgr::Instance()->Init(_isShow, 0.72);

	TrackCtrl tc(DT_DistJaccard,
		KT_TypeUnscented,//KT_TypeUnscented,//KT_TypeLinear,//KT_TypeUnscented,
		0.3f, // Delta time for Kalman filter
		0.1f, // Accel noise magnitude for Kalman filter
		0.8f, // Distance threshold between region and object on two frames
		1 // Maximum allowed skipped frames
	);

	_tt = new CTracker(tc);


	//_imgDir = "e:/code/deep_sort-master/MOT16/ff/fr/img1/";
	//_rcFile = "e:/code/deep_sort-master/MOT16/ff/fr/det/det.txt";
	_imgDir = "e:/code/deep_sort-master/MOT16/tt/xyz/img1/";
	_rcFile = "e:/code/deep_sort-master/MOT16/tt/xyz/det/det.txt";
	_imgCount = 680;// 750;// 680;
	Go();
	return 0;
}
