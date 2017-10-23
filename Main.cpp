#include <opencv2/opencv.hpp>
#include "NT.h"
#include "StrCommon.h"

NT *_tt = NULL;


void DrawData(cv::Mat mm, cv::Mat frame, const std::map<int, DSResult> &map, 
		const std::vector<cv::Rect> &outRcs,
		bool detect){
	std::map<int, DSResult>::const_iterator it;
	for(it = map.begin(); it != map.end(); ++it){
		CvScalar clr = cvScalar(0, 255, 0);
		cv::Rect rc = it->second.rc_;
    		cv::rectangle(frame, rc, clr);
		std::string disp = toStr(it->first);
		cv::putText(frame, 
			disp, 
			cvPoint(rc.x, rc.y), 
			CV_FONT_HERSHEY_SIMPLEX, 
			0.6, 
			cv::Scalar(0, 0, 255));
	}
	//
	CvScalar clr = cvScalar(0, 0, 255);
	for(cv::Rect rc:outRcs){
		cv::rectangle(mm, rc, clr);	
		if(detect){
			std::string disp = "detect";
			cv::putText(frame, 
				disp, 
				cvPoint(100, 100), 
				CV_FONT_HERSHEY_SIMPLEX, 
				1, 
				cv::Scalar(0, 0, 255));
		}
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
cv::VideoWriter *_vw = NULL;
bool _isShow = false;
int _imgCount = 0;

void CB(cv::Mat &frame, int num){
	if (_vw == NULL) {
		_vw = new cv::VideoWriter("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(frame.cols, frame.rows));
	}
	if (_rcMap.empty()) {
		ReadRcFileTotal(_rcFile);
	}
	std::vector<cv::Rect> rcs;
	std::map<int, std::vector<cv::Rect>>::iterator it = _rcMap.find(num);
	if (it != _rcMap.end()) {
		rcs = it->second;
	}
	std::vector<cv::Rect> outRcs;
	
	int64_t tm0 = gtm();
	std::map<int, DSResult> map = _tt->UpdateAndGet(frame, rcs, num, outRcs);
	int64_t tm1 = gtm();
	Mat mm = frame.clone();
	bool detect = (!rcs.empty());
	DrawData(mm, frame, map, outRcs, detect);
	printf("finish %d frame, updatecasttime:%ld\n", num, tm1-tm0);
	//(*_vw) << frame;
	if(_isShow){
		std::string disp = "frame";
		cv::resize(mm, mm, cv::Size(mm.cols/2, mm.rows/2));
		cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));
		cv::imshow("mm", mm);
		cv::imshow(disp, frame);
		cv::waitKey(1);
	}
}

void Go() {
	std::string root = _imgDir;
	for (int i = 1; i < _imgCount; i++) {
		std::string path = root;
		path += to6dStr(i);
		path += ".jpg";
		cv::Mat mat = cv::imread(path);
		CB(mat, i);
	}

}

int main(int argc, char **argv){
	if (argc < 2) {
		printf("usage:\n./tt showornot(0/1)\n");
		return 0;
	}
	_isShow = toInt(argv[1]);
	_tt = new NT();
	if(!_tt->Init()){
		return 0;
	}

	//_imgDir = "e:/code/deep_sort-master/MOT16/tt/xyz/img1/";
	//_rcFile = "e:/code/deep_sort-master/MOT16/tt/xyz/det/det.txt";
	_imgDir = "/home/xyz/code1/xyz/img1/";
	_rcFile = "/home/xyz/code1/xyz/det/det.txt";
	//_rcFile = "/home/xyz/code/test/pp/FaceNumGetter/out/102.txt";


	//_imgDir = "/home/xyz/code1/GEP/FrameBuffer/imglog/img1/";
	//_rcFile = "/home/xyz/code1/GEP/FrameBuffer/imglog/det/det.txt";
	_imgCount = 650;// 2001;// 750;// 680;
	Go();
	return 0;
}
