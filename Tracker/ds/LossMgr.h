#ifndef _LOSSMGRH_
#define _LOSSMGRH_
#include <opencv2/opencv.hpp>
#include "FeatureGetter.h"

typedef std::vector<FEATURE> FEATUREVEC;
typedef std::vector<cv::Mat> FACES;
class LossMgr {
private:
	static LossMgr *self_;
	FeatureGetter fg_;
	std::map<int, FEATUREVEC> lossMap_;
	bool isShow_ = false;
	double threshold_ = 0.66;
	std::map<int, FACES> tmpMap_;
	int tmpMaxW_ = 0;
	int tmpMaxH_ = 0;
	int tmpCount_ = 0;
public:
	static LossMgr *Instance() {
		if (self_ == NULL) {
			self_ = new LossMgr();
		}
		return self_;
	}
	void Init(bool isShow, double threshold) {
		isShow_ = isShow;
		threshold_ = threshold;
		fg_.Init();
	}
	void PreAddLoss() {
		tmpMap_.clear();
		tmpMaxW_ = 0;
		tmpMaxH_ = 0;
		tmpCount_ = 0;

	}
	void AddLoss(int id, const FACES &faces) {
		tmpMap_.insert(std::make_pair(id, faces));
		for (const cv::Mat &face : faces) {
			int w = face.cols;
			int h = face.rows;
			if (w > tmpMaxW_) {
				tmpMaxW_ = w;
			}
			if (h > tmpMaxH_) {
				tmpMaxH_ = h;
			}
			tmpCount_++;
		}
	}
	void CommitAddLoss() {
		if (tmpMap_.empty()) {
			return;
		}
		tmpMaxW_ += 10;
		tmpMaxH_ += 10;

		// prepare whole frame and rcs
		std::vector<cv::Rect> rcs;
		std::map<int, int> posIdMap;
		cv::Mat frame(tmpMaxH_, tmpMaxW_*tmpCount_, CV_8UC3);
		int pos = 0;
		std::map<int, FACES>::iterator tit;
		for (tit = tmpMap_.begin(); tit != tmpMap_.end(); ++tit) {
			for (cv::Mat &face : tit->second) {
				cv::Rect rc = cv::Rect(pos*tmpMaxW_+5, 5, face.cols, face.rows);
				rcs.push_back(rc);
				Mat tmp = frame(rc);
				face.copyTo(tmp);
				posIdMap.insert(std::make_pair(pos, tit->first));
				pos++;
			}
		}
		if (isShow_) {
			//cv::imshow("allframe", frame);
		}
		// do
		FEATUREVEC fts;
		fg_.Get(frame, rcs, fts);
		//
		for (int i = 0; i < fts.size(); i++) {
			int id = posIdMap.at(i);
			std::map<int, FEATUREVEC>::iterator lit = lossMap_.find(id);
			if (lit == lossMap_.end()) {
				FEATUREVEC ftvec;
				lossMap_.insert(std::make_pair(id, ftvec));
				lit = lossMap_.find(id);
			}
			lit->second.push_back(fts[i]);
		}
	}
	int GetSimilaryId(const cv::Mat &face) {
		if (lossMap_.empty()) {
			return -1;
		}
		// get ft
		std::vector<cv::Rect> rcs;
		cv::Rect rc(0, 0, face.cols - 1, face.rows - 1);
		rcs.push_back(rc);
		FEATUREVEC fts;
		fg_.Get(face, rcs, fts);
		FEATURE &ft = fts[0];

		//
		std::map<int, FEATUREVEC>::iterator it;
		double maxScore = 0;
		int maxId = -1;
		for (it = lossMap_.begin(); it != lossMap_.end(); ++it) {
			for (FEATURE &in : it->second) {
				double score = this->Score(in, ft);
				if (score < 0) {
					score = -score;
				}
				if (score > maxScore) {
					maxScore = score;
					maxId = it->first;
				}
				printf("id:%d, score:%lf\n", it->first, score);
			}
		}
		//
		
		if (maxScore > threshold_) {
			printf("*********maxId:%d, maxScore:%lf\n", maxId, maxScore);
			return maxId;
		}
		return -1;
	}
	void RemoveLoss(int id) {
		lossMap_.erase(id);
	}
private:
	static double Score(const FEATURE &feature1, const FEATURE &feature2) {
		cv::Mat mat1(feature1);
		cv::Mat mat2(feature2);

		double ab = mat1.dot(mat2);
		double aa = mat1.dot(mat1);
		double bb = mat2.dot(mat2);
		double score = -ab / sqrt(aa*bb);
		return score;
	}
};
#endif
