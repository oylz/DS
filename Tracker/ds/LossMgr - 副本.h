#ifndef _LOSSMGRH_
#define _LOSSMGRH_
#include <opencv2/opencv.hpp>
#include "FeatureGetter.h"

class LossMgr {
private:
	static LossMgr *self_;
	FeatureGetter fg_;
	std::map<int, FEATURE> lossMap_;
	bool isShow_ = false;
	double threshold_ = 0.66;
	std::map<int, cv::Rect> tmpMap_;
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
	}
	void CommitAddLoss(const cv::Mat &frame) {
		if (tmpMap_.empty()) {
			return;
		}
		std::vector<cv::Rect> rcs;
		std::vector<FEATURE> fts;
		int i = 0;
		std::map<int, cv::Rect>::iterator it;
		for(it = tmpMap_.begin(); it != tmpMap_.end(); ++it){
			cv::Rect rc = it->second;
			if (isShow_) {
				cv::imshow(toStr(i++), frame(rc));
			}

			rcs.push_back(rc);
		}
		fg_.Get(frame, rcs, fts);

		//waitKey();
		int pos = 0;
		for (it = tmpMap_.begin(); it != tmpMap_.end(); ++it) {
			int id = it->first;
			std::map<int, FEATURE>::iterator itL = lossMap_.find(id);
			if (itL == lossMap_.end()) {
				lossMap_.insert(std::make_pair(id, fts[pos]));
				pos++;
				continue;
			}
			printf("error!!!");
			itL->second.clear();
			std::copy(fts[pos].begin(), fts[pos].end(), std::back_inserter(itL->second));
			pos++;
		}
	}
	void AddLoss(int id, const cv::Rect &rc) {
		tmpMap_.insert(std::make_pair(id, rc));
	}
	int GetSimilaryId(const cv::Mat &face) {
		if (lossMap_.empty()) {
			return -1;
		}
		// get ft
		std::vector<cv::Rect> rcs;
		cv::Rect rc(0, 0, face.cols - 1, face.rows - 1);
		rcs.push_back(rc);
		std::vector<FEATURE> fts;
		fg_.Get(face, rcs, fts);
		//
		FEATURE &ft = fts[0];
		std::map<int, FEATURE>::iterator it;
		double maxScore = 0;
		int maxId = -1;
		for (it = lossMap_.begin(); it != lossMap_.end(); ++it) {
			FEATURE &in = it->second;
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
