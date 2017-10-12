#ifndef _NNMATCHINGH_
#define _NNMATCHINGH_
#include <vector>
#include "Detection.h"
#include <Eigen>
#include <map>

Eigen::VectorXf _nn_cosine_distance(const FEATURESS &x, 
	const FEATURESS &y){
	
	FEATURESS a = x;
	FEATURESS b = y;
	//std::cout << "a---b\n" << a << "\na----e\n" << std::endl;
	//std::cout << "b---b\n" << b << "\nb----e\n" << std::endl;

	for (int row = 0; row < a.rows(); row++) {
		auto t = a.row(row);
		t = t / t.norm();
		a.row(row) = t;
	}
	for (int row = 0; row < b.rows(); row++) {
		auto t = b.row(row);
		t = t / t.norm();
		b.row(row) = t;
	}
	//std::cout << "a---b\n" << a << "\na----e\n" << std::endl;
	//std::cout << "b---b\n" << b << "\nb----e\n" << std::endl;
	auto tmp = a*b.transpose();
	auto tmp1 = tmp.array();
	auto tmp2 = -(tmp1 - 1);
	DYNAMICM distances = tmp2.matrix();
	Eigen::VectorXf re(distances.cols());
#ifdef WIN32
	auto rea = re.array();
	for (int col = 0; col < distances.cols(); col++) {
		auto cc = distances.col(col);
		float min = cc.minCoeff();
		rea.row(col) = min;
	}
	re = rea.matrix();
#else
	for (int col = 0; col < distances.cols(); col++) {
		auto cc = distances.col(col);
		float min = cc.minCoeff();
		re(col) = min;
	}
#endif
	//std::cout << "re---b\n" << re << "\nre----e\n" << std::endl;
	return re;
}

class NearestNeighborDistanceMetric{
private:
    static NearestNeighborDistanceMetric *self_;
    float matching_threshold_ = 0;
    int budget_ = 0;
    std::map<int, std::vector<FEATURE> > samples_;
public:
	float matching_threshold() {
		return matching_threshold_;
	}
    static NearestNeighborDistanceMetric *Instance(){
        if(self_ == NULL){
            self_ = new NearestNeighborDistanceMetric();
        }
        return self_;
    }
    void Init(float matching_threshold, int budget){
		matching_threshold_ = matching_threshold;
		budget_ = budget;
    }
    
    void partial_fit(const FEATURESS &features, 
                        const IDS &ids, 
                        const IDS &active_ids){
        //samples_.clear();
        for(int i = 0; i < features.rows(); i++){
            FEATURE feature = features.row(i);
            int iid = ids[i];
            //
            {
                bool isIn = false;
                for(int k = 0; k < active_ids.size(); k++){
                    if(iid == active_ids[k]){
                        isIn = true;
                        break;
                    }
                }
                if(!isIn){
                    continue;
                }
            }
            //
            std::map<int, std::vector<FEATURE>>::iterator it = samples_.find(iid);
            if(it == samples_.end()){
                std::vector<FEATURE> tmps;
                samples_.insert(std::make_pair(iid, tmps));
                it = samples_.find(iid);
            }
			it->second.push_back(feature);
            if(samples_.size() > budget_){
                samples_.erase(samples_.begin());
            }
        }
    }
    DYNAMICM distance(const FEATURESS &features, const IDS &ids){
		DYNAMICM cost_matrix = DYNAMICM(ids.size(), features.rows());
        for(int i = 0; i < ids.size(); i++){
            int iid = ids[i];
			std::vector<FEATURE> &ftsvec = samples_[iid];
			FEATURESS fts(ftsvec.size(), 128);
			for (int k = 0; k < ftsvec.size(); k++) {
				fts.row(k) = ftsvec[k];
			}
			cost_matrix.row(i) = _nn_cosine_distance(fts, features);
        }
		//std::cout << "\nb-haha\n" << cost_matrix << "\ne-haha\n";
		return cost_matrix;
    }
};
#endif


