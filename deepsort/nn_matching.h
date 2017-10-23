#ifndef _NNMATCHINGH_
#define _NNMATCHINGH_
#include <vector>
#include "Detection.h"
#include <Eigen>
#include <map>
#include <sys/time.h>

#ifdef USETBB
#include <tbb.h>
#endif
#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>


static int64_t nn_gtm() {
	struct timeval tm;
	gettimeofday(&tm, 0);
	int64_t re = ((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec;
	return re;
}
Eigen::VectorXf _nn_cosine_distance(const FEATURESS &x, 
	const FEATURESS &y){
	int64_t nntm1 = nn_gtm();	
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
	int64_t nntm2 = nn_gtm();	
	std::cout << "_nn_cosine_distance(" << x.rows() << "," << y.rows() << ")----nntm2-nntm1:" << (nntm2-nntm1) << "\n";
	//std::cout << "re---b\n" << re << "\nre----e\n" << std::endl;
	return re;
}

class NearestNeighborDistanceMetric{
private:
    static boost::shared_ptr<NearestNeighborDistanceMetric> self_;
    float matching_threshold_ = 0;
    int budget_ = 0;
    std::map<int, std::vector<FEATURE> > samples_;
public:
	float matching_threshold() {
		return matching_threshold_;
	}
    static boost::shared_ptr<NearestNeighborDistanceMetric> Instance(){
        if(self_.get() == NULL){
            self_.reset(new NearestNeighborDistanceMetric());
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
#if 1
			std::vector<FEATURE>::iterator ii = it->second.begin();
			if(it->second.size() > budget_){
				it->second.erase(ii);
			}
#else
            /*if(samples_.size() > budget_){
                samples_.erase(samples_.begin());
            }*/
#endif
        }
    }
    struct DDS{
    public:
	void Push(int pos, const Eigen::VectorXf &dd){
		boost::mutex::scoped_lock lock(mutex_);
		dds_.push_back(
			std::make_pair(pos, dd)
		);
	}
	void Get(std::vector<std::pair<int, Eigen::VectorXf> > &dds){
		dds = dds_;
	}
    private:
	std::vector<std::pair<int, Eigen::VectorXf> > dds_;
	boost::mutex mutex_;
    };

    DYNAMICM distance(const FEATURESS &features, const IDS &ids){
#ifdef USETBB
	static DYNAMICM cost_matrix;
	cost_matrix = DYNAMICM(ids.size(), features.rows());
	int64_t dtm0 = nn_gtm();

	using namespace tbb;
 	parallel_for( blocked_range<size_t>(0,ids.size()),
                [=](const blocked_range<size_t> &r){
                        for(int i = r.begin(); i != r.end(); ++i){
 			        int iid = ids[i];
				std::vector<FEATURE> &ftsvec = samples_[iid];
				FEATURESS fts(ftsvec.size(), 128);
				for (int k = 0; k < ftsvec.size(); k++) {
					fts.row(k) = ftsvec[k];
				}
				int64_t dtm1 = nn_gtm();
				cost_matrix.row(i) = _nn_cosine_distance(fts, features);
				int64_t dtm2 = nn_gtm();
				std::cout << "distance(" << iid<< ")----dtm2-dtm1:" << (dtm2-dtm1) << "\n";
                        }
                   }
                );
#else
	static DYNAMICM cost_matrix;
	cost_matrix = DYNAMICM(ids.size(), features.rows());
	int64_t dtm0 = nn_gtm();
	DDS dds;
	#pragma omp parallel for
        for(int i = 0; i < ids.size(); i++){
            int iid = ids[i];
			std::vector<FEATURE> &ftsvec = samples_[iid];
			FEATURESS fts(ftsvec.size(), 128);
			for (int k = 0; k < ftsvec.size(); k++) {
				fts.row(k) = ftsvec[k];
			}
			int64_t dtm1 = nn_gtm();
			//cost_matrix.row(i) = _nn_cosine_distance(fts, features);
			Eigen::VectorXf dd = _nn_cosine_distance(fts, features);
			dds.Push(i, dd);
			int64_t dtm2 = nn_gtm();
			std::cout << "distance(" << iid<< ")----dtm2-dtm1:" << (dtm2-dtm1) << "\n";
        }
	std::vector<std::pair<int, Eigen::VectorXf>> vec;
	dds.Get(vec);
	for(int i = 0; i < vec.size(); i++){
		std::pair<int, Eigen::VectorXf> pa = vec[i];
		cost_matrix.row(pa.first) =  pa.second;
	} 
#endif
		int64_t dtm4 = nn_gtm();
		std::cout << "distance----dtm4-dtm0:" << (dtm4-dtm0) << "\n";
		//std::cout << "\nb-haha\n" << cost_matrix << "\ne-haha\n";
		return cost_matrix;
    }
};
#endif


