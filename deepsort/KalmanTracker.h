#ifndef _KALMANTRACKERH_
#define _KALMANTRACKERH_
#include "FeatureGetter/FeatureGetter.h"
#include "kalman_filter.h"

enum TrackState{
    TS_NONE = 0,
    Tentative,
    Confirmed,
    Deleted
};

class KalmanTracker{
public:
	int time_since_update_ = 0;
	MEAN mean_;
	VAR covariance_;
	int track_id = 0;
	std::vector<FEATURE> features_;
private:
	int hits_ = 0;
	int age_ = 0;
	TrackState state_ = TS_NONE;
	int _n_init_;
	int _max_age_;

public:
    KalmanTracker(const MEAN &mean, 
		const VAR &covariance, 
		int tid, 
		int n_init, 
		int max_age,
        const FEATURE &feature, bool featureFull=false){
        
		mean_ = mean;
		covariance_ = covariance;
		this->track_id = tid;
		hits_ = 1;
		age_ = 1;
		time_since_update_ = 0;

		state_ = Tentative;
		if (featureFull) {
			features_.push_back(feature);
		}

		_n_init_ = n_init;
		_max_age_ = max_age;
    }
    DSBOX to_tlwh() const{
		DSBOX ret;
		ret(0) = mean_(0);
		ret(1) = mean_(1);
		ret(2) = mean_(2);
		ret(3) = mean_(3);
		ret(2) *= ret(3);
		ret(0) -= ret(2) / 2;
		ret(1) -= ret(3) / 2;
		return ret;
    }
    DSBOX to_tlbr(){
		DSBOX ret = to_tlwh();
		ret(2) = ret(0) + ret(2);
		ret(3) = ret(1) + ret(3);
		return ret;
    }
    void predict(const KF &kalmanFilter){
		std::pair<MEAN, VAR> pa = kalmanFilter.predict(mean_, covariance_);
		mean_ = pa.first;
		covariance_ = pa.second;
		age_ += 1;
		time_since_update_ += 1;
    }
    void update(const KF &kalmanFilter, const Detection &detection){
		DSBOX box = detection.to_xyah();
		std::pair<MEAN, VAR> pa = kalmanFilter.update(
			mean_, covariance_, box);
		mean_ = pa.first;
		covariance_ = pa.second;
		features_.push_back(detection.feature_);

		hits_ += 1;
		time_since_update_ = 0;
		if (state_ == Tentative && hits_ >= _n_init_) {
			state_ = Confirmed;
		}
    }
    void mark_missed(){
        if(state_ == Tentative){
            state_ = Deleted;
        }
        else if(time_since_update_ > _max_age_){
            state_ = Deleted;
        }
    }
    bool is_tentative(){
		return state_ == Tentative;
    }
    bool is_confirmed()const {
		return state_ == Confirmed;
    }

    bool is_deleted(){
		return state_ == Deleted;
    }
};
#endif


