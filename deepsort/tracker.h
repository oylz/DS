#ifndef _TTH_
#define _TTH_
#include "nn_matching.h"
#include "linear_assignment.h"
#include <algorithm>
#include <vector>
#include <iterator>
#include "iou_matching.h"
#include "FeatureGetter/FeatureGetter.h"
#include "../NTN.h"

DYNAMICM getCostMatrixByNND(const std::vector<KalmanTracker> &kalmanTrackers,
	const std::vector<Detection> &dets,
	IDS *track_indices,
	IDS *detection_indices);
class TTracker *p;

class TTracker{
public:
	std::vector<KalmanTracker> kalmanTrackers_;
private:
        float max_iou_distance_ = 0;
        int max_age_ = 0;
        int n_init_ = 0;
        
        
        int _next_id_ = 0;
public:    
    TTracker(float max_iou_distance=0.7, int max_age=30, int n_init=3){
        max_iou_distance_ = max_iou_distance;
        max_age_ = max_age;
        n_init_ = n_init;
		_next_id_ = 1;
		p = this;
    }

    NewAndDelete update(const std::vector<Detection> &detections){
	NewAndDelete re;

	int64_t uptm1 = line_gtm();
        for(KalmanTracker kalmanTrack : kalmanTrackers_){
            kalmanTrack->predict(*KF::Instance());
        }

	int64_t uptm2 = line_gtm();
        //# Run matching cascade.
        RR rr = this->_match(detections);

	int64_t uptm3 = line_gtm();
        //# Update track set.
        //# -matches
        for(int i = 0; i < rr.matches.size(); i++){
            std::pair<int, int> pa = rr.matches[i];
            int track_idx = pa.first;
            int detection_idx = pa.second;
            kalmanTrackers_[track_idx]->update(*KF::Instance(), 
                                detections[detection_idx]);
        }
        //# -unmatches(track)    
        for(int i = 0; i < rr.unmatched_tracks.size(); i++){
            int track_idx = rr.unmatched_tracks[i];
            kalmanTrackers_[track_idx]->mark_missed();
        }
        //# -unmatches(detect)    
        for(int i = 0; i < rr.unmatched_detections.size(); i++){
            int detection_idx = rr.unmatched_detections[i];
            int id = this->_NewTrack(detections[detection_idx]);
		re.news_.insert(std::make_pair(id, detections[detection_idx].oriPos_));
        }
        
	int64_t uptm4 = line_gtm();
        
		std::vector<KalmanTracker>::iterator it;
		while (1) {
			bool cn = false;
			for (it = kalmanTrackers_.begin(); it != kalmanTrackers_.end(); ++it) {
				KalmanTracker p = *it;
				if (p->is_deleted()) {
					re.deletes_.push_back(p->track_id);
					kalmanTrackers_.erase(it);
					//delete p;
					cn = true;
					break;
				}
			}
			if (cn) {
				continue;
			}
			break;
		}

        //# Update distance nearestNeighborDistanceMetric.
        IDS active_ids;
        for(KalmanTracker t : kalmanTrackers_){
            if(t->is_confirmed()){
                active_ids.push_back(t->track_id);
            }
        }
        
	int64_t uptm5 = line_gtm();
		int featureCount = 0;
        IDS ids;
        for(KalmanTracker t : kalmanTrackers_){
            if(!t->is_confirmed()){
                continue;
            }
			std::vector<FEATURE> &fts = t->features_;
			featureCount += fts.size();
			//ids += [kalmanTrack.track_id_ for _ in kalmanTrack.features_]
			// 就是这个意思 
			for (int kk = 0; kk < fts.size(); kk++) {
				ids.push_back(t->track_id);
			}
        }
		FEATURESS features(featureCount, 128);
		int pos = 0;
		for (KalmanTracker t : kalmanTrackers_) {
			if (!t->is_confirmed()) {
				continue;
			}
			std::vector<FEATURE> &fts = t->features_;
			for (int i = 0; i < fts.size(); i++) {
				FEATURE tt = fts.at(i);
				features.row(pos++) = tt;
			}
			t->features_.clear();
		}

	int64_t uptm6 = line_gtm();
		NearestNeighborDistanceMetric::Instance()->partial_fit(
			features, ids, active_ids);
	int64_t uptm7 = line_gtm();
	std::cout << "up----uptm2-uptm1:" << (uptm2-uptm1) << 
			", uptm3-uptm1:" << uptm3-uptm1 << 
			", uptm4-uptm1:" << (uptm4-uptm1) << 
			", uptm5-uptm1:" << (uptm5-uptm1) << 
			", uptm6-uptm1:" << (uptm6-uptm1) << "\n";
	return re;
    }
    
private:        
    RR _match(const std::vector<Detection> &detections){
	int64_t mtm1 = line_gtm();
        //Split track set into confirmed and unconfirmed kalmanTrackers.
        IDS confirmed_trackIds;
        IDS unconfirmed_trackIds;
        for(int i = 0; i < kalmanTrackers_.size(); i++){
            KalmanTracker t = kalmanTrackers_[i]; 
            if(t->is_confirmed()){
                confirmed_trackIds.push_back(i);
            }
            else{
                unconfirmed_trackIds.push_back(i);
            }
        }
        
        //# Associate confirmed kalmanTrackers using appearance features.
        RR rr = linear_assignment::matching_cascade(
                getCostMatrixByNND, 
                NearestNeighborDistanceMetric::Instance()->matching_threshold(), 
                max_age_,
                kalmanTrackers_, 
                detections, 
                &confirmed_trackIds);
        std::vector<std::pair<int, int> > matches_a = rr.matches;
        IDS unmatched_tracks_a = rr.unmatched_tracks;
        IDS unmatched_detections = rr.unmatched_detections;
        
	int64_t mtm2 = line_gtm();

        //# Associate remaining kalmanTrackers together with unconfirmed kalmanTrackers using IOU.
        IDS iou_track_candidateIds, tmp;
        std::copy(unconfirmed_trackIds.begin(), 
                    unconfirmed_trackIds.end(),
                    std::back_inserter(iou_track_candidateIds));
        for(int k = 0; k < unmatched_tracks_a.size(); k++){
            int id = unmatched_tracks_a[k];
            if(kalmanTrackers_[id]->time_since_update_ == 1){
                iou_track_candidateIds.push_back(id);
            }
            else{
                tmp.push_back(id);
            }
        }
        unmatched_tracks_a.clear();
        unmatched_tracks_a = tmp;
        
	int64_t mtm3 = line_gtm();
        //
        RR rr1 = linear_assignment::min_cost_matching(
                iou_matching::getCostMatrixByIOU, 
                max_iou_distance_, 
                kalmanTrackers_,
                detections, 
                &iou_track_candidateIds, 
                &unmatched_detections);
        std::vector<std::pair<int, int> > matches_b = rr1.matches;
        IDS unmatched_tracks_b = rr1.unmatched_tracks;
        unmatched_detections = rr1.unmatched_detections;
                
	int64_t mtm4 = line_gtm();
        // all
        RR re;
        re.matches = matches_a;
        std::copy(matches_b.begin(), matches_b.end(),
                    std::back_inserter(re.matches));
        re.unmatched_detections = unmatched_detections;
        re.unmatched_tracks = unmatched_tracks_a;
        std::copy(unmatched_tracks_b.begin(),
                    unmatched_tracks_b.end(),
                    std::back_inserter(re.unmatched_tracks));
	int64_t mtm5 = line_gtm();
	std::cout << "match----mtm2-mtm1:" << (mtm2-mtm1) << 
			", mtm3-mtm1:" << (mtm3-mtm1) << 
			", mtm4-mtm1:" << (mtm4-mtm1) <<
			", mtm5-mtm1:" << (mtm5-mtm1) << "\n";
        return re;
    }    

    int _NewTrack(const Detection &detection){
	int id = _next_id_;
        std::pair<MEAN, VAR>  pa = 
                    KF::Instance()->initiate(detection.to_xyah());
	KalmanTracker newt(new KalmanTrackerN(
            pa.first, pa.second, _next_id_, n_init_, max_age_,
            detection.feature_, true, detection.oriPos_));
        kalmanTrackers_.push_back(newt);/*new KalmanTracker(
            pa.first, pa.second, _next_id_, n_init_, max_age_,
            detection.feature_, true, detection.oriPos_));*/
        _next_id_ += 1;
	return id;
    }
};

DYNAMICM getCostMatrixByNND(const std::vector<KalmanTracker> &kalmanTrackers,
	const std::vector<Detection> &dets,
	IDS *track_indicesi,
	IDS *detection_indicesi) {
	int64_t gtm1 = line_gtm();
	IDS track_indices = *track_indicesi;
	IDS detection_indices = *detection_indicesi;
	FEATURESS features(detection_indices.size(), 128);
	for (int i = 0; i < detection_indices.size(); i++) {
		int pos = detection_indices[i];
		features.row(i) = dets[pos].feature_;
	}
	IDS ids;
	for (int i = 0; i < track_indices.size(); i++) {
		int pos = track_indices[i];
		ids.push_back(p->kalmanTrackers_[pos]->track_id);
	}
	DYNAMICM cost_matrix =
		NearestNeighborDistanceMetric::Instance()->distance(features, ids);
	int64_t gtm2 = line_gtm();
	cost_matrix = linear_assignment::gate_cost_matrix(
		*KF::Instance(), cost_matrix, kalmanTrackers, dets, track_indices,
		detection_indices);
	int64_t gtm3 = line_gtm();
	std::cout << "getCostMatrixByNND----gtm2-gtm1:" << (gtm2-gtm1) <<
			", gtm3-gtm1:" << (gtm3-gtm1) << "\n";
	return cost_matrix;
}
#endif











