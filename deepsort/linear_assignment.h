#ifndef _LASMH_
#define _LASMH_
#include <vector>
#include "Detection.h"
#include <Eigen>

#include "KalmanTracker.h"
#include "FeatureGetter/FeatureGetter.h"
#include "HungarianOper.h"

const static int INFTY_COST = 1e+5;
struct RR {
	std::vector<std::pair<int, int> > matches;
	IDS unmatched_detections;
	IDS unmatched_tracks;
};

typedef DYNAMICM (*GetCostMarixFun)(const std::vector<KalmanTracker*> &tracks,
	const std::vector<Detection> &detections,
	IDS *track_indices,
	IDS *detection_indices);

double chi2inv95[10] = {
	0,
	3.8415,
	5.9915,
	7.8147,
	9.4877,
	11.070,
	12.592,
	14.067,
	15.507,
	16.919 };

class linear_assignment{
public:
    static RR min_cost_matching(
            const GetCostMarixFun &getCostMarixFun, float max_distance,
        const std::vector<KalmanTracker*> &tracks, 
        const std::vector<Detection> &detections, 
        IDS *track_indicesi=NULL,
        IDS *detection_indicesi=NULL){
		IDS track_indices;
		IDS detection_indices;
		if (track_indicesi == NULL) {
			for (int i = 0; i < tracks.size(); i++) {
				track_indices.push_back(i);
			}
		}
		else {
			track_indices = *track_indicesi;
		}

		if (detection_indicesi == NULL) {
			for (int i = 0; i < detections.size(); i++) {
				detection_indices.push_back(i);
			}
		}
		else {
			detection_indices = *detection_indicesi;
		}

        if (detection_indices.empty() || track_indices.empty()) {
            RR rr;
            rr.unmatched_tracks = track_indices;
            rr.unmatched_detections = detection_indices;
            return rr;
        }
        // 5x5
        DYNAMICM cost_matrix = getCostMarixFun(
            tracks, detections, &track_indices, &detection_indices);
		//std::cout << "\n----mmmmm----\n" << cost_matrix << "\n----vvvvv-----\n" << std::endl;
		for (int i = 0; i < cost_matrix.rows(); i++) {
			for (int j = 0; j < cost_matrix.cols(); j++) {
				float tmp = cost_matrix(i, j);
				if (tmp > max_distance) {
					cost_matrix(i, j) = max_distance + 1e-5;
				}
			}
		}
		//std::cout << "\n----222mmmmm----\n" << cost_matrix << "\n----222vvvvv-----\n" << std::endl;
		//Eigen::Matrix<float, -1, 2> indices = KF::Instance()->LinearAssignmentForCpp(cost_matrix);
		Eigen::Matrix<float, -1, 2> indices =
			HungarianOper::Solve(cost_matrix);
		//std::cout << "indices:\n" << indices << std::endl;
        //xyztodo: indices = linear_assignment(cost_matrix)
        // (-1, 2)

        RR rr;
        // 是否在第2列
        for (int col = 0; col < detection_indices.size(); col++) {
            // check if col is in indecis[:,1]
            bool isIn = false;
            for (int i = 0; i < indices.rows(); i++) {
                int iid = indices(i, 1);
                if (col == iid) {
                    isIn = true;
                    break;
                }
            }
            if (!isIn) {
                int detection_idx = detection_indices[col];
                rr.unmatched_detections.push_back(detection_idx);
            }
        }
        // 是否在第1列
        for (int row = 0; row < track_indices.size(); row++) {
            // check of row is in indecis[:,0]
            bool isIn = false;
            for (int i = 0; i < indices.rows(); i++) {
                int iid = indices(i, 0);
                if (row == iid) {
                    isIn = true;
                    break;
                }
            }
            if (!isIn) {
                int track_idx = track_indices[row];
                rr.unmatched_tracks.push_back(track_idx);
            }
        }
        for (int i = 0; i < indices.rows(); i++) {
			int row = indices(i, 0);
			int col = indices(i, 1);
            //for (int j = 0; j < indices.cols(); j++) {
             //   int row = i;
             //   int col = j;
                int track_idx = track_indices[row];
                int detection_idx = detection_indices[col];
                if (cost_matrix(row, col) > max_distance) {
                    rr.unmatched_tracks.push_back(track_idx);
                    rr.unmatched_detections.push_back(detection_idx);
                }
                else {
                    rr.matches.push_back(std::make_pair(track_idx, detection_idx));
                }
            //}
        }
        return rr;
    }

    static RR matching_cascade(
        const GetCostMarixFun &getCostMarixFun, float max_distance,
        int cascade_depth,
        const std::vector<KalmanTracker*> &tracks,
        const std::vector<Detection> &detections,
        IDS *track_indicesi = NULL,
        IDS *detection_indicesi = NULL){
		IDS track_indices;
		IDS detection_indices;
        if(track_indicesi == NULL) {
			for (int i = 0; i < tracks.size(); i++) {
				track_indices.push_back(i);
			}
        }
		else {
			track_indices = *track_indicesi;
		}

        if (detection_indicesi == NULL) {
			for (int i = 0; i < detections.size(); i++) {
				detection_indices.push_back(i);
			}
        }
		else {
			detection_indices = *detection_indicesi;
		}
        RR re;
        std::map<int, int> tmpMap;
        IDS unmatched_detections = detection_indices;
        for (int level = 0; level < cascade_depth; level++) {
            if (unmatched_detections.empty()) {
                break;
            }
            IDS track_indices_l;
            for (int k = 0; k < track_indices.size(); k++) {
                if (tracks[k]->time_since_update_ == level + 1) {
                    track_indices_l.push_back(track_indices[k]);
                }
            }
            if (track_indices_l.empty()) {
                continue;
            }
            RR rr = min_cost_matching(
                getCostMarixFun, max_distance, tracks, detections,
                &track_indices_l, &unmatched_detections);
            unmatched_detections = rr.unmatched_detections;
            for (int i = 0; i < rr.matches.size(); i++) {
                std::pair<int, int> pa = rr.matches[i];
                re.matches.push_back(pa);
                tmpMap.insert(pa);
            }
        }
        re.unmatched_detections = unmatched_detections;
        for (int i = 0; i < track_indices.size(); i++) {
            int tid = track_indices[i];
            std::map<int, int>::iterator it = tmpMap.find(tid);
            if (it == tmpMap.end()) {
                re.unmatched_tracks.push_back(tid);
            }
        }

        return re;
    }

    static DYNAMICM gate_cost_matrix(
        const KF &kalmanFilter,
        DYNAMICM &cost_matrix, 
        const std::vector<KalmanTracker*> &tracks,
        const std::vector<Detection> &detections,
        IDS track_indices,
        IDS detection_indices,
        int gated_cost=INFTY_COST, 
        bool only_position=false){
        int gating_dim = only_position ? 2 : 4;
		float gating_threshold = chi2inv95[gating_dim];
        DSBOXS measurements(detection_indices.size(), 4);
        for (int i = 0; i < detection_indices.size(); i++) {
            int pos = detection_indices[i];
			DSBOX tmp = detections[pos].to_xyah();
			measurements.row(i) = tmp;
        }
        for (int row = 0; row < track_indices.size(); row++) {
            int track_idx = track_indices[row];
            KalmanTracker *track = tracks[track_idx];
			// gating_distance is a vector
			Eigen::Matrix<float, 1, -1> gating_distance = kalmanFilter.gating_distance(
                track->mean_, track->covariance_, measurements, only_position);
			for (int i = 0; i < gating_distance.cols(); i++) {
					if (gating_distance(0, i) > gating_threshold) {
						cost_matrix(row, i) = gated_cost;
					}
			}
			//std::cout << "\nb--ggg\n" << cost_matrix << "\e--ggg\n";
        }
        return cost_matrix;
    }
};
#endif
