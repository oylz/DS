#ifndef _IOUMH_
#define _IOUMH_
#include <vector>
#include "Detection.h"
#include <Eigen>
#include "linear_assignment.h"
#include <iterator>

class iou_matching{
private:
    static Eigen::VectorXf _iouFun(const DSBOX &bbox, const DSBOXS &candidates){
		Eigen::VectorXf area_candidates(candidates.rows());

		//
		PT2 bbox_tl; bbox_tl(0, 0) = bbox[0]; bbox_tl(0, 1) = bbox[1];
		PT2 bbox_br; bbox_br(0, 0) = (bbox[0] + bbox[2]); bbox_br(0, 1) = (bbox[1] + bbox[3]);
		DYNAMICM ctl(candidates.rows(), 2);
		DYNAMICM cbr(candidates.rows(), 2);
		for(int i = 0; i < candidates.rows(); i++){
			DSBOX candidate = candidates.row(i);
			PT2 candidates_tl;
			candidates_tl(0, 0) = candidate[0]; candidates_tl(0, 1) = candidate[1];
			ctl.row(i) = candidates_tl;
			PT2 candidates_br;
			candidates_br(0, 0) = (candidate[0] + candidate[2]);
			candidates_br(0, 1) = (candidate[1] + candidate[3]);
			cbr.row(i) = candidates_br;
			{
				area_candidates(i) = candidate[2] * candidate[3];
			}
		}
		std::cout << "ctl-b:\n" << ctl << "ctl-e\n" << std::endl;
		std::cout << "cbr-b:\n" << cbr << "cbr-e\n" << std::endl;
		DYNAMICM tl(candidates.rows(), 2);
		float btl0 = bbox_tl(0, 0);
		float btl1 = bbox_tl(0, 1);
		for (int i = 0; i < tl.rows(); i++) {
			//DYNAMICM row = tl.row(i);
			float m = max(btl0, ctl(i, 0));
			tl(i, 0) = m;
			m = max(btl1, ctl(i, 1));
			tl(i, 1) = m;
			std::cout << "tl-b:\n" << tl << "tl-e\n" << std::endl;
		}
		std::cout << "tl-b:\n" << tl << "tl-e\n" << std::endl;
		DYNAMICM br(candidates.rows(), 2);
		float bbr0 = bbox_br(0, 0);
		float bbr1 = bbox_br(0, 1);
		for (int i = 0; i < br.rows(); i++) {
			//DYNAMICM row = br.row(i);
			br(i, 0) = min(bbr0, cbr(i, 0));
			br(i, 1) = min(bbr1, cbr(i, 1));
		}
		std::cout << "br-b:\n" << br << "br-e\n" << std::endl;
		DYNAMICM wh(candidates.rows(), 2);
		Eigen::VectorXf area_intersection(candidates.rows());
		for (int i = 0; i < wh.rows(); i++) {
			for (int j = 0; j < wh.cols(); j++) {
				float tmp = br(i, j) - tl(i, j);
				wh(i, j) = tmp>0?tmp:0;
			}
			area_intersection(i) = wh(i, 0)*wh(i, 1);
		}
		std::cout << "wh-b:\n" << wh << "wh-e\n" << std::endl;
		float area_bbox = bbox(0, 2)*bbox(0, 3);
		
		Eigen::VectorXf re(candidates.rows());
		for (int i = 0; i < re.rows(); i++) {
			re(i) = area_intersection(i) / (area_bbox +
										area_candidates(i) -
										area_intersection(i));
		}
		return re;
    }

public:
    static DYNAMICM getCostMatrixByIOU(const std::vector<KalmanTracker*> &tracks, 
                    const std::vector<Detection> &detections, 
                    IDS *track_indicesi=NULL,
                    IDS *detection_indicesi=NULL){
		IDS track_indices;
		if (track_indicesi == NULL) {
			for (int i = 0; i < tracks.size(); i++) {
				track_indices.push_back(i);
			}
		}
		else {
			track_indices = *track_indicesi;
		}

		IDS detection_indices;
		if (detection_indicesi == NULL) {
			for (int i = 0; i < detections.size(); i++) {
				detection_indices.push_back(i);
			}
		}
		else {
			detection_indices = *detection_indicesi;
		}
		DYNAMICM cost_matrix(track_indices.size(), detection_indices.size());
		for (int row = 0; row < track_indices.size(); row++) {
			int track_idx = track_indices[row];
			if (tracks[track_idx]->time_since_update_ > 1) {
				for (int c = 0; c < cost_matrix.cols(); c++) {
					cost_matrix(row, c) = INFTY_COST;
				}
				continue;
			}
			DSBOX bbox = tracks[track_idx]->to_tlwh();
			DSBOXS candidates(detection_indices.size(), 4);
			for (int k = 0; k < detection_indices.size(); k++) {
				DSBOX tmp = detections[detection_indices[k]].tlwh_;
				candidates.row(k) = tmp;
			}
			std::cout << "mmm" << candidates << "vvvv" << std::endl;
			Eigen::VectorXf tmpm = _iouFun(bbox, candidates);
			std::cout << "tmpm--b" << tmpm << "tmpm--e" << std::endl;
			auto tmp1 = tmpm.array();
			auto tmp2 = -(tmp1 - 1);
			cost_matrix.row(row) = tmp2.matrix();
			std::cout << "nnnnn" << cost_matrix << "uuuu" << std::endl;
		}
		return cost_matrix;
    }
};
#endif