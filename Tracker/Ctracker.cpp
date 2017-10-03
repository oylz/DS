#include "Ctracker.h"
#include "./HungarianAlg/HungarianAlg.h"

#include "ds/LossMgr.h"

CTracker::CTracker(const TrackCtrl &tc):NextTrackID(0){
	tc_ = tc;
}

CTracker::~CTracker(void){
}

void CTracker::Update(
        const std::vector<Point_t>& detections,
        const regions_t& regions,
        cv::Mat frame){
	// check if the same regions, and not update
	while (0) {
		if (regions.empty()) {
			break;
		}
		if (lastRgs_.size() != regions.size()) {
			break;
		}
		bool br = false;
		for (int i = 0; i < regions.size(); i++) {
			CRegion rg1 = regions[i];
			CRegion rg2 = lastRgs_[i];
			if (rg1.m_rect != rg2.m_rect) {
				br = true;
				break;
			}
		}
		if (br) {
			break;
		}
		// is the same regions
		//for (size_t i = 0; i < kalmanTrackres_.size(); i++) {
		//	kalmanTrackres_[i]->Update(CRegion(), frame, true);
		//}
		return;
	}
	static cv::Mat lossFrame;
	static std::map<int, int> lastFalseIds;

	printf("========BEGIN========================================================================================================================^^^^^^^\n");
    assert(detections.size() == regions.size());

	bool preEmpty = kalmanTrackres_.empty();

    // -----------------------------------
    // If there is no tracks yet, then every cv::Point begins its own track.
    // -----------------------------------
    if (kalmanTrackres_.size() == 0){
        // If no tracks yet
        for (size_t i = 0; i < detections.size(); ++i){
			int id = NextTrackID++;
			printf("*************new tracker1:%d\n", id);
			kalmanTrackres_.push_back(std::make_unique<KalmanTracker>(detections[i],
                                                      regions[i],
													  tc_.m_kalmanType,
                                                      tc_.dt,
                                                      tc_.accelNoiseMag,
                                                      id,
														frame.cols, frame.rows));
        }
    }

    size_t N = kalmanTrackres_.size();		// треки
    size_t M = detections.size();	// детекты

    assignments_t assignment(N, -1); // назначения
	LossMgr::Instance()->PreAddLoss();
    while(1){// begin while(1)A
		if (kalmanTrackres_.empty()) {
			break;
		}
        // get Cost matrix
        distMatrix_t Cost(N * M);
		track_t maxCost = 0;
		switch (tc_.m_distType){
        case DT_DistCenters:
            for (size_t i = 0; i < kalmanTrackres_.size(); i++){
                for (size_t j = 0; j < detections.size(); j++){
					auto dist = kalmanTrackres_[i]->CalcDist(detections[j]);
					Cost[i + j * N] = dist;
					if (dist > maxCost){
						maxCost = dist;
					}
                }
            }
            break;

        case DT_DistRects:
            for (size_t i = 0; i < kalmanTrackres_.size(); i++){
                for (size_t j = 0; j < detections.size(); j++){
					auto dist = kalmanTrackres_[i]->CalcDist(regions[j].m_rect);
					Cost[i + j * N] = dist;
					if (dist > maxCost){
						maxCost = dist;
					}
                }
            }
            break;

        case DT_DistJaccard:
            for (size_t i = 0; i < kalmanTrackres_.size(); i++){
                for (size_t j = 0; j < detections.size(); j++){
                    auto dist = kalmanTrackres_[i]->CalcDistJaccard(regions[j].m_rect);
                    Cost[i + j * N] = dist;
                    if (dist > maxCost){
                        maxCost = dist;
                    }
                }
            }
            break;
        }
		printf("*******solve hungarian\n");
        // -----------------------------------
        // Solving assignment problem (tracks and predictions of Kalman filter)
        // -----------------------------------
		{
			AssignmentProblemSolver APS;
			APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::optimal);
		}

		// -----------------------------------
		// clean assignment from pairs with large distance
		// -----------------------------------
		for (size_t i = 0; i < assignment.size(); i++){
			if (assignment[i] != -1){
				if (Cost[i + assignment[i] * N] > tc_.dist_thres){
					assignment[i] = -1;
					kalmanTrackres_[i]->m_skippedFrames++;
				}
			}
			else{
				// If track have no assigned detect, then increment skipped frames counter.
				kalmanTrackres_[i]->m_skippedFrames++;
			}
		}

		// -----------------------------------
        // If track didn't get detects long time, remove it.
        // -----------------------------------
        while(1){
				bool cn = false;
				printf("begin==================================\n");
				for (int i = 0; i < static_cast<int>(kalmanTrackres_.size()); i++){
					if (kalmanTrackres_[i]->m_skippedFrames > tc_.maximum_allowed_skipped_frames){
						printf("*********erase tracker:%d\n", kalmanTrackres_[i]->m_trackID);

						cv::Rect rr = kalmanTrackres_[i]->GetLastRect();
						if (rr.x>=10 &&
							rr.y>=10 &&
							rr.x + rr.width<=frame.cols - 10 &&
							rr.y + rr.height<=frame.rows - 10) {
							LossMgr::Instance()->AddLoss(kalmanTrackres_[i]->m_trackID,
								kalmanTrackres_[i]->lastFaces_);
						}

						kalmanTrackres_.erase(kalmanTrackres_.begin() + i);
						assignment.erase(assignment.begin() + i);
						i--;
						cn = true;
						break;
					}
				}
				printf("end==================================\n");
				if(cn){
					continue;
				}
				break;
		}
		break;
    }// end while(1)A
	LossMgr::Instance()->CommitAddLoss();
    // -----------------------------------
    // Search for unassigned detects and start new tracks for them.
    // -----------------------------------
    if(!preEmpty){
		for (size_t i = 0; i < detections.size(); ++i){
			if (find(assignment.begin(), assignment.end(), i) == assignment.end()){
				int id = 0;
				int ddis = -1;
				cv::Mat face = frame(regions[i].m_rect).clone();
				int iid = LossMgr::Instance()->GetSimilaryId(face);
				if (iid >= 0) {
					id = iid;
					printf("*******new tracker2(reuse id):%d\n", id);
					LossMgr::Instance()->RemoveLoss(iid);
				}
				else {
					id = NextTrackID++;
					printf("*******new tracker2:%d\n", id);
				}
			
				kalmanTrackres_.push_back(std::make_unique<KalmanTracker>(detections[i],
														  regions[i],
								  tc_.m_kalmanType,
														  tc_.dt,
														  tc_.accelNoiseMag,
														  id,
														  frame.cols, frame.rows));
				int pos = kalmanTrackres_.size() -1;
				kalmanTrackres_[pos]->m_skippedFrames = 0;
				kalmanTrackres_[pos]->Update(regions[i], frame, false);
			
			}
		}
    }

    // Update Kalman Filters state

    for (size_t i = 0; i < assignment.size(); i++){
        // If track updated less than one time, than filter state is not correct.

        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
			kalmanTrackres_[i]->m_skippedFrames = 0;
			kalmanTrackres_[i]->Update(regions[assignment[i]], frame, false);
        }
    }
    frame.copyTo(m_prevFrame);
	lastRgs_ = regions;
	printf("==END==============================================================================================================================UUUUUUU\n");
}
