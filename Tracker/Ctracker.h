#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>

#include "defines.h"
#include "track.h"

enum DistType
{
	DT_DistCenters = 0,
	DT_DistRects = 1,
	DT_DistJaccard = 2
};
struct TrackCtrl {
	TrackCtrl() {

	}
	TrackCtrl(
		DistType _m_distType,
		KalmanType _m_kalmanType,
		track_t _dt,
		track_t _accelNoiseMag,
		track_t _dist_thres,
		size_t _maximum_allowed_skipped_frames) {
			m_distType = _m_distType,
			m_kalmanType = _m_kalmanType,
			dt = _dt,
			accelNoiseMag = _accelNoiseMag,
			dist_thres = _dist_thres,
			maximum_allowed_skipped_frames = _maximum_allowed_skipped_frames;
	}

	DistType m_distType = DT_DistCenters;
	KalmanType m_kalmanType = KT_TypeLinear;

	// Шаг времени опроса фильтра
	track_t dt = 0;

	track_t accelNoiseMag = 0;

	// Порог расстояния. Если точки находятся дуг от друга на расстоянии,
	// превышающем этот порог, то эта пара не рассматривается в задаче о назначениях.
	track_t dist_thres = 0;
	// Максимальное количество кадров которое трек сохраняется не получая данных о измерений.
	size_t maximum_allowed_skipped_frames = 0;

};
class CTracker
{
public:
    CTracker(const TrackCtrl &tc);
	~CTracker(void);

	void Update(const std::vector<Point_t>& detections,
		const regions_t& regions,
		cv::Mat frame);

#if SAVE_TRAJECTORIES
    void WriteAllTracks();
#endif
	KalmanTrackers &GetKalmanTrackersFor_MouseExample_MonitorDetect_FaceDetector_PedestrianDetector() {
		return kalmanTrackres_;
	}
private:
	TrackCtrl tc_;
	KalmanTrackers kalmanTrackres_;

	size_t NextTrackID;

    cv::Mat m_prevFrame;
	regions_t lastRgs_;
};
