#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>

#include "defines.h"
#include "Kalman.h"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"


class KalmanTracker{
public:
	KalmanTracker(
            const Point_t& pt,
            const CRegion& region,
            KalmanType kalmanType,
            track_t deltaTime,
            track_t accelNoiseMag,
            size_t trackID,
			int width,
			int height
            )
		:
        m_trackID(trackID),
        m_skippedFrames(0),
        m_lastRegion(region),
        m_predictionPoint(pt){
        {
            m_kalman = new TKalmanFilter(kalmanType, 
				region.m_rect, 
				deltaTime, 
				accelNoiseMag);
        }
		width_ = width;
		height_ = height;
	}

	~KalmanTracker(){
		if(m_kalman){
			delete m_kalman;
		}
	}

    track_t CalcDist(const Point_t& pt){
        Point_t diff = m_predictionPoint - pt;
		return sqrtf(diff.x * diff.x + diff.y * diff.y);
	}

    track_t CalcDist(const cv::Rect& r){
		std::array<track_t, 4> diff;
        diff[0] = m_predictionPoint.x - m_lastRegion.m_rect.width / 2 - r.x;
        diff[1] = m_predictionPoint.y - m_lastRegion.m_rect.height / 2 - r.y;
        diff[2] = static_cast<track_t>(m_lastRegion.m_rect.width - r.width);
        diff[3] = static_cast<track_t>(m_lastRegion.m_rect.height - r.height);

		track_t dist = 0;
		for (size_t i = 0; i < diff.size(); ++i){
			dist += diff[i] * diff[i];
		}
		return sqrtf(dist);
	}

    track_t CalcDistJaccard(const cv::Rect& r){
        cv::Rect rr(GetLastRect());

        track_t intArea = (r & rr).area();
        track_t unionArea = r.area() + rr.area() - intArea;

        return 1 - intArea / unionArea;
    }
    void Update(const CRegion& region, const cv::Mat &frame, bool noUseRegion){
		{
		//printf("prevFrame(%d, %d)\n", prevFrame.cols, prevFrame.rows);
            RectUpdate(region, frame, noUseRegion);
        }
		if (!noUseRegion) {
			m_lastRegion = region;
		}
    }

    size_t m_trackID;
    size_t m_skippedFrames;
    CRegion m_lastRegion;

    cv::Rect GetLastRect() const{
        return m_predictionRect;
    }
	std::vector<cv::Mat> lastFaces_;
private:
    Point_t m_predictionPoint;
    cv::Rect m_predictionRect;
    TKalmanFilter* m_kalman;

	cv::Rect lastR_;	
	int width_ = 0;
	int height_ = 0;
    void RectUpdate(const CRegion& region, const cv::Mat &frame, bool noUseRegion){
        //m_kalman->GetRectPrediction();
        m_predictionRect = m_kalman->GetRectPrediction();

		{
			m_predictionRect =
				m_kalman->Update(region.m_rect, !noUseRegion);
		}

        if (m_predictionRect.width < 2){
            m_predictionRect.width = 2;
        }
        if (m_predictionRect.x < 0){
            m_predictionRect.x = 0;
        }
        else if (m_predictionRect.x + m_predictionRect.width > width_ - 1){
            m_predictionRect.x = width_ - 1 - m_predictionRect.width;
        }
        if (m_predictionRect.height < 2){
            m_predictionRect.height = 2;
        }
        if (m_predictionRect.y < 0){
            m_predictionRect.y = 0;
        }
        else if (m_predictionRect.y + m_predictionRect.height > height_ - 1){
            m_predictionRect.y = height_ - 1 - m_predictionRect.height;
        }

        m_predictionPoint = (m_predictionRect.tl() + m_predictionRect.br()) / 2;
		lastR_ = region.m_rect;
		if (noUseRegion) {
			return;
		}
		lastFaces_.push_back(frame(m_predictionRect).clone());
		if (lastFaces_.size() > 10) {
			lastFaces_.erase(lastFaces_.begin());
		}

    }
};

typedef std::vector<std::unique_ptr<KalmanTracker>> KalmanTrackers;
