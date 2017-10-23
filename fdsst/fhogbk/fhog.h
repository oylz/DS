#ifndef FHOG_H
#define FHOG_H
#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
//#include <cstdlib>
#include <cmath>
#include <cstring>
#include "sse.hpp"
#include <map>


/**
    Inputs:
        float* I        - a gray or color image matrix with shape = channel x width x height
        int *h, *w, *d  - return the size of the returned hog features
        int binSize     -[8] spatial bin size
        int nOrients    -[9] number of orientation bins
        float clip      -[.2] value at which to clip histogram bins
        bool crop       -[false] if true crop boundaries

    Return:
        float* H        - computed hog features with shape: (nOrients*3+5) x (w/binSize) x (h/binSize), if not crop

    Author:
        Sophia
    Date:
        2015-01-15
**/

//float* fhog(float* I,int height,int width,int channel,int *h,int *w,int *d,int binSize = 4,int nOrients = 9,float clip=0.2f,bool crop = false);
float *HOGXYZ(const cv::Mat &input, int &len);
cv::Mat fhog(const cv::Mat& input, int binSize = 4,int nOrients = 9,float clip=0.2f,bool crop = false);

void change_format(float *des,float *source,int height,int width,int channel);


void hoglog();
float* acosTable(); 
#endif
