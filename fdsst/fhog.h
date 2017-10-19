#ifndef FHOG_H
#define FHOG_H
#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
#ifdef WIN32
#ifndef _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC
#endif
#include <stdlib.h>  
#include <crtdbg.h> 
#ifdef _DEBUG  
	#ifndef new
	#define new   new(_NORMAL_BLOCK, __FILE__, __LINE__)  
	#endif
#endif
#endif
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
cv::Mat fhog(const cv::Mat& input, int binSize = 4,int nOrients = 9,float clip=0.2f,bool crop = false);

void change_format(float *des,float *source,int height,int width,int channel);
//static std::map<void *, int> _nnmm;
//static boost::mutex nmm_;
// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void* wrCalloc(size_t num, size_t size) { 
	/*unsigned char *buf = new unsigned char[num*size]; 
	_nnmm.insert(std::make_pair(buf, 0));
	return (void*)buf;
	*/
	//boost::mutex::scoped_lock lock(nmm_);
	void *buf = calloc(num, size);
	
	//_nnmm.insert(std::make_pair(buf, 0));
	return buf;
}
inline void* wrMalloc( size_t size ) { 
	//boost::mutex::scoped_lock lock(nmm_);
	void * buf = malloc(size); 
	
	//_nnmm.insert(std::make_pair(buf, 0));
	return buf;
	/*unsigned char *buf = new unsigned char[size];
	_nnmm.insert(std::make_pair(buf, 0));
	return (void*)buf;*/
}
inline void wrFree( void * ptr ) { 
	//boost::mutex::scoped_lock lock(nmm_);
	//std::map<void *, int>::iterator it;
	//it = _nnmm.find(ptr);
	//if (it == _nnmm.end()) {
	//	printf("wo qu!!\n");
		//exit(0);
	//	return;
	//}
	free(ptr); 
	//_nnmm.erase(it);
	/*unsigned char *buf = (unsigned char *)ptr;
	std::map<unsigned char *, int>::iterator it;
	it = _nnmm.find(buf);
	if (it == _nnmm.end()) {
		printf("wo qu!!\n");
		exit(0);
	}
	delete []buf;
	_nnmm.erase(it);*/
}


void hoglog();
float* acosTable(); 
#endif
