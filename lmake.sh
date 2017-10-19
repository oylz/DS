#!/bin/bash


#IINCLUDE="-I/usr/local/include/opencv -I/usr/local/include -I/usr/include/eigen3/Eigen"


#LLIBPATH="-L/usr/local/lib -L/home/xyz/code1/DS/deepsort/FeatureGetter"
#LLIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lFeatureGetter"

#rm libDS.so -rf
#g++ --std=c++14 -fPIC -shared -o libDS.so $IINCLUDE $LLIBPATH $LLIBS deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp  NT.cpp

#===================================================================================================================================================

IINCLUDE="-I/home/xyz/code/test/pp/opencvlib/include -I/usr/local/include -I/usr/include/eigen3/Eigen -I/home/xyz/code1/tbb-2018_U1/include/tbb -I/home/xyz/code1/tbb-2018_U1/include"


LLIBPATH="-L/home/xyz/code/test/pp/opencvlib/lib -L/usr/local/lib -L/home/xyz/code1/DS/deepsort/FeatureGetter -L/home/xyz/code1/tbb-2018_U1/build/linux_intel64_gcc_cc5.4.0_libc2.17_kernel3.10.0_release"

rm libDS.so -rf


function BOPENMP(){
	#LLIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lFeatureGetter"
	LLIBS="-lopencv_corexyz -lopencv_imgprocxyz  -lopencv_highguixyz -lFeatureGetter"
	g++ --std=c++14 -ggdb -fPIC -shared -fopenmp -o libDS.so $IINCLUDE $LLIBPATH $LLIBS deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp  NT.cpp fdsst/fdssttracker.cpp fdsst/fhog.cpp
}


function BTBB(){
	LLIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lFeatureGetter -ltbb"
	g++ --std=c++14 -fPIC -shared -DUSETBB -o libDS.so $IINCLUDE $LLIBPATH $LLIBS deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp  NT.cpp
}


BOPENMP



