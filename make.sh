#!/bin/bash


IINCLUDE="-I/usr/local/include/opencv -I/usr/local/include -I/usr/include/eigen3/Eigen"


LLIBPATH="-L/usr/local/lib -L/home/xyz/code1/DS/deepsort/FeatureGetter"
LLIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lFeatureGetter"

rm DS -rf
g++ --std=c++14 -o DS $IINCLUDE $LLIBPATH $LLIBS Main.cpp deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp 






