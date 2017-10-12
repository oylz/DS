#!/bin/bash


#IINCLUDE="-I/usr/local/include/opencv -I/usr/local/include -I/usr/include/python2.7 -I/usr/lib64/python2.7/site-packages/numpy/core/include -I/usr/include/eigen3/Eigen"





IINCLUDE="-I/usr/local/include/opencv -I/usr/local/include -I/home/xyz/anaconda2/include/python2.7 -I/home/xyz/anaconda2/lib/python2.7/site-packages/numpy/core/include -I/usr/include/eigen3/Eigen"
LLIBPATH="-L/usr/local/lib -L/home/xyz/anaconda2/lib"
LLIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lpython2.7"

rm DS -rf
#g++ --std=c++14 -ggdb -DPYKF -o DS $IINCLUDE $LLIBPATH $LLIBS Main.cpp deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp
g++ --std=c++14 -ggdb -DKLOG -o DS $IINCLUDE $LLIBPATH $LLIBS Main.cpp deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp






