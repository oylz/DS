#!/bin/bash

function getbazel(){
	LINE=`readlink -f /home/$USER/code1/tensorflow-1.4.0-rc0/bazel-bin/`

	POS1="_bazel_$USER/"
	STR=${LINE##*$POS1}

	BAZEL=${STR:0:32}

	echo $BAZEL
}



BAZEL=`getbazel`




IINCLUDE="-I/home/$USER/code/test/pp/opencvlib/include -I/usr/local/include -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive/Eigen -I/home/$USER/code1/tbb-2018_U1/include/tbb -I/home/$USER/code1/tbb-2018_U1/include"


LLIBPATH="-L/home/$USER/code/test/pp/opencvlib/lib -L/usr/local/lib -L/home/$USER/code1/DS/deepsort/FeatureGetter -L/home/$USER/code1/tbb-2018_U1/build/linux_intel64_gcc_cc5.4.0_libc2.17_kernel3.10.0_release "

rm DS -rf


function BOPENMP(){
	LLIBS="-lopencv_corexyz -lopencv_imgprocxyz  -lopencv_highguixyz -lFeatureGetter -lboost_system -lglog -ltcmalloc"
	g++ --std=c++14 -O3 -fopenmp -DUDL -o DS $IINCLUDE $LLIBPATH  deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp  NT.cpp fdsst/fdssttracker.cpp fdsst/fhog.cpp Main.cpp $LLIBS
}


function BTBB(){
	LLIBS="-lopencv_corexyz -lopencv_imgprocxyz -lopencv_highguixyz -lFeatureGetter -lboost_system -lglog -ltbb"
	g++ --std=c++14 -DUSETBB -o DS $IINCLUDE $LLIBPATH deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp  NT.cpp Main.cpp $LLIBS
}


function BOPENMPHOG(){
	LLIBS="-lopencv_corexyz -lopencv_imgprocxyz  -lopencv_highguixyz  -lboost_system -lglog -ltcmalloc"
	g++ --std=c++14 -O3 -fopenmp -o DS $IINCLUDE $LLIBPATH  deepsort/munkres/munkres.cpp deepsort/munkres/adapters/adapter.cpp deepsort/munkres/adapters/boostmatrixadapter.cpp  NT.cpp fdsst/fdssttracker.cpp fdsst/fhog.cpp Main.cpp $LLIBS
}

BOPENMPHOG




