#!/bin/bash
function getbazel(){
	LINE=`readlink -f /home/$USER/code1/tensorflow-1.4.0-rc0/bazel-bin/`

	POS1="_bazel_$USER/"
	STR=${LINE##*$POS1}

	BAZEL=${STR:0:32}

	echo $BAZEL
}



BAZEL=`getbazel`

function TF(){
IINCLUDE="-I/home/$USER/code/test/pp/opencvlib/include -I/usr/local/include -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive/Eigen -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive -I/home/$USER/code1/tensorflow-1.4.0-rc0 -I/home/$USER/code1/tensorflow-1.4.0-rc0/bazel-genfiles -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/nsync/public"

LLIBPATH="-L/home/$USER/code/test/pp/opencvlib/lib -L/usr/local/lib -L/home/$USER/code1/tensorflow-1.4.0-rc0/bazel-bin/tensorflow"
LLIBS="-lopencv_corexyz -lopencv_imgprocxyz -lopencv_highguixyz -ltensorflow_cc -ltensorflow_framework"

rm libFeatureGetter.so -rf
g++ --std=c++14 -O3 -fopenmp -fPIC -shared -o libFeatureGetter.so $IINCLUDE $LLIBPATH  FeatureGetter.cpp $LLIBS

}


#CAFFEROOT="/home/xyz/code/py-faster-rcnn/caffe-fast-rcnn"
CAFFEROOT="/home/$USER/code1/caffe-master"
#CAFFEROOT="/home/$USER/code1/caffe-ssd"


function CAFFE(){
IINCLUDE="-I/home/$USER/code/test/pp/opencvlib/include -I/usr/local/include -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive/Eigen -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive -I$CAFFEROOT/include -I/usr/local/cuda-8.0-cudnn5.0.5/include -I$CAFFEROOT/build/src"
LLIBPATH="-L/home/$USER/code/test/pp/opencvlib/lib -L$CAFFEROOT/distribute/lib"
LLIBS="-lopencv_corexyz -lopencv_imgprocxyz -lopencv_highguixyz -lcaffe"

rm libFeatureGetter.so -rf
g++ --std=c++14 -O3 -fopenmp -DOONE -DUSE_CAFFE_SHUFFE_NET -fPIC -shared -o libFeatureGetter.so $IINCLUDE $LLIBPATH  FeatureGetter.cpp $LLIBS
#g++ --std=c++14 -O3 -fopenmp  -DUSE_CAFFE_SHUFFE_NET -fPIC -shared -o libFeatureGetter.so $IINCLUDE $LLIBPATH  FeatureGetter.cpp $LLIBS


}



#CAFFE
TF






