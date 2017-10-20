#!/bin/bash
function getbazel(){
	LINE=`readlink -f /home/$USER/code1/tensorflow-1.4.0-rc0/bazel-bin/`

	POS1="_bazel_$USER/"
	STR=${LINE##*$POS1}

	BAZEL=${STR:0:32}

	echo $BAZEL
}



BAZEL=`getbazel`

IINCLUDE="-I/home/$USER/code/test/pp/opencvlib/include -I/usr/local/include -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive/Eigen -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/eigen_archive -I/home/$USER/code1/tensorflow-1.4.0-rc0 -I/home/$USER/code1/tensorflow-1.4.0-rc0/bazel-genfiles -I/home/$USER/.cache/bazel/_bazel_$USER/$BAZEL/external/nsync/public"

LLIBPATH="-L/home/$USER/code/test/pp/opencvlib/lib -L/usr/local/lib -L/home/$USER/code1/tensorflow-1.4.0-rc0/bazel-bin/tensorflow"
LLIBS="-lopencv_corexyz -lopencv_imgprocxyz -lopencv_highguixyz -ltensorflow_cc -ltensorflow_framework"




rm libFeatureGetter.so -rf
g++ --std=c++14 -fPIC -shared -o libFeatureGetter.so $IINCLUDE $LLIBPATH  FeatureGetter.cpp $LLIBS






