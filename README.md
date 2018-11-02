# DS

C++ implementation of Simple Online Realtime Tracking with a Deep Association Metric

# 1. depencies
component|version
-|-
eigen|3.3
opencv|-
boost|-
tensorflow|1.4

# 2. build
>./make.sh

# 3. prepare data

> change the var values at lines(335-337) in Main.cpp:
```
_imgDir = "/home/xyz/code1/xyz/img1/"; // MOT format

_rcFile = "/home/xyz/code1/xyz/det/det.txt"; // MOT format

_imgCount = 680;  // frames count
```


# 4. run

> ./r.sh

# 5.tips

>tensorflow build:
```
(1) ./configure
---
(2) bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda  tensorflow:libtensorflow_cc.so
```








