

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NewFaceTracker
TEMPLATE = app
DEFINES += QT_DEPRECATED_WARNINGS


SOURCES += Main.cpp
	

HEADERS  += StrCommon.h \
       ./deepsort/*.h

	INCLUDEPATH += ./
	
    unix{
    INCLUDEPATH += /usr/local/include/opencv
    INCLUDEPATH += /usr/local/include
    LIBS += -L/usr/local/lib
    LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio
	LIBS += -lglog -lboost_system -lopencv_tracking
    }
    win32{
	INCLUDEPATH += E:\code\opencv-3.2.0\xyz\install\include
    INCLUDEPATH += E:\code\opencv-3.2.0\xyz\install\include\opencv
	INCLUDEPATH += D:\svnclientdata\libs\glog-master\src\windows
	INCLUDEPATH += D:\svnclientdata\libs\thrift-master\lib\cpp\src
	INCLUDEPATH += D:\svnclientdata\libs\ffmpeg-20170422-207e6de-win64-dev\include
	INCLUDEPATH += D:\svnclientdata\libs\boost_1_55_0
    INCLUDEPATH += "C:\Program Files\Anaconda3\Lib\site-packages\numpy\core\include"
    INCLUDEPATH += "C:\Program Files\Anaconda3\include"
    INCLUDEPATH += "C:\Program Files\Eigen3\include\eigen3\Eigen"
	LIBS += -llibglog
	LIBS += -llibthrift
    
    }
    
    
    CONFIG += debug_and_release
    CONFIG(debug, debug|release){
	win32{
		LIBS += -LD:\svnclientdata\libs\thrift-master\lib\cpp\x64\Debug 
        LIBS += -LE:\code\opencv-3.2.0\xyz\install\x64\vc14\lib
        LIBS += -LD:\svnclientdata\libs\ffmpeg-20170422-207e6de-win64-dev\lib 
        LIBS += -LD:\svnclientdata\libs\boost_1_55_0\stage\lib 
        LIBS += -LD:\svnclientdata\libs\glog-master\x64\Debug 
        LIBS += -L"C:\Program Files\Anaconda3\libs"
        LIBS += -lopencv_core320d -lopencv_imgproc320d -lopencv_imgcodecs320d -lopencv_highgui320d -lopencv_video320d -lopencv_videoio320d
        LIBS += -lopencv_tracking320d
		QMAKE_POST_LINK += copy /y D:\svnclientdata\libs\ffmpeg-20170422-207e6de-win64-shared\bin\*.dll  .\debug\ &
		QMAKE_POST_LINK += copy /y E:\code\opencv-3.2.0\xyz\install\x64\vc14\bin\*.dll .\debug\ &
		QMAKE_POST_LINK += copy /y D:\svnclientdata\libs\glog-master\x64\Debug\libglog.dll .\debug\ &
	}    
    unix{
      TARGET = bin/NewFaceTrackerd
    }
    }
    else{
	win32{
        LIBS += -LD:\svnclientdata\libs\thrift-master\lib\cpp\x64\Release 
        LIBS += -LE:\code\opencv-3.2.0\xyz\install\x64\vc14\lib
        LIBS += -LD:\svnclientdata\libs\ffmpeg-20170422-207e6de-win64-dev\lib 
        LIBS += -LD:\svnclientdata\libs\boost_1_55_0\stage\lib 
        LIBS += -LD:\svnclientdata\libs\glog-master\x64\Release 
		LIBS += -lopencv_core320 -lopencv_imgproc320 -lopencv_imgcodecs320 -lopencv_highgui320 -lopencv_video320 -lopencv_videoio320
        LIBS += -lopencv_tracking320
		QMAKE_POST_LINK += copy /y D:\svnclientdata\libs\ffmpeg-20170422-207e6de-win64-shared\bin\*.dll  .\release\ &
		QMAKE_POST_LINK += copy /y E:\code\opencv-3.2.0\xyz\install\x64\vc14\bin\*.dll .\release\ &
		QMAKE_POST_LINK += copy /y D:\svnclientdata\libs\glog-master\x64\Release\libglog.dll .\release\ &
	}    
    unix{
      TARGET = bin/NewFaceTracker
    }
    }


LIBS += -lavcodec
LIBS += -lavdevice
LIBS += -lavfilter
LIBS += -lavformat
LIBS += -lavutil
LIBS += -lswscale

CONFIG += c++14




QT += widgets
QT += opengl

DEFINES += USE_OCV_UKF



