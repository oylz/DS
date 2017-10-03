#ifndef _FEATUREGETTERH_
#define _FEATUREGETTERH_

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>


#include "../../StrCommon.h"

typedef std::vector<float> FEATURE;

class FeatureGetter {
	friend class LossMgr;
protected:
	bool Init() {
		Py_Initialize();
		if (!Py_IsInitialized()) {
			return false;
		}

		PyRun_SimpleString("import sys \nsys.argv = ['']");
		PyRun_SimpleString("sys.path.append('E:/wk20170824/DeepSort/')");

		PyObject *pyModule = PyImport_ImportModule("generate_detections");
		if (!pyModule) {
			printf("Can not open python module\n");
			return false;
		}
		PyObject *gdi = PyObject_GetAttrString(pyModule, "Gd");
		gd_ = PyObject_CallObject(gdi, NULL);

		PreEnc(gdi);
		enc_ = PyObject_GetAttrString(gdi, "encodeForCpp");
	}
	void Get(const cv::Mat &img, const std::vector<cv::Rect> &rcs,
		std::vector<FEATURE> &fts) {
		this->Enc(img, rcs, fts);
	}

public:
	~FeatureGetter() {
		Py_Finalize();
	}
private:
	PyObject *gd_ = NULL;
	PyObject *enc_ = NULL;
	bool _init = false;
private:
	int ii() {
		if (_init) {
			return 0;
		}
		//必须添加如下函数，否则无法执行PyArray_SimpleNewFromData
		import_array();
		_init = true;
	}
	void Enc(const cv::Mat &img, const std::vector<cv::Rect> &rcs,
		std::vector<FEATURE> &fts) {
		PyObject *args = PyTuple_New(5);
		// self
		PyTuple_SetItem(args, 0, Py_BuildValue("O", gd_));

		// img
		PyByteArrayObject *imga;
		ii();
		int LLL = img.cols*img.rows * 3;
		npy_intp IMGSHAPE[1] = { LLL };
		int nr = img.rows;
		int nc = img.cols;
		uchar *buf = new uchar[LLL];
		memset(buf, 0, LLL);
		if (img.isContinuous())
		{
			nr = 1;
			nc = LLL;
		}
		int pos = 0;
		for (int i = 0; i<nr; i++)
		{
			const uchar* inData = img.ptr<uchar>(i);
			for (int j = 0; j<nc; j++)
			{
				buf[pos++] = *inData++;
			}
		}
		imga = reinterpret_cast<PyByteArrayObject *>
			(PyArray_SimpleNewFromData(1, IMGSHAPE, NPY_BYTE, reinterpret_cast<void *>(buf)));
		PyTuple_SetItem(args, 1, reinterpret_cast<PyObject *>(imga));

		// width & height
		PyTuple_SetItem(args, 2, Py_BuildValue("i", img.rows));
		PyTuple_SetItem(args, 3, Py_BuildValue("i", img.cols));
		// boxes
		{
			std::string boxesStr = "";
			for (cv::Rect rc : rcs) {
				std::string line = "";
				line += toStr(rc.x);
				line += " ";
				line += toStr(rc.y);
				line += " ";
				line += toStr(rc.width);
				line += " ";
				line += toStr(rc.height);
				line += "\n";
				boxesStr += line;
			}
			PyTuple_SetItem(args, 4, Py_BuildValue("s", boxesStr.c_str()));
		}
		uint64_t tmc = gtm();
		PyObject *pyResult = NULL;
		try {
			pyResult = PyObject_CallObject(enc_, args);
		}
		catch (std::exception &e) {
			std::string err(e.what());
			printf("enc error:%s\n", err.c_str());
		}
		uint64_t tmd = gtm();
		printf("tmd-tmc:%d\n", (int)(tmd - tmc));

		if (pyResult) {
			PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pyResult);
			// Convert back to C++ array and print.
			int COUNT = PyArray_SHAPE(np_ret)[0];
			int LEN = PyArray_SHAPE(np_ret)[1];
			float *c_out = reinterpret_cast<float*>(PyArray_DATA(np_ret));
			printf("=====begin=================\n");

			for (int i = 0; i < COUNT; i++) {
				printf("---b---------------\n");
				FEATURE ft;
				for (int j = 0; j < LEN; j++) {
					/*if (j>0) {
						printf(",");
					}
					printf("%f", c_out[i*LEN + j]);
					if (j == LEN - 1) {
						printf("\n");
					}*/
					ft.push_back(c_out[i*LEN + j]);
				}
				fts.push_back(ft);
				printf("---e---------------\n");
			}
			printf("=====end=================\n");
		}
		delete[]buf;
	}
	bool PreEnc(PyObject *gdi) {
		PyObject *preEnc = PyObject_GetAttrString(gdi, "preEncode");
		PyObject *args = PyTuple_New(2);
		PyTuple_SetItem(args, 0, Py_BuildValue("O", gd_));
		PyTuple_SetItem(args, 1, Py_BuildValue("s",
			"E:/code/deep_sort-master/resources/networks/mars-small128.ckpt-68577"));
		PyObject *pyResult = PyObject_CallObject(preEnc, args);
		return (pyResult != NULL);
	}
};
#endif