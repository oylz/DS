#ifndef _FEATUREGETTERH_
#define _FEATUREGETTERH_

#ifndef PYKF
#include "feature_getter.h"
#else

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
	#ifdef WIN32
	#include <python.h>
	#else
	#include <Python.h>
	#endif
#endif
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>


#include "../StrCommon.h"


#include "Detection.h"


typedef std::vector<double> DSR;
typedef std::vector<DSR> DSRS;
typedef std::vector<int> IDSR;
typedef std::vector<IDSR> IDSRS;
static const std::string PYROOT = "e:/code/DS/py-win/";
class Tpy {
public:
	static IDSRS PPI(PyObject *pyResult) {
		IDSRS re;
		PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pyResult);
		// Convert back to C++ array and print.
		int COUNT = PyArray_SHAPE(np_ret)[0];
		int LEN = PyArray_SHAPE(np_ret)[1];
		int *c_out = reinterpret_cast<int*>(PyArray_DATA(np_ret));
		//printf("=====begin=================\n");
		for (int i = 0; i < COUNT; i++) {
			IDSR dsr;
			for (int j = 0; j < LEN; j++) {
				if (j>0) {
					//printf(",");
				}
				int tmp = c_out[i*LEN + j];
				dsr.push_back(tmp);
				//std::cout << tmp;
				if (j == LEN - 1) {
					//printf("\n");
				}
			}
			re.push_back(dsr);
		}
		//printf("=====end=================\n");
		return re;
	}

	static DSRS PP(PyObject *pyResult, bool onlyone = false) {
		DSRS re;
		PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pyResult);
		// Convert back to C++ array and print.
		int COUNT = PyArray_SHAPE(np_ret)[0];
		int LEN = PyArray_SHAPE(np_ret)[1];
		if (onlyone) {
			LEN = COUNT;
			COUNT = 1;
		}
		double *c_out = reinterpret_cast<double*>(PyArray_DATA(np_ret));
		//printf("=====begin=================\n");
		//std::cout.precision(20);
		for (int i = 0; i < COUNT; i++) {
			DSR dsr;
			for (int j = 0; j < LEN; j++) {
				if (j>0) {
					//printf(",");
				}
				double tmp = c_out[i*LEN + j];
				dsr.push_back(tmp);
				//printf("%llf", tmp);
				//std::cout << tmp;
				if (j == LEN - 1) {
					//printf("\n");
				}
			}
			re.push_back(dsr);
		}
		//printf("=====end=================\n");
		return re;
	}
	static void TDSBOX(const DSBOX &xyah, PyObject *args, int pos) {
		npy_intp IMGSHAPE[1] = { 4 };
		float *a = (float *)xyah.data();
		PyByteArrayObject *imga = reinterpret_cast<PyByteArrayObject *>
			(PyArray_SimpleNewFromData(1,
				IMGSHAPE,
				NPY_FLOAT,
				reinterpret_cast<void *>(a)));
		PyTuple_SetItem(args, pos, reinterpret_cast<PyObject *>(imga));
	}
	static void TMEAN(const MEAN &mean, PyObject *args, int pos) {
		npy_intp IMGSHAPE[1] = { 8 };
		float *a = (float *)mean.data();
		PyByteArrayObject *imga = reinterpret_cast<PyByteArrayObject *>
			(PyArray_SimpleNewFromData(1,
				IMGSHAPE,
				NPY_FLOAT,
				reinterpret_cast<void *>(a)));
		PyTuple_SetItem(args, pos, reinterpret_cast<PyObject *>(imga));
	}
	static void TVAR(const VAR &var, PyObject *args, int pos) {
		npy_intp IMGSHAPE[1] = { 64 };
		float *a = (float *)var.data();
		PyByteArrayObject *imga = reinterpret_cast<PyByteArrayObject *>
			(PyArray_SimpleNewFromData(1,
				IMGSHAPE,
				NPY_FLOAT,
				reinterpret_cast<void *>(a)));
		PyTuple_SetItem(args, pos, reinterpret_cast<PyObject *>(imga));
	}
	static void TDSBOXS(const DSBOXS &xyahs, PyObject *args, int pos) {
		int rows = xyahs.rows();
		npy_intp IMGSHAPE[1] = { 4 * rows };
		float *a = (float *)xyahs.data();
		PyByteArrayObject *imga = reinterpret_cast<PyByteArrayObject *>
			(PyArray_SimpleNewFromData(1,
				IMGSHAPE,
				NPY_FLOAT,
				reinterpret_cast<void *>(a)));
		PyTuple_SetItem(args, pos, reinterpret_cast<PyObject *>(imga));
	}
	static void TMATRIX(const DYNAMICM &xyahs, PyObject *args, int pos) {
		int rows = xyahs.rows();
		int cols = xyahs.cols();
		npy_intp IMGSHAPE[1] = { cols * rows };
		float *a = (float *)xyahs.data();
		PyByteArrayObject *imga = reinterpret_cast<PyByteArrayObject *>
			(PyArray_SimpleNewFromData(1,
				IMGSHAPE,
				NPY_FLOAT,
				reinterpret_cast<void *>(a)));
		PyTuple_SetItem(args, pos, reinterpret_cast<PyObject *>(imga));
	}

	static PyObject *PreArg(int count, PyObject *obj) {
		PyObject *args = PyTuple_New(count);
		// self
		PyTuple_SetItem(args, 0, Py_BuildValue("O", obj));
		return args;
	}
	static PyObject *Call(PyObject *fun, PyObject *args) {
		PyObject *pyResult = NULL;
		try {
			pyResult = PyObject_CallObject(fun, args);
		}
		catch (std::exception &e) {
			std::string err(e.what());
			printf("initiate error:%s\n", err.c_str());
		}
		return pyResult;
	}
	static std::pair<MEAN, VAR> TResult(PyObject *pyResult) {
		std::pair<MEAN, VAR> re;
		PyObject *b1, *b2;
		PyArg_ParseTuple(pyResult, "OO", &b1, &b2);
		//
		MEAN mean;
		DSRS dsrs = PP(b1, true);
		if (dsrs.empty()) {
			return re;
		}
		DSR &dsr = dsrs[0];
		for (int i = 0; i < dsr.size(); i++) {
			mean(0, i) = dsr[i];
		}
		//
		VAR var;
		dsrs = PP(b2);
		if (dsrs.size() < 8) {
			return re;
		}
		for (int i = 0; i < dsrs.size(); i++) {
			DSR dsr1 = dsrs[i];
			for (int j = 0; j < dsr1.size(); j++) {
				var(i, j) = dsr1[j];
			}
		}
		re.first = mean;
		re.second = var;
		return re;
	}
};


class KF {
private:
	static KF *self_;
	PyObject *kf_ = NULL;
	PyObject *gating_distance_ = NULL;
	PyObject *initiate_ = NULL;
	PyObject *predict_ = NULL;
	PyObject *update_ = NULL;
	PyObject *linearAssignment_ = NULL;
private:
	KF() {

	}
public:
	static KF *Instance() {
		if (self_ == NULL) {
			self_ = new KF();
		}
		return self_;
	}
	Eigen::Matrix<float, -1, 2> LinearAssignmentForCpp(
					const DYNAMICM &cost_matrix) {
		PyObject *args = Tpy::PreArg(4, kf_);
		Tpy::TMATRIX(cost_matrix, args, 1);
		PyTuple_SetItem(args, 2, Py_BuildValue("i", cost_matrix.rows()));
		PyTuple_SetItem(args, 3, Py_BuildValue("i", cost_matrix.cols()));
		PyObject *pyResult = Tpy::Call(linearAssignment_, args);
		IDSRS dsrs = Tpy::PPI(pyResult);
		int pos = 0;
		Eigen::Matrix<float, -1, 2> re(dsrs.size(), 2);
		for (int i = 0; i < dsrs.size(); i++) {
			IDSR &dsr = dsrs[i];
			for (int j = 0; j < dsr.size(); j++) {
				re(i, j) = dsr[j];
			}
		}
		

		return re;
	}
	Eigen::Matrix<float, 1, -1> gating_distance(
		const MEAN &mean,
		const VAR &covariance,
		const DSBOXS &measurements,
		bool only_position) const{
		
		PyObject *args = Tpy::PreArg(5, kf_);
		Tpy::TMEAN(mean, args, 1);
		Tpy::TVAR(covariance, args, 2);
		Tpy::TDSBOXS(measurements, args, 3);
		PyTuple_SetItem(args, 4, Py_BuildValue("i", only_position));
		PyObject *pyResult = Tpy::Call(gating_distance_, args);
		DSRS dsrs = Tpy::PP(pyResult, true);// xyz 调试所得 设onlyone=true
		int pos = 0;
		int count = 0;
		for (int i = 0; i < dsrs.size(); i++) {
			DSR &dsr = dsrs[i];
			for (int j = 0; j < dsr.size(); j++) {
				count++;
			}
		}
		Eigen::Matrix<float, 1, -1> re(1, count);
		for (int i = 0; i < dsrs.size(); i++) {
			DSR &dsr = dsrs[i];
			for (int j = 0; j < dsr.size(); j++) {
				re(0, pos++) = dsr[j];
			}
		}

		return re;
	}
	std::pair<MEAN, VAR> initiate(const DSBOX &xyah){
		std::pair<MEAN, VAR> re;

		PyObject *args = Tpy::PreArg(2, kf_);
		Tpy::TDSBOX(xyah, args, 1);
		PyObject *pyResult = Tpy::Call(initiate_, args);
		re = Tpy::TResult(pyResult);
		return re;
	}
	std::pair<MEAN, VAR> predict(const MEAN &mean, const VAR &var) const{
		std::pair<MEAN, VAR> re;
		PyObject *args = Tpy::PreArg(3, kf_);
		Tpy::TMEAN(mean, args, 1);
		Tpy::TVAR(var, args, 2);
		PyObject *pyResult = Tpy::Call(predict_, args);
		re = Tpy::TResult(pyResult);

		return re;
	}
	std::pair<MEAN, VAR> update(
		const MEAN &mean,
		const VAR &covariance,
		const DSBOX &xyah) const{
		std::pair<MEAN, VAR> re;
		PyObject *args = Tpy::PreArg(4, kf_);
		Tpy::TMEAN(mean, args, 1);
		Tpy::TVAR(covariance, args, 2);
		Tpy::TDSBOX(xyah, args, 3);
		PyObject *pyResult = Tpy::Call(update_, args);
		re = Tpy::TResult(pyResult);

		return re;
	}
	bool Init() {
		PyObject *pyModule = PyImport_ImportModule("kalman_filter");
		if (!pyModule) {
			printf("Can not open python module kalman_filter\n");
			return false;
		}
		PyObject *kfi = PyObject_GetAttrString(pyModule, "KalmanFilter");
		kf_ = PyObject_CallObject(kfi, NULL);

		
		gating_distance_ = PyObject_GetAttrString(kfi, "gating_distance");
		initiate_ = PyObject_GetAttrString(kfi, "initiate");
		predict_ = PyObject_GetAttrString(kfi, "predict");
		update_ = PyObject_GetAttrString(kfi, "update");
		linearAssignment_ = PyObject_GetAttrString(kfi, "LinearAssignmentForCpp");
		return true;
	}
};



class FeatureGetter {
private:
	static FeatureGetter *self_;
public:
	static FeatureGetter *Instance() {
		if (self_ == NULL) {
			self_ = new FeatureGetter();
		}
		return self_;
	}
	bool Init() {
		Py_Initialize();
		if (!Py_IsInitialized()) {
			return false;
		}

		PyRun_SimpleString("import sys \nsys.argv = ['']");
		std::string ss = "sys.path.append('";
		ss += PYROOT;
		ss += "')";
		//PyRun_SimpleString("sys.path.append('E:/code/NewFaceTracker/py/')");
		PyRun_SimpleString(ss.c_str());

		PyObject *pyModule = PyImport_ImportModule("generate_detections");
		if (!pyModule) {
			printf("Can not open python module generate_detections\n");
			return false;
		}
		PyObject *gdi = PyObject_GetAttrString(pyModule, "Gd");
		if(gdi == NULL){
			printf("gdi is null\n");
			return false;
		}
		gd_ = PyObject_CallObject(gdi, NULL);
		if(gd_ == NULL){
			printf("gd is null");
			return false;
		}

		PreEnc(gdi);
		enc_ = PyObject_GetAttrString(gdi, "encodeForCpp");
		ii();
		return true;
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
#ifdef WIN32
		import_array();
#else
		if (_import_array() < 0) {
			PyErr_Print(); 
			PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
		}
#endif
		_init = true;
	}
	void Enc(const cv::Mat &img, const std::vector<cv::Rect> &rcs,
		std::vector<FEATURE> &fts) {
		PyObject *args = PyTuple_New(5);
		// self
		PyTuple_SetItem(args, 0, Py_BuildValue("O", gd_));

		// img
		//ii();
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
		PyByteArrayObject *imga = reinterpret_cast<PyByteArrayObject *>
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
		if(pyResult == NULL){
			printf("call enc return null, exit\n");
			exit(0);
		}
		uint64_t tmd = gtm();
		printf("(incpp)call encode cost time:tmd-tmc:%d\n", (int)(tmd - tmc));

		if (pyResult) {
			PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pyResult);
			// Convert back to C++ array and print.
			int COUNT = PyArray_SHAPE(np_ret)[0];
			int LEN = PyArray_SHAPE(np_ret)[1];
			float *c_out = reinterpret_cast<float*>(PyArray_DATA(np_ret));
			//printf("=====begin=================\n");

			for (int i = 0; i < COUNT; i++) {
				//printf("---b---------------\n");
				FEATURE ft;
				for (int j = 0; j < LEN; j++) {
					/*if (j>0) {
						printf(",");
					}
					printf("%f", c_out[i*LEN + j]);
					if (j == LEN - 1) {
						printf("\n");
					}*/
					ft(0, j) = c_out[i*LEN + j];
				}
				fts.push_back(ft);
				//printf("---e---------------\n");
			}
			//printf("=====end=================\n");
		}
		delete[]buf;
	}
	bool PreEnc(PyObject *gdi) {
		PyObject *preEnc = PyObject_GetAttrString(gdi, "preEncode");
		PyObject *args = PyTuple_New(2);
		PyTuple_SetItem(args, 0, Py_BuildValue("O", gd_));
		std::string ss = PYROOT;
		ss += "networks/mars-small128.ckpt-68577";
		PyTuple_SetItem(args, 1, Py_BuildValue("s",
			ss.c_str()));
		PyObject *pyResult = PyObject_CallObject(preEnc, args);
		return (pyResult != NULL);
	}
};
#endif

#endif