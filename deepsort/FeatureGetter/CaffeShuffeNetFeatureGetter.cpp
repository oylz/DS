
#include "FeatureGetter.h"

#include <caffe/net.hpp>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>


static int64_t fgtm() {
	struct timeval tm;
	gettimeofday(&tm, 0);
	int64_t re = ((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec;
	return re;
}


FeatureGetter *FeatureGetter::self_ = NULL;

typedef unsigned char uint8;

typedef boost::shared_ptr<caffe::Net<float> >  XNET;
typedef caffe::Blob<float>* XINPUT;

#ifdef OONE
static const int XCOUNT = 1;
#else
static const int XCOUNT = 10;
#endif
std::map<int, XNET> _xnets;
std::map<int, XINPUT> _xinputs;
int _iw = -1;
int _ih = -1;
int _outLayer = -1;
static const std::string _outLayerName = "fc1000";

static const std::string rootp = "/home/xyz/code1/ShuffleNet-Model/";
static const std::string modelp = rootp + "incode.prototxt";//"ssd_shufflenet_21_test.prototxt";
//static const std::string modelp = rootp + "ssd_shufflenet_21_test.prototxt";
static const std::string weightp = rootp + "shufflenet_1x_g3.caffemodel";

void to_buffer(const cv::Mat &img, float *buf){
    if (img.isContinuous()) {
      memcpy(buf, img.ptr<float>(0),
             static_cast<size_t>(img.total()) * sizeof(float));
    } 
    else {
      for (int i = 0; i < img.rows; i++) {
        memcpy(buf, img.ptr<float>(i),
               static_cast<size_t>(img.cols) * sizeof(float));
        buf += img.cols;
      }
    }
}


	bool FeatureGetter::Init() {
     		caffe::Caffe::set_mode(caffe::Caffe::GPU);
    		caffe::Caffe::SetDevice(0);
		for(int i = 0; i < XCOUNT; i++){
        		caffe::Net<float> *net = new caffe::Net<float>(modelp, caffe::TEST);
      			net->CopyTrainedLayersFrom(weightp);
			XNET xnet;
      			xnet.reset(net); 
			//
			xnet->ForwardFrom(0);
			_xnets.insert(std::make_pair(i, xnet));

			auto &blobs = xnet->input_blobs();
			XINPUT xinput = blobs[0];
			_xinputs.insert(std::make_pair(i, xinput));

			if(i == 0){
    				auto shape = blobs[0]->shape();
    				//////shape[0] = 1;
    				//////blobs[0]->Reshape(shape);
				_iw = (int)shape[2];
				_ih = (int)shape[3];	
				printf("_iw:%d, _ih:%d\n", _iw, _ih);

				int index = 0;
  				for (auto const &layer : xnet->layers()) {
    					auto const &param = layer->layer_param();
    					for (auto const &top_name : param.top()) {
      						if (top_name == _outLayerName) {
							_outLayer = index;
        						break;	
      						}
    					}
    					index++;
  				}
				std::cout << "_outLayer:" << _outLayer << "\n";
			}
		}
		return true;
	}
	// -----------------------------------------
	bool GetCore(const XNET &xnet, const XINPUT &xinput, const cv::Mat &imgin, FFEATURE &ft) {
		auto dst_data = xinput->mutable_cpu_data();
		cv::Mat mm;
		cv::resize(imgin, mm, cv::Size(_iw, _ih));
		cv::Mat img;
		mm.convertTo(img, CV_32FC3, 1, 0);
		std::vector<cv::Mat> channels;
		cv::split(img, channels);
  		for (size_t j = 0; j < channels.size(); j++) {
    			channels[j] += (-175);
			to_buffer(channels[j], dst_data);
    			dst_data += _iw*_ih;
		}
		// go
		xnet->ForwardFrom(0);
		xnet->ForwardTo(_outLayer);
		// get
		std::map<std::string, std::pair<const float *, size_t>> output_data;
   		auto output_blob = xnet->blob_by_name(_outLayerName);
    		if (output_blob->count() == 0) {
      			//throw std::runtime_error(name + " blob is empty");
      			printf("blob is empty");
			return false;
    		}
		auto tmp1 = output_blob->cpu_data();
		auto len = output_blob->count();
		std::cout << "begin tmp1:\n" << tmp1 << 
			"\nend tmp1\nbegin len:\n" 
			<< len << "\nend len\n";
		for(int i = 0; i < len; i++){
			if(i>=128){
				continue;
			}
			ft(i) = tmp1[i];
		}
		return true;
  	}

struct XFTS{
public:
	void Push(int id, const FFEATURE &ff){
		boost::mutex::scoped_lock lock(mutex_);
		ffs_.push_back(std::make_pair(id, ff));
	}
	void Get(std::vector<std::pair<int, FFEATURE> > &ffs){
		ffs = ffs_;
	}
private:
	std::vector<std::pair<int, FFEATURE > > ffs_;
	boost::mutex mutex_;
};


	bool FeatureGetter::Get(const cv::Mat &img, const std::vector<cv::Rect> &rcs,
		std::vector<FFEATURE> &fts) {
		int64_t ftm1 = fgtm();	
		XFTS xfts;
#ifndef OONE
		#pragma omp parallel for
#endif
		for(int i = 0; i < rcs.size(); i++){
			cv::Mat tmp = img(rcs[i]);
			FFEATURE ft;
			int64_t ftm11 = fgtm();	
#ifndef OONE
			XNET &xnet = _xnets[i];
			XINPUT &xinput = _xinputs[i];
#else
			XNET &xnet = _xnets[0];
			XINPUT &xinput = _xinputs[0];
#endif
			bool re = GetCore(xnet, xinput, tmp, ft);
			int64_t ftm12 = fgtm();	
			std::cout << "\t----ftm12-ftm11:" << (ftm12-ftm11) << "\n";
			if(!re){
				printf("error!\n");
				exit(0);
			}
			xfts.Push(i, ft);
			fts.push_back(ft);
		}
		//
		fts.resize(rcs.size());
		std::vector<std::pair<int, FFEATURE> > pairs;
		xfts.Get(pairs);
		for(int i = 0; i < pairs.size(); i++){
			std::pair<int, FFEATURE> pa = pairs[i];
			fts[pa.first] = pa.second;	
		}
		int64_t ftm2 = fgtm();
		std::cout << "caffe.forward--shufflenet--rcs.size():" << rcs.size() 
			<< ", ftm2-ftm1:" << (ftm2-ftm1) << "\n";
 
	}


