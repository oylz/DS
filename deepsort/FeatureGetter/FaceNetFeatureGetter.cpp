
#include "FeatureGetter.h"


#include <tensorflow/core/public/session.h>
#include <fstream>
#include <iostream>

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/c/checkpoint_reader.h>
#include <tensorflow/c/c_api_internal.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/cc/ops/math_ops.h>
namespace tf = tensorflow;

#include <sys/time.h>


static int64_t fgtm() {
	struct timeval tm;
	gettimeofday(&tm, 0);
	int64_t re = ((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec;
	return re;
}


boost::shared_ptr<FeatureGetter> FeatureGetter::self_;

typedef float uint8;

std::unique_ptr<tf::Session> session;

void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf) {
	int pos = 0;
	for (cv::Mat img : imgs) {
		int LLL = img.cols*img.rows * 3;
		int nr = img.rows;
		int nc = img.cols;
		if (img.isContinuous())
		{
			nr = 1;
			nc = LLL;
		}

		for (int i = 0; i < nr; i++)
		{
			const uchar* inData = img.ptr<uchar>(i);
			for (int j = 0; j < nc; j++)
			{
				buf[pos] = *inData++;
				pos++;
			}
		}
	}
}
void tobufferA(const std::vector<cv::Mat> &imgs, float *buf){
	int pos = 0;
	for (cv::Mat img : imgs) {
    		if (img.isContinuous()) {
      			memcpy(buf+pos, img.ptr<float>(0),
             			static_cast<size_t>(img.total()) * sizeof(float));
			pos += static_cast<size_t>(img.total()) * sizeof(float);
    		} 
		else {
			printf("error\n");
			exit(0);
      		}
	}
}




typedef std::vector<double> DSR;
typedef std::vector<DSR> DSRS;
typedef std::vector<int> IDSR;
typedef std::vector<IDSR> IDSRS;


	bool FeatureGetter::Init() {
        tf::Session* session_ptr;
        auto status = NewSession(tf::SessionOptions(), &session_ptr);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return false;
        }
        session.reset(session_ptr);

        //------------------
        tf::GraphDef graph_def;

        auto status1 = ReadBinaryProto(tf::Env::Default(), "./data/facenet.pb", &graph_def);
        if (!status1.ok()) {
            printf("ReadBinaryProto failed: %s\n", status1.ToString().c_str());
			return false;
        }

        status = session->Create(graph_def);
        if (!status.ok()) {
            printf("create graph in session failed: %s\n", status.ToString().c_str());
			return false;
        }

        std::vector<std::string> node_names;
        for (const auto &node : graph_def.node()) {
		printf("node name:%s\n", node.name().c_str());
            node_names.push_back(node.name());
        }
        
		return true;
	}
	bool FeatureGetter::Get(const cv::Mat &img, const std::vector<cv::Rect> &rcs,
		std::vector<FFEATURE> &fts) {
        std::vector<cv::Mat> mats;
        for(cv::Rect rc:rcs){
            cv::Mat mat1 = img(rc).clone();
            cv::resize(mat1, mat1, cv::Size(160, 160));

 	/*auto face_mat = face.get_face_image()                                                            
                          .resize(input_width, input_height)                                           
                          .convert_to(CV_32FC3)                                                        
                          .get_cv_mat();*/
/*	cv::Mat face_mat;
	//模型希望是rgb顺序
	cv::cvtColor(mat1, face_mat, CV_BGR2RGB);
	face_mat = face_mat.reshape(1);

	// whitten
	cv::Scalar means;
	cv::Scalar stds;
	cv::meanStdDev(face_mat, means, stds);
 
	stds[0] =
    		std::max(float(stds[0]), 1.0f / sqrtf(160 * 160 * 3));
	face_mat -= means[0];
	face_mat /= stds[0];

		cv::Mat tmp = face_mat.clone();
            mats.push_back(tmp);
*/
		mats.push_back(mat1);
        }
        int count = mats.size();
        
        tensorflow::Tensor input_tensor0(tensorflow::DT_FLOAT, { count, 160, 160, 3 });
        tobuffer(mats, input_tensor0.flat<uint8>().data());

        std::vector<tensorflow::Tensor> output_tensors;

        std::vector<std::pair<std::string, tensorflow::Tensor>> ins;
        std::pair<std::string, tensorflow::Tensor> pa;
        pa.first = "input";
        pa.second = input_tensor0;
        ins.push_back(pa);
	{
        	std::pair<std::string, tensorflow::Tensor> pa1;
        	pa1.first ="phase_train";
		tf::Tensor phase_train(tf::DT_BOOL, tf::TensorShape());
		phase_train.scalar<bool>()() = false;
		pa1.second = phase_train;
		ins.push_back(pa1);
	}
        std::vector<std::string> outnames;
        outnames.push_back("embeddings");
        std::vector<std::string> ts;
	int64_t ftm1 = fgtm();	
        auto status = session->Run(
            ins,
            outnames,
            ts,
            &output_tensors);
	int64_t ftm2 = fgtm();
	std::cout << "session.run----rcs.size():" << rcs.size() << ", ftm2-ftm1:" << (ftm2-ftm1) << "\n";
        if (!status.ok()) {
            printf("error 3%s \n", status.ToString().c_str());
            return false;
        }
        float *tensor_buffer = 
            output_tensors[0].flat<float>().data();
        int len = output_tensors[0].flat<float>().size() / count;
        for (int i = 0; i < count; i++) {
            //printf("begin====\n");
			FFEATURE ft;
            for (int j = 0; j < len; j++) {
				ft(j) = tensor_buffer[i*len + j];
                //printf(",%f", tensor_buffer[i*len+j]);
            }
			fts.push_back(ft);
            //printf("\nend====\n");
        }            
		return true;
	}



