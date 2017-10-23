#ifndef PYKF
#ifndef _KKALMANFILTERH_
#define _KKALMANFILTERH_
#include <boost/shared_ptr.hpp>

typedef Eigen::Matrix<float, 4, 8, Eigen::RowMajor> UPM;

class KF{
private:
    static boost::shared_ptr<KF> self_;
    VAR _motion_mat_;
    UPM _update_mat_;
    double _std_weight_position_;
    double _std_weight_velocity_;
public:
	static boost::shared_ptr<KF> Instance() {
		if (self_.get() == NULL) {
			self_.reset(new KF());
		}
		return self_;
	}
    bool Init(){
        return true;
    }
private:        
     KF(){
        int ndim = 4;
        double dt = 1.;

        _motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
        for(int i = 0; i < ndim; i++){
            _motion_mat_(i, ndim + i) = dt;
        }
        _update_mat_ = Eigen::MatrixXf::Identity(4, 8);
        
        _std_weight_position_ = 1. / 20;
        _std_weight_velocity_ = 1. / 160;
    }
	VAR Diag(const MEAN &mean) const{
	        VAR var;
	        for(int i = 0; i < var.rows(); i++){
	                for(int j = 0; j < var.cols(); j++){
	                        if(i == j){
	                                var(i, j) = mean(i);
	                        }
	                        else{
	                                var(i, j) = 0;
	                        }
	                }
	        }
		return var;
	}
	NVAR NDiag(const NMEAN &mean) const{
	        NVAR var;
	        for(int i = 0; i < var.rows(); i++){
	                for(int j = 0; j < var.cols(); j++){
	                        if(i == j){
	                                var(i, j) = mean(i);
	                        }
	                        else{
	                                var(i, j) = 0;
	                        }
	                }
	        }
		return var;
	}
public:
    std::pair<MEAN, VAR> initiate(const DSBOX &measurement) const{
        DSBOX mean_pos = measurement;
        DSBOX mean_val;
        for(int i = 0; i < 4; i++){
            mean_val(i) = 0;
        } 
        MEAN mean;
        for(int i = 0; i < 8; i++){
            if(i < 4){
                mean(i) = mean_pos(i);
                continue;
            }
            mean(i) = mean_val(i - 4);
        }


        MEAN std;
        std(0) = 2 * _std_weight_position_ * measurement[3];
        std(1) = 2 * _std_weight_position_ * measurement[3];
        std(2) = 1e-2;
        std(3) = 2 * _std_weight_position_ * measurement[3];
        std(4) = 10 * _std_weight_velocity_ * measurement[3];
        std(5) = 10 * _std_weight_velocity_ * measurement[3];
        std(6) = 1e-5;
        std(7) = 10 * _std_weight_velocity_ * measurement[3];


        MEAN tmp = std.array().square();
        
        VAR var = Diag(tmp); 
#ifdef KLOG
      	std::cout << "[-4--]begin mean:\n"  << mean << "\n[-4--]end mean\n";
	std::cout << "[-4--]begin covariance:\n" << var << "\n[-4--]end covariance\n";
#endif
        std::pair<MEAN, VAR> pa;
        pa.first = mean;
        pa.second = var;
        return pa;
    }

    std::pair<MEAN, VAR> predict(const MEAN &mean, const VAR &covariance) const{
        DSBOX std_pos;
        std_pos <<  
            _std_weight_position_ * mean(3),
            _std_weight_position_ * mean(3),
            1e-2,
            _std_weight_position_ * mean(3);

        DSBOX std_vel;
        std_vel << 
            _std_weight_velocity_ * mean(3),
            _std_weight_velocity_ * mean(3),
            1e-5,
            _std_weight_velocity_ * mean(3);

        MEAN mtmp;
        for(int i = 0; i < 8; i++){
            if(i < 4){
                mtmp(i) = std_pos(i);
                continue;
            }
            mtmp(i) = std_vel(i - 4);
        }
        MEAN tmp = mtmp.array().square();
        VAR motion_cov = Diag(tmp);
#ifdef KLOG 
        std::cout << "[-3--]begin square\n";
        std::cout << tmp << "\n";
        std::cout << "[-3--]end square\n";
#endif 
        //
        MEAN mean1 = _motion_mat_ * mean.transpose();
#ifdef KLOG

        std::cout << "[-3--]begin self._motion_mat_\n";
        std::cout << _motion_mat_ << "\n";
        std::cout << "[-3--]end self._motion_mat_\n";
        std::cout << "[-3--]begin covariance\n";
        std::cout << covariance << "\n";
        std::cout << "[-3--]end covariance\n";
    
        std::cout << "[-3--]begin motion_cov:\n";
        std::cout << motion_cov << "\n";
        std::cout << "[-3--]end motion_cov:\n";
#endif

        VAR var = _motion_mat_ * covariance * (_motion_mat_.transpose());
	VAR var1 = var + motion_cov;
#ifdef KLOG

        std::cout << "[-3--]begin covariance result\n";
        std::cout << var1 << "\n";
        std::cout << "[-3--]end covariance result\n";
#endif 
        std::pair<MEAN, VAR> pa;
        pa.first = mean1;
        pa.second = var1;
        return pa;
    }

    std::pair<MEAN, VAR> update(const MEAN &mean,  const VAR &covariance, const DSBOX &measurement) const{
       
        std::pair<NMEAN, NVAR> pa1 = _project(mean, covariance); 
        NMEAN projected_mean = pa1.first;
        NVAR projected_cov = pa1.second;

        auto ddd = covariance * (_update_mat_.transpose());
        Eigen::Matrix<float, -1, 4> kalman_gain = projected_cov.llt().solve(ddd.transpose()).transpose(); // eg.8x4
        Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
#ifdef KLOG

        std::cout << "[-1--]bbegin ddd\n";
        std::cout << ddd << "\n";
        std::cout << "[-1--]bend ddd\n";
        std::cout << "[-1--]bbegin kalman_gain\n";
        std::cout << kalman_gain << "\n";
        std::cout << "[-1--]bend kalman_gain\n";
        std::cout << "[-1--]begin measurement\n";
        std::cout << measurement << "\n";
        std::cout << "[-1--]end measurement\n";
        std::cout << "[-1--]begin projected_mean\n";
        std::cout << projected_mean << "\n";
        std::cout << "[-1--]end projectd_mean\n";
        std::cout << "[-1--]begin innovation\n";
        std::cout << innovation << "\n";
        std::cout << "[-1--]end innovation\n";
        std::cout << "[-1--]begin projected_cov\n";
        std::cout << projected_cov << "\n";
        std::cout << "[-1--]end projectd_cov\n";
#endif

	auto tmp = innovation*(kalman_gain.transpose());
        MEAN new_mean = (mean.array() + tmp.array()).matrix();
        VAR new_covariance = covariance - kalman_gain*projected_cov*(kalman_gain.transpose());
        std::pair<MEAN, VAR> pa2;
        pa2.first = new_mean;
        pa2.second = new_covariance;
        return pa2;
    }
    Eigen::Matrix<float, 1, -1> gating_distance(const MEAN &meani, const VAR &covariance, 
                        const DSBOXS &measurements,
                        bool only_position=false) const{
        MEAN mean = meani; 
#ifdef KLOG

        std::cout << "[-2--]begin mean\n";
        std::cout << mean << "\n";
        std::cout << "[-2--]end mean\n";
        std::cout << "[-2--]begin covariance\n";
        std::cout << covariance << "\n";
        std::cout << "[-2--]end covariance\n";
        std::cout << "[-2--]begin measurements\n";
        std::cout << measurements << "\n";
        std::cout << "[-2--]end measurements\n";
#endif 
        std::pair<NMEAN, NVAR>  pa1 = _project(mean, covariance);
        if(only_position){
             printf("not implement!!!exit\n");
             exit(0);
        }
        NMEAN mean1 = pa1.first;
        NVAR var1 = pa1.second;
#ifdef KLOG

        std::cout << "[-2--]begin mean1\n";
        std::cout << mean1 << "\n";
        std::cout << "[-2--]end mean1\n";
        std::cout << "[-2--]begin covariance1\n";
        std::cout << var1 << "\n";
        std::cout << "[-2--]end covariance1\n";
#endif 
	int count = measurements.rows();
	DSBOXS d(count, 4);
	for(int i = 0; i < count; i++){
		d.row(i) = measurements.row(i) - mean1;
	}
#ifdef KLOG

        std::cout << "[-2--]bbegin d\n";
        std::cout << d << "\n";
        std::cout << "[-2--]bend d\n";
#endif
 	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = var1.llt().matrixL();
        Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
#ifdef KLOG

        std::cout << "[-2--]begin z\n";
        std::cout << z << "\n";
        std::cout <<  "[-2--]end z\n";
#endif
#if 1
	auto zz = ((z.array())*(z.array())).matrix();
	auto squared_maha = zz.colwise().sum();
#else 
        Eigen::Matrix<float, 1, -1> squared_maha = z.colwise().sum();
#endif
#ifdef KLOG

        std::cout << "[-2--]begin squared_maha\n";
        std::cout << squared_maha << "\n";
        std::cout << "[-2--]end squared_maha\n";
#endif 
        return squared_maha;

    } 
    std::pair<NMEAN, NVAR> _project(const MEAN &mean, const VAR &covariance) const{
        NMEAN std;
        std <<
            _std_weight_position_ * mean[3],
            _std_weight_position_ * mean[3],
            1e-1,
            _std_weight_position_ * mean[3];
        NMEAN mtmp = std.array().square();
#ifdef KLOG

        std::cout << "[-0--]begin mtmp\n";
        std::cout << mtmp << "\n";
        std::cout << "[-0--]end mtmp\n";
#endif

        NVAR innovation_cov = NDiag(mtmp);
        NMEAN mean1 = _update_mat_*mean.transpose();
#ifdef KLOG

        std::cout << "[-0--]begin innovation_cov\n";
        std::cout << innovation_cov << "\n";
        std::cout << "[-0--]end innovation_cov\n";

        std::cout << "[-0--]begin var\n";
        std::cout << covariance << "\n";
        std::cout << "[-0--]end var\n";

        std::cout << "[-0--]begin _update_mat_\n";
        std::cout << _update_mat_ << "\n";
        std::cout << "[-0--]end _update_mat_\n";
#endif

        NVAR var = _update_mat_ * covariance * (_update_mat_.transpose());
        NVAR var1 = var + innovation_cov;
#ifdef KLOG

        std::cout << "[-0--]begin var1\n";
        std::cout << var << "\n";
        std::cout << "[-0--]end var1\n";
#endif

        std::pair<NMEAN, NVAR> pa;
        pa.first = mean1;
        pa.second = var1;
        return pa;
    }

};

#endif 
#endif



