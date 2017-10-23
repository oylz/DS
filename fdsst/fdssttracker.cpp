/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#if 1

#include <time.h>

#include "fdssttracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"

#include "fhog.h"

#include "labdata.hpp"
#include <glog/logging.h>

// #define PFS_DEBUG

static double t_start, t_end;


template <typename T>
cv::Mat rangeToColVector(int begin, int end, int n)
{
	cv::Mat_<T> colVec(1, n);

	for (int i = begin, j = 0; i <= end; ++i, j++)
		colVec.template at<T>(0, j) = static_cast<T>(i);

	return colVec;
}


template <typename BT, typename ET>
cv::Mat pow(BT base_, const cv::Mat_<ET>& exponent)
{
	cv::Mat dst = cv::Mat(exponent.rows, exponent.cols, exponent.type());
	int widthChannels = exponent.cols * exponent.channels();
	int height = exponent.rows;

	// http://docs.opencv.org/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-efficient-way
	if (exponent.isContinuous())
	{
		widthChannels *= height;
		height = 1;
	}

	int row = 0, col = 0;
	const ET* exponentd = 0;
	ET* dstd = 0;

	for (row = 0; row < height; ++row)
	{
		exponentd = exponent.template ptr<ET>(row);
		dstd = dst.template ptr<ET>(row);

		for (col = 0; col < widthChannels; ++col)
		{
			dstd[col] = std::pow(base_, exponentd[col]);
		}
	}

	return dst;
}

void shift(const cv::Mat& src, cv::Mat& dst, cv::Point2f delta, int fill, cv::Scalar value = cv::Scalar(0, 0, 0, 0)) {
	// error checking
	CV_Assert(fabs(delta.x) < src.cols && fabs(delta.y) < src.rows);

	// split the shift into integer and subpixel components
	cv::Point2i deltai(static_cast<int>(ceil(delta.x)), static_cast<int>(ceil(delta.y)));
	cv::Point2f deltasub(fabs(delta.x - deltai.x), fabs(delta.y - deltai.y));

	// INTEGER SHIFT
	// first create a border around the parts of the Mat that will be exposed
	int t = 0, b = 0, l = 0, r = 0;
	if (deltai.x > 0) l = deltai.x;
	if (deltai.x < 0) r = -deltai.x;
	if (deltai.y > 0) t = deltai.y;
	if (deltai.y < 0) b = -deltai.y;
	cv::Mat padded;
	cv::copyMakeBorder(src, padded, t, b, l, r, fill, value);

	// SUBPIXEL SHIFT
	float eps = std::numeric_limits<float>::epsilon();
	if (deltasub.x > eps || deltasub.y > eps) {
		switch (src.depth()) {
		case CV_32F:
		{
			cv::Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
			cv::Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
			sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
			break;
		}
		case CV_64F:
		{
			cv::Matx<double, 1, 2> dx(1 - deltasub.x, deltasub.x);
			cv::Matx<double, 2, 1> dy(1 - deltasub.y, deltasub.y);
			sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
			break;
		}
		default:
		{
			cv::Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
			cv::Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
			padded.convertTo(padded, CV_32F);
			sepFilter2D(padded, padded, CV_32F, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
			break;
		}
		}
	}

	// construct the region of interest around the new matrix
	cv::Rect roi = cv::Rect(std::max(-deltai.x, 0), std::max(-deltai.y, 0), 0, 0) + src.size();
	//xyz2017.06.17 cv::Rect roi = cv::Rect(max(-deltai.x, 0), max(-deltai.y, 0), 0, 0) + src.size();
	dst = padded(roi);
}





// Constructor
FDSSTTracker::FDSSTTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{
	Reset();
    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5;
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;

    if (hog) {    // HOG
        // VOT
        interp_factor = 0.015;
        sigma = 0.6;
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5;
        cell_size = 4;
        _hogfeatures = true;

		num_compressed_dim = 13;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4;
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            LOG(ERROR) << "Lab features are only used with HOG features.\n";
            _labfeatures = false;
        }
    }




    if (multiscale) { // multiscale
        template_size = 96;
        //scale parameters initial
        scale_padding = 1.0;
        scale_step = 1.05;
        scale_sigma_factor = 1.0 / 16;

		n_scales = 9;
        n_interp_scales = 33;

        scale_lr = 0.025;
        scale_max_area = 512;
        currentScaleFactor = 1;
        scale_lambda = 0.01;

        if (!fixed_window) {
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
		// begin xyz add ==================
		template_size = 64;
		currentScaleFactor = 1;
		n_scales = 3;
		n_interp_scales = 1;
        	scale_max_area = 256;
		cell_size = 8;
		// end xyz add ==================
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
    success_ = true;
}

// Initialize tracker
// Initialize tracker
void FDSSTTracker::init(const cv::Rect &roi, cv::Mat image)
{
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);
	_tmpl = getFeatures(image, 1);
	if(!success_){
		return;
	}
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);
	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));

	dsstInit(roi, image);
	//_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
	//_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
	train(_tmpl, 1.0); // train with initial frame
}

// Update position based on the new frame
cv::Rect FDSSTTracker::update(cv::Mat image)
{
    if(!success_){
	return cv::Rect(0, 0, 0, 0);
    }
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    float peak_value;

#ifdef PFS_DEBUG
	t_start = clock();
#endif
    cv::Point2f res = detect(getFeatures(image, 0, 1.0f), peak_value);
    if(!success_){
	return cv::Rect(0, 0, 0, 0);
    }
#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "translation detction duration: " << (t_end - t_start) / CLOCKS_PER_SEC << "\n";
#endif
    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale * currentScaleFactor);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale * currentScaleFactor);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    // Update scale

#ifdef PFS_DEBUG
	t_start = clock();
#endif
    cv::Point2i scale_pi = detect_scale(image);
    if(!success_){
        return cv::Rect(0, 0, 0, 0);
    }
#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "scale detction duration: " << (t_end - t_start) / CLOCKS_PER_SEC << "\n";
#endif  
	currentScaleFactor = currentScaleFactor * interp_scaleFactors[scale_pi.x];
//	std::cout << currentScaleFactor<<"\n";
    if(currentScaleFactor < min_scale_factor)
      currentScaleFactor = min_scale_factor;
    // else if(currentScaleFactor > max_scale_factor)
    //   currentScaleFactor = max_scale_factor;

	update_roi();

    train_scale(image);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;


    assert(_roi.width >= 0 && _roi.height >= 0);
    cv::Mat x = getFeatures(image, 0);
    if(!success_){
	return cv::Rect(0, 0, 0, 0);
    }
    train(x, interp_factor);


    return _roi;
}

// Detect the new scaling rate
cv::Point2i FDSSTTracker::detect_scale(cv::Mat image)
{
  cv::Mat xsf = FDSSTTracker::get_scale_sample(image);
  if(!success_){
    return cv::Point2i(0, 0);
  }

  // Compute AZ in the paper
  cv::Mat add_temp;
  cv::reduce(FFTTools::complexMultiplication(sf_num, xsf), add_temp, 0, CV_REDUCE_SUM);

  // compute the final y
  cv::Mat scale_responsef = FFTTools::complexDivisionReal(add_temp, (sf_den + scale_lambda));

  cv::Mat interp_scale_responsef = resizeDFT(scale_responsef, n_interp_scales);

  cv::Mat interp_scale_response;
  cv::idft(interp_scale_responsef, interp_scale_response);

  interp_scale_response = FFTTools::real(interp_scale_response);

  // Get the max point as the final scaling rate
  cv::Point2i pi;
  double pv;
  cv::minMaxLoc(interp_scale_response, NULL, &pv, NULL, &pi);

  return pi;
}

// Detect object in the current frame.
cv::Point2f FDSSTTracker::detect(cv::Mat x, float &peak_value)
{
	if(x.empty()){
		return cv::Point2f(0, 0);
	}
	using namespace FFTTools;

	x = features_projection(x);

	cv::Mat z = features_projection(_tmpl);
#ifdef PFS_DEBUG
	double t_start1 = clock();
#endif
	cv::Mat k = gaussianCorrelation(x, z);
#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "**************gaussianCorrelation duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif 

#ifdef PFS_DEBUG
	t_start = clock();
#endif

	cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "complexMultiplication *******************: " << (t_end - t_start) / CLOCKS_PER_SEC << "\n";
#endif 
	//minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
	cv::Point2i pi;
	double pv;

	cv::minMaxLoc(res, NULL, &pv, NULL, &pi);

	peak_value = (float)pv;

	//subpixel peak estimation, coordinates will be non-integer
	cv::Point2f p((float)pi.x, (float)pi.y);

	if (pi.x > 0 && pi.x < res.cols - 1) {
		p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
	}

	if (pi.y > 0 && pi.y < res.rows - 1) {
		p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
	}



	p.x -= (res.cols) / 2;
	p.y -= (res.rows) / 2;

	return p;
}


// train tracker with a single image
void FDSSTTracker::train(cv::Mat x, float train_interp_factor)
{
	using namespace FFTTools;

	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)* x;

	cv::Mat W, U, VT, X, out;

	X = _tmpl * _tmpl.t();
	cv::SVD::compute(X, W, U, VT);

	VT.rowRange(0, num_compressed_dim).copyTo(proj_matrix);

	x = features_projection(x);
	
	cv::Mat k = gaussianCorrelation(x, x);
	cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

	_alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)* alphaf;


}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat FDSSTTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
   
#ifdef PFS_DEBUG
	double t_start1 = clock();
#endif			
	
	cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
    // HOG features


    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            

			cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);

            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);

        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }

#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "gaussianCorrelation computation A duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif

    cv::Mat d;
	cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 
		0, 
		d);
	//xyz2017.06.17 cvmax(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]),
	//	0, d);
#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "gaussianCorrelation computation B duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif 
    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
   
#ifdef PFS_DEBUG
	t_end = clock();
	std::cout << "gaussianCorrelation computation ALL duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif 
	
	return k;
}


// Create Gaussian Peak. Function called only in the first frame.
cv::Mat FDSSTTracker::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);

	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;

	float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);

	for (int i = 0; i < sizey; i++)
		for (int j = 0; j < sizex; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
		}
	return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat FDSSTTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    if (inithann) {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;

        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }

        if (_hogfeatures) {
            // Round to cell size and also make it even
            _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
            _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
        }
        else {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width * currentScaleFactor;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height * currentScaleFactor;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::resize(z, z, _tmpl_sz);
    }

    // HOG features   
	FeaturesMap = fhog(z,cell_size );
	if(FeaturesMap.empty()){
		success_ = false;
		return FeaturesMap;
	}

	FeaturesMap = FeaturesMap.reshape(1, z.cols * z.rows / (cell_size * cell_size));

	FeaturesMap = FeaturesMap.t();

    if (inithann) {
		size_patch[0] = z.rows / cell_size;
		size_patch[1] = z.cols / cell_size;
		size_patch[2] = num_compressed_dim;
        createHanningMats();
    }


    return FeaturesMap;
}


cv::Mat FDSSTTracker::features_projection(const cv::Mat &FeaturesMap)
{

	cv::Mat out;
	out = proj_matrix * FeaturesMap;

	out = hann.mul(out);

	return out;
}

// Initialize Hanning window. Function called only in the first frame.
void FDSSTTracker::createHanningMats()
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
	// HOG features
	if (_hogfeatures) {
		cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

		hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
		for (int i = 0; i < size_patch[2]; i++) {
			for (int j = 0; j<size_patch[0] * size_patch[1]; j++) {
				hann.at<float>(i, j) = hann1d.at<float>(0, j);
			}
		}
	}
	// Gray features
	else {
		hann = hann2d;
	}
}

// Calculate sub-pixel peak for one dimension
float FDSSTTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}

// Initialization for scales
void FDSSTTracker::dsstInit(const cv::Rect &roi, cv::Mat image)
{
  // The initial size for adjusting
  base_width = roi.width;
  base_height = roi.height;

  // Guassian peak for scales (after fft)

  // 处理插值前的尺度序列，即需要提取多尺度特征的一组值
  cv::Mat colScales =
	  rangeToColVector<float>(-floor((n_scales - 1) / 2),
	  ceil((n_scales - 1) / 2), n_scales);

  colScales *= (float)n_interp_scales / (float)n_scales;

  cv::Mat ss;
  shift(colScales, ss,
	  cv::Point(-floor(((float)n_scales - 1) / 2), 0),
	  cv::BORDER_WRAP, cv::Scalar(0, 0, 0, 0));

  cv::Mat ys;

  float scale_sigma = scale_sigma_factor * n_interp_scales;

  exp(-0.5 * ss.mul(ss) / (scale_sigma * scale_sigma), ys);


  ysf = FFTTools::fftd(ys);

  s_hann = createHanningMatsForScale();

  // Get all scale changing rate
  scaleFactors = pow<float, float>(scale_step, colScales);
  

  // 处理插值后的尺度序列
  cv::Mat interp_colScales =
	  rangeToColVector<float>(-floor((n_interp_scales - 1) / 2),
	  ceil((n_interp_scales - 1) / 2), n_interp_scales);

  cv::Mat ss_interp;
  shift(interp_colScales, ss_interp,
	  cv::Point(-floor(((float)n_interp_scales - 1) / 2), 0),
	  cv::BORDER_WRAP, cv::Scalar(0, 0, 0, 0));

  interp_scaleFactors = pow<float, float>(scale_step, ss_interp);




  // Get the scaling rate for compressing to the model size
  float scale_model_factor = 1;
  if(base_width * base_height > scale_max_area)
  {
    scale_model_factor = std::sqrt(scale_max_area / (float)(base_width * base_height));
  }
  scale_model_width = (int)(base_width * scale_model_factor);
  scale_model_height = (int)(base_height * scale_model_factor);

  // Compute min and max scaling rate
  min_scale_factor = std::pow(scale_step,
    std::ceil(std::log((std::fmax(5 / (float) base_width, 5 / (float) base_height) * (1 + scale_padding))) / 0.0086));
  max_scale_factor = std::pow(scale_step,
    std::floor(std::log(std::fmin(image.rows / (float) base_height, image.cols / (float) base_width)) / 0.0086));

  train_scale(image, true);

}

// Train method for scaling
void FDSSTTracker::train_scale(cv::Mat image, bool ini)
{
  cv::Mat xsf = get_scale_sample(image);
  if(!success_){
	return;
  }
  // Adjust ysf to the same size as xsf in the first time
  if(ini)
  {
    int totalSize = xsf.rows;
    ysf = cv::repeat(ysf, totalSize, 1);
  }

  // Get new GF in the paper (delta A)
  cv::Mat new_sf_num;
  cv::mulSpectrums(ysf, xsf, new_sf_num, 0, true);

  // Get Sigma{FF} in the paper (delta B)
  cv::Mat new_sf_den;
  cv::mulSpectrums(xsf, xsf, new_sf_den, 0, true);
  cv::reduce(FFTTools::real(new_sf_den), new_sf_den, 0, CV_REDUCE_SUM);

  if(ini)
  {
    sf_den = new_sf_den;
    sf_num = new_sf_num;
  }else
  {
    // Get new A and new B
    cv::addWeighted(sf_den, (1 - scale_lr), new_sf_den, scale_lr, 0, sf_den);
    cv::addWeighted(sf_num, (1 - scale_lr), new_sf_num, scale_lr, 0, sf_num);
  }

  update_roi();

}

// Update the ROI size after training
void FDSSTTracker::update_roi()
{
  // Compute new center
  float cx = _roi.x + _roi.width / 2.0f;
  float cy = _roi.y + _roi.height / 2.0f;


  // Recompute the ROI left-upper point and size
  _roi.width = base_width * currentScaleFactor;
  _roi.height = base_height * currentScaleFactor;

  _roi.x = cx - _roi.width / 2.0f;
  _roi.y = cy - _roi.height / 2.0f;

}

// Compute the F^l in the paper
cv::Mat FDSSTTracker::get_scale_sample(const cv::Mat & image)
{
  
  cv::Mat xsf; // output
  int totalSize; // # of features

  for(int i = 0; i < n_scales; i++)
  {
    // Size of subwindow waiting to be detect
    float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
    float patch_height = base_height * scaleFactors[i] * currentScaleFactor;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    // Get the subwindow
    cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
    cv::Mat im_patch_resized;

    // Scaling the subwindow
	if(im_patch.cols<=0 || im_patch.rows<=0 || scale_model_width<=0 || scale_model_height<=0){
		success_ = false;
		return xsf;
	}
    if(scale_model_width > im_patch.cols)
		resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_NEAREST);
    else{
      resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_NEAREST);
      //resize(im_patch, im_patch_resized, cv::Size(im_patch.cols, im_patch.rows), 0, 0, cv::INTER_LINEAR);
    }
    // Compute the FHOG features for the subwindow
	cv::Mat hogs = fhog(im_patch_resized, cell_size);
	if(hogs.empty()){
		success_ = false;
		return xsf;
	}
    if(i == 0)
    {
		totalSize = hogs.cols * hogs.rows * 32;
      xsf = cv::Mat(cv::Size(n_scales,totalSize), CV_32F, float(0));
    }

    // Multiply the FHOG results by hanning window and copy to the output
	cv::Mat FeaturesMap = hogs.reshape(1, totalSize);
    float mul = s_hann.at<float > (0, i);
    FeaturesMap = mul * FeaturesMap;
    FeaturesMap.copyTo(xsf.col(i));

  }

 
  // Do fft to the FHOG features row by row
  xsf = FFTTools::fftd(xsf, 0, 1);

  return xsf;
}

// Compute the FFT Guassian Peak for scaling
cv::Mat FDSSTTracker::computeYsf()
{
    float scale_sigma2 = n_scales / std::sqrt(n_scales) * scale_sigma_factor;
    scale_sigma2 = scale_sigma2 * scale_sigma2;
    cv::Mat res(cv::Size(n_scales, 1), CV_32F, float(0));
    float ceilS = std::ceil(n_scales / 2.0f);

    for(int i = 0; i < n_scales; i++)
    {
      res.at<float>(0,i) = std::exp(- 0.5 * std::pow(i + 1- ceilS, 2) / scale_sigma2);
    }

    return FFTTools::fftd(res);

}

// Compute the hanning window for scaling
cv::Mat FDSSTTracker::createHanningMatsForScale()
{
  cv::Mat hann_s = cv::Mat(cv::Size(n_scales, 1), CV_32F, cv::Scalar(0));
  for (int i = 0; i < hann_s.cols; i++)
      hann_s.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann_s.cols - 1)));

  return hann_s;
}


cv::Mat FDSSTTracker::resizeDFT(const cv::Mat &A, int real_scales)
{
	float scaling = (float)real_scales / n_scales;

	cv::Mat M = cv::Mat(cv::Size(real_scales, 1), CV_32FC2, cv::Scalar(0));

	int mids = ceil(n_scales / 2);
	int mide = floor((n_scales - 1) / 2) - 1;

	A *= scaling;

	A(cv::Range::all(), cv::Range(0, mids)).copyTo(M(cv::Range::all(), cv::Range(0, mids)));

	A(cv::Range::all(), cv::Range(n_scales - mide - 1, n_scales)).copyTo(M(cv::Range::all(), cv::Range(real_scales - mide - 1, real_scales)));

	return M;
}
#endif
