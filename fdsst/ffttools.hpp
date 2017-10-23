/*
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de
Department Augmented Vision DFKI

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

#pragma once

//#include <cv.h>

#ifndef _OPENCV_FFTTOOLS_HPP_
#define _OPENCV_FFTTOOLS_HPP_
#endif
//NOTE: FFTW support is still shaky, disabled for now.
/*#ifdef USE_FFTW
#include <fftw3.h>
#endif*/

namespace FFTTools
{



	cv::Mat fftd(cv::Mat img, bool backwards = false, bool byRow = false)
{

    if (img.channels() == 1)
    {
        cv::Mat planes[] = {cv::Mat_<float> (img), cv::Mat_<float>::zeros(img.size())};
        //cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
        cv::merge(planes, 2, img);
    }
    if(byRow)
      cv::dft(img, img, (cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT));
    else
      cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0 );

    return img;



}

cv::Mat real(cv::Mat img)
{
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    return planes[0];
}

cv::Mat imag(cv::Mat img)
{
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    return planes[1];
}

cv::Mat magnitude(cv::Mat img)
{
    cv::Mat res;
    std::vector<cv::Mat> planes;
    cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    if (planes.size() == 1) res = cv::abs(img);
    else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); // planes[0] = magnitude
    else assert(0);
    return res;
}

cv::Mat complexMultiplication(cv::Mat a, cv::Mat b, bool conj = false)
{
    std::vector<cv::Mat> pa;
    std::vector<cv::Mat> pb;
    cv::split(a, pa);
    cv::split(b, pb);

	if (conj)
		pb[1] *= -1.0;

    std::vector<cv::Mat> pres;
    pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
    pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

    cv::Mat res;
    cv::merge(pres, res);

    return res;
}

cv::Mat complexDivisionReal(cv::Mat a, cv::Mat b)
{
    std::vector<cv::Mat> pa;
    cv::split(a, pa);

    std::vector<cv::Mat> pres;

    cv::Mat divisor = 1. / b;

    pres.push_back(pa[0].mul(divisor));
    pres.push_back(pa[1].mul(divisor));

    cv::Mat res;
    cv::merge(pres, res);
    return res;
}

cv::Mat complexDivision(cv::Mat a, cv::Mat b)
{
    std::vector<cv::Mat> pa;
    std::vector<cv::Mat> pb;
    cv::split(a, pa);
    cv::split(b, pb);

    cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

    std::vector<cv::Mat> pres;

    pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
    pres.push_back((pa[1].mul(pb[0]) + pa[0].mul(pb[1])).mul(divisor));

    cv::Mat res;
    cv::merge(pres, res);
    return res;
}

void rearrange(cv::Mat &img)
{
    // img = img(cv::Rect(0, 0, img.cols & -2, img.rows & -2));
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right
    cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left
    cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
/*
template < typename type>
cv::Mat fouriertransFull(const cv::Mat & in)
{
    return fftd(in);

    cv::Mat planes[] = {cv::Mat_<type > (in), cv::Mat_<type>::zeros(in.size())};
    cv::Mat t;
    assert(planes[0].depth() == planes[1].depth());
    assert(planes[0].size == planes[1].size);
    cv::merge(planes, 2, t);
    cv::dft(t, t);

    //cv::normalize(a, a, 0, 1, CV_MINMAX);
    //cv::normalize(t, t, 0, 1, CV_MINMAX);

    // cv::imshow("a",real(a));
    //  cv::imshow("b",real(t));
    // cv::waitKey(0);

    return t;
}*/

void normalizedLogTransform(cv::Mat &img)
{
    img = cv::abs(img);
    img += cv::Scalar::all(1);
    cv::log(img, img);
    // cv::normalize(img, img, 0, 1, CV_MINMAX);
}

typedef std::vector<cv::Mat> ComplexMats;

ComplexMats MultiChannelsDFT(const cv::Mat &img, int flags = 0)
{
	std::vector<cv::Mat> chls;
	std::vector<cv::Mat> out;
	cv::split(img, chls);
	out.resize(chls.size());
	for (int i = 0; i < chls.size(); i++)
	{
		cv::dft(chls[i], out[i], cv::DFT_COMPLEX_OUTPUT);
		//out[i] = (out[i]);
	}
	// cv::Mat out_m;
	// cv::merge(out, out_m);
	return out;
}

ComplexMats ComplexMatsMultiMat(const ComplexMats &A, cv::Mat b)
	{
		
		ComplexMats out;
		out.resize(A.size());
		for (int i = 0; i < A.size(); i++)
		{
			out[i] = complexMultiplication(b, A[i]);
		}
		return out;
	}


ComplexMats ComplexMatsMultiComplexMats(const ComplexMats &A, const ComplexMats &B)
{

	ComplexMats out;
	assert(A.size() == B.size());
	out.resize(A.size());
	for (int i = 0; i < A.size(); i++)
	{
		out[i] = complexMultiplication(A[i], B[i]);
	}
	return out;
}

     ComplexMats MCComplexConjMultiplication(const ComplexMats &A)
	{

		ComplexMats out;
		out.resize(A.size());
		for (int i = 0; i < A.size(); i++)
		{
			out[i] = (complexMultiplication(A[i], A[i], true));
		}
		//cv::Mat out_m;
		//cv::merge(out, out_m);
		return out;
	}

	cv::Mat MCMulti(cv::Mat a, cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		cv::split(a, pa);

		std::vector<cv::Mat> pres;

		pres.resize(pa.size());

		for (int i = 0; i < pa.size(); i++)
			pres[i] = pa[i].mul(b);
		cv::Mat res;
		cv::merge(pres, res);

		return res;
	}


	cv::Mat MCSum(const ComplexMats &a)
	{
		//std::vector<cv::Mat> pa;
		//cv::split(a, pa);
		assert(a.size() != 0);
		cv::Mat out;
		a[0].copyTo(out);
		for (int i = 1; i < a.size(); i++)
			out = out + a[i];

		return out;
	}

	cv::Mat MCSum(const cv::Mat &a)
	{
		std::vector<cv::Mat> pa;
		cv::split(a, pa);
		assert(pa.size() != 0);
		cv::Mat out;
		pa[0].copyTo(out);
		for (int i = 1; i < pa.size(); i++)
			out = out + pa[i];

		return out;
	}

}
