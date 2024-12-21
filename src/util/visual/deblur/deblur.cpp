#include "deblur.h"
#include <iostream>

using namespace std;
using namespace cv;

cv::Mat MotionBlur::deblur(cv::Mat frame)
{
    constexpr int len = 125;
    constexpr double theta = 0;
    constexpr double snr = 700;

    if (frame.empty()) {
        cout << "[ERROR] [Deblur] Empty Frame!" << endl;
        return {};
    }

    cv::Mat output;
    cv::Mat Hw, h;

    // Region of Interest
    const cv::Rect roi(0, 0, frame.cols & -2, frame.rows & -2);

    calculatePSF(h, roi.size(), len, theta);
    calculateWnrFilter(h, Hw, 1.0 / snr);

    frame.convertTo(output, CV_32F);
    edgeTaper(frame, frame);

    filter2DFreq(frame(roi), output, Hw);
    output.convertTo(output, CV_8U);
    cv::normalize(output, output, 0, 255, NORM_MINMAX);

    return output;
}

void MotionBlur::calculatePSF(cv::Mat output, const cv::Size filter, const int len, const double theta)
{
    cv::Mat h(filter, CV_32F, cv::Scalar(0));
    const cv::Point point(filter.width / 2, filter.height / 2);
    cv::ellipse(h, point, cv::Size(0, cvRound(static_cast<float>(len) / 2.0)), 90.0 - theta, 0, 360, cv::Scalar(255), FILLED);
    cv::Scalar summa = sum(h);

    output = h / summa[0];
}

void MotionBlur::FFTShift(const cv::Mat& input, cv::Mat& output)
{
    output = input.clone();
    const int cx = output.cols / 2;
    const int cy = output.rows / 2;

    cv::Mat q0(output, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(output, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(output, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(output, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void MotionBlur::filter2DFreq(const cv::Mat& input, cv::Mat& output, const cv::Mat& H)
{
    cv::Mat planes[2] = {
        cv::Mat_<float>(input.clone()),
        cv::Mat::zeros(input.size(), CV_32F)
    };

    cv::Mat complexI;
    merge(planes, 2, complexI);
    cv::dft(complexI, complexI, DFT_SCALE);

    const cv::Mat planesH[2] = {
        cv::Mat_<float>(H.clone()),
        cv::Mat::zeros(H.size(), CV_32F)
    };

    cv::Mat complexH;
    cv::merge(planesH, 2, complexH);

    cv::Mat complexIH;
    cv::mulSpectrums(complexI, complexH, complexIH, 0);

    cv::idft(complexIH, complexIH);
    cv::split(complexIH, planes);

    output = planes[0];
}

void MotionBlur::calculateWnrFilter(const cv::Mat& input, cv::Mat& output, const double nsr)
{
    cv::Mat h;
    FFTShift(input, h);

    Mat planes[2] = {
        cv::Mat_<float>(h.clone()),
        cv::Mat::zeros(h.size(), CV_32F)
    };
    cv::Mat complexI;

    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);

    cv::Mat denom;
    cv::pow(cv::abs(planes[0]), 2, denom);
    denom += nsr;

    cv::divide(planes[0], denom, output);
}

void MotionBlur::edgeTaper(const cv::Mat& input, cv::Mat& output, const double gamma, const double beta)
{
    const int nx = input.cols;
    const int ny = input.rows;
    cv::Mat w1(1, nx, CV_32F, cv::Scalar(0));
    cv::Mat w2(1, ny, CV_32F, cv::Scalar(0));

    auto* p1 = w1.ptr<float>(0);
    auto* p2 = w2.ptr<float>(0);

    const auto dx = static_cast<float>(2.0 * CV_PI / nx);
    auto x = static_cast<float>(-CV_PI);

#pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        p1[i] = static_cast<float>(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
        x += dx;
    }

    const auto dy = static_cast<float>(2.0 * CV_PI / ny);
    auto y = static_cast<float>(-CV_PI);

#pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        p2[i] = static_cast<float>(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));

        y += dy;
    }

    const cv::Mat w = w2 * w1;
    cv::multiply(input, w, output);
}
