#ifndef DEBLUR_H
#define DEBLUR_H

#include <opencv2/imgproc.hpp>

class MotionBlur {
public:
    static cv::Mat deblur(cv::Mat frame);

    static void calculatePSF(cv::Mat output, cv::Size filter, int len, double theta);

    static void FFTShift(const cv::Mat& input, cv::Mat& output);

    static void filter2DFreq(const cv::Mat& input, cv::Mat& output, const cv::Mat& H);

    static void calculateWnrFilter(const cv::Mat& input, cv::Mat& output, double nsr);

    static void edgeTaper(const cv::Mat& input, cv::Mat& output, double gamma = 0.5, double beta = 0.2);
private:
};

#endif
