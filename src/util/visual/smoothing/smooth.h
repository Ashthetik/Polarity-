#ifndef SMOOTH_H
#define SMOOTH_H

#include <opencv2/imgproc.hpp>

inline int DELAY_CAPTION = 1500;
inline int DELAY_BLUR = 100;
inline int MAX_KERNEL_SIZE = 31;

class Smooth {
public:
    static void smooth(const cv::Mat& frame);
};

#endif