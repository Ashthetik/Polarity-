#include "smooth.h"

#include <iostream>
#include "deblur.h"

void Smooth::smooth(const cv::Mat &frame) {
    const cv::Mat clone = frame.clone();

    cv::Mat dst = MotionBlur::deblur(clone);
    dst = cv::Mat::zeros(frame.size(), frame.type());

    if (!frame.data) {
        std::cerr << "Smooth::smooth(): no data available" << std::endl;
        return;
    }

    for (int i = 0; i <MAX_KERNEL_SIZE; i = i + 2) {
        cv::GaussianBlur(frame, dst, cv::Size(i, i), 0, 0);
    }
}
