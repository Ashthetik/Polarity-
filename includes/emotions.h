#ifndef EMOTIONS_H
#define EMOTIONS_H

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

class NED {
public:
    std::string detectEmotion(cv::VideoCapture webCam);
private:
    float getDistance(
        const cv::Point2f& point1, const cv::Point2f& point2
    );

    std::string getEmotion(
        cv::InputArray frame, cv::InputArray lastFrame,
        cv::CascadeClassifier classifier,
        std::vector<std::vector<cv::Point2f>> landmarks,
        const std::vector<cv::Rect>& faces, 
        const cv::Ptr<cv::face::Facemark>& facemark
    );
};

#endif