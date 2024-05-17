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

std::vector<float> distTo33;
std::vector<float> lastDistTo33;

float dist20to33 = distTo33[20] - lastDistTo33[20];
float dist21to33 = distTo33[21] - lastDistTo33[21];
float dist22to33 = distTo33[22] - lastDistTo33[22];
float dist23to33 = distTo33[23] - lastDistTo33[23];
float dist17to36 = distTo33[17] - lastDistTo33[17];
float dist26to45 = distTo33[26] - lastDistTo33[26];
float dist19to37 = distTo33[19] - lastDistTo33[19];
float dist24to44 = distTo33[24] - lastDistTo33[24];
float dist38to41 = distTo33[38] - lastDistTo33[38];
float dist43to46 = distTo33[43] - lastDistTo33[43];
float dist39to42 = distTo33[39] - lastDistTo33[39];
float dist44to47 = distTo33[44] - lastDistTo33[44];
float dist48to60 = distTo33[48] - lastDistTo33[60];
float dist54to64 = distTo33[54] - lastDistTo33[64];
float dist51to57 = distTo33[51] - lastDistTo33[57];
float dist61to65 = distTo33[61] - lastDistTo33[65];
float dist62to66 = distTo33[62] - lastDistTo33[66];
float dist48to66 = distTo33[48] - lastDistTo33[66];
float dist54to62 = distTo33[54] - lastDistTo33[62];
float dist61to67 = distTo33[61] - lastDistTo33[67];
float dist62to58 = distTo33[62] - lastDistTo33[58];
float dist63to65 = distTo33[63] - lastDistTo33[65];
float dist56to58 = distTo33[56] - lastDistTo33[58];
float dist63to67 = distTo33[63] - lastDistTo33[67];
float dist56to58_2 = distTo33[56] - lastDistTo33[58];
float dist33to38 = distTo33[33] - lastDistTo33[38];
float dist33to37 = distTo33[33] - lastDistTo33[37];
float dist33to43 = distTo33[33] - lastDistTo33[43];
float dist33to44 = distTo33[33] - lastDistTo33[44];
float dist8to33 = distTo33[8] - lastDistTo33[33];
float dist7to33 = distTo33[7] - lastDistTo33[33];
float dist9to33 = distTo33[9] - lastDistTo33[33];
float dist21to22 = distTo33[21] - lastDistTo33[22];
float dist20to23 = distTo33[20] - lastDistTo33[23];
float dist5to48 = distTo33[5] - lastDistTo33[48];
float dist11to54 = distTo33[11] - lastDistTo33[54];
float dist33to56 = distTo33[33] - lastDistTo33[56];
float dist33to57 = distTo33[33] - lastDistTo33[57];
float dist33to58 = distTo33[33] - lastDistTo33[58];
float dist27to40 = distTo33[27] - lastDistTo33[40];
float dist27to47 = distTo33[27] - lastDistTo33[47];
float dist50to61 = distTo33[50] - lastDistTo33[61];
float dist51to62 = distTo33[51] - lastDistTo33[62];
float dist52to63 = distTo33[52] - lastDistTo33[63];
float dist36to48 = distTo33[36] - lastDistTo33[48];
float dist45to54 = distTo33[45] - lastDistTo33[54];

#endif