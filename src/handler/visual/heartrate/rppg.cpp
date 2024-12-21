#include "rppg.h"
#include <cstdio>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

using namespace cv;

inline VectorStats::VectorStats(vec_iter_ld start, vec_iter_ld end)
{
    this->begin_ = start;
    this->end_ = end;
    this->compute();
}

inline void VectorStats::compute()
{
    ld sum = std::accumulate(begin_, end_, 0.0);
    uint distance = std::distance(begin_, end_);
    ld mean = sum / distance;
    std::vector<ld> diff(distance);

    std::transform(begin_, end_, diff.begin(), [mean](ld x) { return x - mean; });

    ld sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(),
        0.0); // Differential Inner Product
    ld std_dev = std::sqrt(sq_sum / distance); // Standard Deviation

    this->m1 = mean;
    this->m2 = std_dev;
}

inline ld VectorStats::mean() const { return m1; }

inline ld VectorStats::standardDeviation() const { return m2; }

inline bool RPPG::load(const rPPGAlgorithm rppga,
    const faceDetectionAlgorithm fda, const int width,
    const int height, const double samplingFrequency,
    const double rescanFrequency, const int minSignalSize,
    const int maxSignalSize, const std::string& dnnProtoPath,
    const std::string& dnnModelPath)
{
    try {
        this->rppga = rppga;
        this->fda = fda;
        this->lastSamplingTime = 0;
        this->minFaceSize = cv::Size(std::min(width, height) * REL_MIN_FACE_SIZE,
            std::min(width, height) * REL_MIN_FACE_SIZE);
        this->maxSignalSize = maxSignalSize;
        this->minSignalSize = minSignalSize;
        this->rescanFlag = false;
        this->rescanFrequency = rescanFrequency;
        this->samplingFrequency = samplingFrequency;
        this->timeBase = timeBase;

        dnnClassifier = cv::dnn::readNetFromCaffe(dnnProtoPath, dnnModelPath);

        return true; // Return a successful load
    } catch (std::exception& e) {
        std::cerr << "[!] Runtime Error: " << e.what() << std::endl;
        return false;
    }
}

inline int RPPG::runDetection()
{
    try {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr
                << "[!] Runtime Error: Could not Open Camera. Is it already open?"
                << std::endl;
            return 1; // 1 = Error
        }

        Mat frame, gray;

        if (frame.empty()) {
            std::cerr << "[!] Runtime Error: Could not read frame, it's empty."
                      << std::endl;
            return 1;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        this->processFrame(frame, gray, 0);
        return 0;
    } catch (std::exception& e) {
        std::cerr << "[!] Runtime Error: " << e.what() << std::endl;
        return 1;
    }
}

inline Mat upscaleImage(const Mat& img, const std::string& modelName,
    const std::string& modelPath, const int scale)
{
    dnn_superres::DnnSuperResImpl sr;
    sr.readModel(modelPath);
    sr.setModel(modelName, scale);
    Mat result;
    sr.upsample(img, result);
    return result;
}

inline float RPPG::processFrame(Mat& frame, Mat& frameGray, int time)
{
    float bpm = 0.0;

    frame = upscaleImage(frame, "lapsrn", "LapSRN_x8.pb", 8);
    frameGray = upscaleImage(frameGray, "lapsrn", "LapSRN_x8.pb", 8);

    this->time = time;

    if (!faceValid) {
        lastScanTime = time;
        detectFace(frame, frameGray);
    } else if (
        // If the sum of the current scan time is equivalent to
        // the time base divided by the recurring frequency, rescan
        (time - lastScanTime) * timeBase >= (1 / rescanFrequency)) {

    } else {
        trackFace(frameGray);

        if (faceValid) {
            // TODO: Finish RPPG
            // fps =
        }
    }
};
