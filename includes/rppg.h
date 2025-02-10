#ifndef RPPG_H
#define RPPG_H

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iterator>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/types.hpp>

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25

typedef long double ld;
typedef unsigned int uint;
typedef std::vector<ld>::iterator vec_iter_ld;

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const v) {
    os << '[';
    if (!v.empty()) {
        std::copy(
            v.begin(), v.end() -1,
            std::ostream_iterator<T>(os, ", ")
        );
        os << v.back();
    }
    os << ']';

    return os;
};

class VectorStats {
public:
    VectorStats(vec_iter_ld begin, vec_iter_ld end);

    void compute();
    ld mean() const;
    ld standardDeviation() const;

private:
    vec_iter_ld begin_;
    vec_iter_ld end_;
    ld m1{}; // Standard Mean
    ld m2{}; // Deviation mean
};

enum rPPGAlgorithm {
    g, pca, xminay
};

enum faceDetectionAlgorithm {
    haar, deep
};

class RPPG {
public:
    ~RPPG() {
        cv::destroyAllWindows();
    }
    static std::unordered_map<std::string, std::vector<ld>> z_score_threshold(
        std::vector<ld> input, int lag,
        ld threshold, ld influence
    );

    static std::unordered_map<std::string, std::vector<ld>> z_score_threshold(
        std::vector<ld>, int, ld
    );

    auto load(
        rPPGAlgorithm rppga, faceDetectionAlgorithm fda,
       int width, int height, double samplingFrequency,
       double rescanFrequency, int minSignalSize,
       int maxSignalSize,
       const std::string &dnnProtoPath, const std::string &dnnModelPath
    ) -> bool;
    int runDetection();
    float processFrame(cv::Mat& frame, cv::Mat& gray, int time);
    static void exit();

    typedef std::vector<cv::Point2f> Contour2f;

private:
    void detectFace(const cv::Mat &frameRGB, cv::Mat &frameGray);
    void setNearestBox(std::vector<cv::Rect> boxes);
    void detectCorners(cv::Mat& frameGray);
    void trackFace(cv::Mat& frameGray);
    void updateMask(cv::Mat& frameGray);
    void updateROI();
    void extractSignal_g();
    void extractSignal_pca();
    void extractSignal_xminay();
    float estimateHeartrate();
    void invalidateFace();

    // Camera
    int getFps(cv::Mat1d tile, int64_t time) const;

    // Algorithms
    rPPGAlgorithm rppga;
    faceDetectionAlgorithm fda;
    static cv::dnn::Net dnnClassifier;

    // Settings
    cv::Size minFaceSize;
    int maxSignalSize{};
    int minSignalSize{};
    double rescanFrequency{};
    double samplingFrequency{};
    double timeBase{};

    // State variables
    int64_t time{};
    double fps{};
    int high{};
    int64_t lastSamplingTime{};
    int64_t lastScanTime{};
    int low{};

    // int64_t now;
    bool faceValid{};
    bool rescanFlag{};

    // Tracking
    cv::Mat lastFrameGray;
    Contour2f corners;

    // Mask
    cv::Rect box;
    cv::Mat1b mask;
    cv::Rect roi;

    // Raw signal
    cv::Mat1d s;
    cv::Mat1d t;
    cv::Mat1b re;

    // Estimation
    cv::Mat1d s_f;
    cv::Mat1d bpms;
    cv::Mat1d powerSpectrum;
    double bpm = 0.0;
    double meanBpm{};
    double minBpm{};
    double maxBpm{};
};

#endif