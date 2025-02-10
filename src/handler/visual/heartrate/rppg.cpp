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
            Scalar means = mean(frame, mask);

            double values[3] = { means(0), means(1), means(2) };
            s.push_back(Mat(1, 3, CV_64F, values));
            t.push_back(time);

            fps = 30;

            // Save the scan flag
            re.push_back(rescanFlag);

            // Update the band spectrum limits
            low = static_cast<int>(std::floor(s.rows * LOW_BPM / 60 / fps));
            high = static_cast<int>(std::ceil(s.rows * HIGH_BPM / 60 / fps)) + 1;

            if (s.rows >= (fps * minSignalSize) && s.rows) {
                switch (rppga) {
                    case g:
                        extractSignal_g();
                        break;
                    case pca:
                        extractSignal_pca();
                        break;
                    case xminay:
                        extractSignal_xminay();
                        break;
                    default:
                        extractSignal_xminay();
                        break;
                }

                bpm = estimateHeartrate();
            }
        }
    }

    rescanFlag = false;
    frameGray.copyTo(lastFrameGray, mask);

    return bpm;
};


inline void RPPG::exit() {
    cv::destroyAllWindows();
}

/**
 * @brief Applies a Z-score threshold algorithm to detect anomalies in the input signal.
 *
 * This function implements a moving Z-score threshold algorithm to identify
 * anomalies in the input time series data. It calculates the Z-score for each
 * data point and flags it as an anomaly if it exceeds the specified threshold.
 *
 * @param input The input time series data as a vector of long doubles.
 * @param lag The lag parameter for the moving window size.
 * @param threshold The Z-score threshold for anomaly detection.
 *
 * @return An unordered map containing the following key-value pairs:
 *         - "signals": A vector of anomaly flags (-1 for negative anomalies, 1 for positive anomalies, 0 for normal points).
 *         - "filtered_input": The input data with anomalies replaced by the last normal value.
 *         - "filtered_mean": The moving mean of the filtered input.
 *         - "filtered_sd": The moving standard deviation of the filtered input.
 */
inline std::unordered_map<std::string, std::vector<ld>> RPPG::z_score_threshold(
    std::vector<ld> input, int lag, ld threshold
) {
    std::unordered_map<std::string, std::vector<ld>> result = {};
    const uint n = static_cast<uint>(input.size());

    std::vector<ld> signals(input.size());
    std::vector<ld> filtered_input(input.begin(), input.end());
    std::vector<ld> filtered_mean(input.size());
    std::vector<ld> filtered_sd(input.size());

    VectorStats lag_subvector(input.begin(), input.begin() + lag);
    filtered_mean[lag - 1] = lag_subvector.mean();
    filtered_sd[lag - 1] = lag_subvector.standardDeviation();

    #pragma omp parallel for
    for (int i = lag; i < n; i++) {
        if (abs(input[i] - filtered_mean[i - 1]) > threshold * filtered_sd[i - 1]) {
            signals[i] = (input[i] > filtered_mean[i - 1]) ? 1 : -1;
        } else {
            signals[i] = 0;
            filtered_input[i] = input[i];
        }

        VectorStats subvector(filtered_input.begin() + i - lag + 1, filtered_input.begin() + i + 1);
        filtered_mean[i] = subvector.mean();
        filtered_sd[i] = subvector.standardDeviation();
    }

    result["signals"] = signals;
    result["filtered_input"] = filtered_input;
    result["filtered_mean"] = filtered_mean;
    result["filtered_sd"] = filtered_sd;

    return result;
}

void RPPG::detectFace(const Mat& frame, Mat& frameGray) {
    std::vector<Rect> boxes = {};

    Mat resize300;
    cv::resize(frame, resize300, Size(300, 300));
    Mat blob = dnn::blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123));
    
    dnnClassifier.setInput(blob);
    Mat detection = dnnClassifier.forward();
    Mat detectionMatrix(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    // We set a higher threshold for the face detection so that only faces with a high confidence score are selected
    float confidenceThreshold = 0.8;

    #pragma omp parallel for
    for (int i = 0; i < detectionMatrix.rows; i++) {
        float confidence = detectionMatrix.at<float>(i, 2);
        if (confidence > confidenceThreshold) {
            int xLeftBottom = static_cast<int>(detectionMatrix.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMatrix.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMatrix.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMatrix.at<float>(i, 6) * frame.rows);

            Rect object(
                (int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom)
            );
            boxes.push_back(object);
        }
    }

    if (boxes.size() > 0) {
        setNearestBox(boxes);
        detectCorners(frameGray);
        updateROI();
        updateMask(frameGray);
        faceValid = true;
    } else {
        invalidateFace();
    }
}

void RPPG::setNearestBox(std::vector<Rect> boxes) {
    int index = 0;
    Point point = box.tl() - boxes.at(0).tl();

    int min = point.x * point.x + point.y * point.y;

    for (size_t i = 1; i < boxes.size(); i++) {
        point = box.tl() - boxes.at(0).tl();
        int dist = point.x * point.x + point.y * point.y;

        if (dist < min) {
            min = dist;
            index = i;
        }
    }

    box = boxes.at(index);
}

void RPPG::detectCorners(Mat& frameGray) {
    Mat trackingRegion = Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
    Point points[1][4] = {};

    points[0][0] = Point(box.tl().x + 0.22 * box.width, box.tl().y + 0.21 * box.height);
    points[0][1] = Point(box.tl().x + 0.78 * box.width, box.tl().y + 0.21 * box.height);
    points[0][2] = Point(box.tl().x + 0.70 * box.width, box.tl().y + 0.65 * box.height);
    points[0][3] = Point(box.tl().x + 0.30 * box.width, box.tl().y + 0.65 * box.height);


    const Point* corners[1] = {points[0]};
    int num_corners[1] = {4};

    // fillPoly(trackingRegion, corners, num_corners, Scalar(255));

    // goodFeaturesToTrack(frameGray, corners, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, trackingRegion, num_corners, 3, false, 0.04);
}


void RPPG::trackFace(Mat& frameGray) {
    if (corners.empty() || corners.size() < MIN_CORNERS) {
        detectCorners(frameGray);
    }

    Contour2f contour;
    Contour2f contour2;

    std::vector<uchar> cornersFound_0 = {};
    std::vector<uchar> cornersFound_1 = {};

    Mat err;

    calcOpticalFlowPyrLK(lastFrameGray, frameGray, contour, contour2, cornersFound_1, err);
    calcOpticalFlowPyrLK(frameGray, lastFrameGray, contour2, contour, cornersFound_0, err);

    Contour2f corners_1v;
    Contour2f corners_0v;

    #pragma omp parallel for
    for (size_t i = 0; i < corners.size(); i++) {
        if (cornersFound_1[i] && cornersFound_0[i]
            && norm(corners[i] - contour2[i]) < 2
        ) {
            corners_1v.push_back(contour2[i]);
            corners_0v.push_back(corners[i]);
        }
    }

}