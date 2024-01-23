#ifndef RPPG_hpp
#define RPPG_hpp

#include <fstream>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include "opencv.hpp"

//---added zScore code
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iterator>
#include <numeric>

typedef long double ld;
typedef unsigned int uint;
typedef std::vector<ld>::iterator vec_iter_ld;

using namespace cv;
using namespace dnn;
using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25

/**
 * Overriding the ostream operator for pretty printing vectors.
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> vec) {
    os << "[";
    if (vec.size() != 0) {
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(os, " "));
        os << vec.back();
    }
    os << "]";
    return os;
}

/**
 * This class calculates mean and standard deviation of a subvector.
 * This is basically stats computation of a subvector of a window size qual to "lag".
 */

class VectorStats {
public:
    /**
     * Constructor for VectorStats class.
     *
     * @param start - This is the iterator position of the start of the window,
     * @param end   - This is the iterator position of the end of the window,
     */
    VectorStats(vec_iter_ld start, vec_iter_ld end) {
        this->start = start;
        this->end = end;
        this->compute();
    }
    
    /**
     * This method calculates the mean and standard deviation using STL function.
     * This is the Two-Pass implementation of the Mean & Variance calculation.
     */
    void compute() {
        ld sum = std::accumulate(start, end, 0.0);
        uint slice_size = std::distance(start, end);
        ld mean = sum / slice_size;
        std::vector<ld> diff(slice_size);
        std::transform(start, end, diff.begin(), [mean](ld x) { return x - mean; });
        ld sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        ld std_dev = std::sqrt(sq_sum / slice_size);
        
        this->m1 = mean;
        this->m2 = std_dev;
    }
    
    ld mean() {
        return m1;
    }
    
    ld standard_deviation() {
        return m2;
    }
    
private:
    vec_iter_ld start;
    vec_iter_ld end;
    ld m1;
    ld m2;
};

//----original RPPG code
using namespace cv;
using namespace dnn;
using namespace std;

enum rPPGAlgorithm { g, pca, xminay };
enum faceDetAlgorithm { haar, deep };

class RPPG {

public:

    // Constructor
    RPPG() {;}

    // Load Settings
    bool load(const rPPGAlgorithm rPPGAlg, const faceDetAlgorithm faceDetAlg,
        const int width, const int height, const double timeBase, const int downsample,
        const double samplingFrequency, const double rescanFrequency,
        const int minSignalSize, const int maxSignalSize,
        const string &haarPath,const string &dnnProtoPath, 
        const string &dnnModelPath
    ) {
        this->rPPGAlg = rPPGAlg;
        this->faceDetAlg = faceDetAlg;
        this->lastSamplingTime = 0;
        this->minFaceSize = Size(min(width, height) * REL_MIN_FACE_SIZE, min(width, height) * REL_MIN_FACE_SIZE);
        this->maxSignalSize = maxSignalSize;
        this->minSignalSize = minSignalSize;
        this->rescanFlag = false;
        this->rescanFrequency = rescanFrequency;
        this->samplingFrequency = samplingFrequency;
        this->timeBase = timeBase;

        // Load classifier
        switch (faceDetAlg) {
        case haar:
            haarClassifier.load(haarPath);
            break;
        case deep:
            dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);
            break;
        }

        return true;
    };

    float processFrame(Mat &frameRGB, Mat &frameGray, int time) {
        float bpm = 0.0;

        // Set time
        this->time = time;
        if (!faceValid) {
            lastScanTime = time;
            detectFace(frameRGB, frameGray);
        } else if ((time - lastScanTime) * timeBase >= 1/rescanFrequency) {
            lastScanTime = time;
            detectFace(frameRGB, frameGray);
            rescanFlag = true;
        } else {
            trackFace(frameGray);
        }

        if (faceValid) {
            // Update fps
            fps = getFps(t, timeBase);

            // Remove old values from raw signal buffer
            while (s.rows > (fps * maxSignalSize)) {
                push(s);
                push(t);
                push(re);
            }
            assert(s.rows == t.rows && s.rows == re.rows);

            // New values
            Scalar means = mean(frameRGB, mask);
            // Add new values to raw signal buffer
            double values[] = {means(0), means(1), means(2)};
            s.push_back(Mat(1, 3, CV_64F, values));
            t.push_back(time);

            // Save rescan flag
            re.push_back(rescanFlag);

            // Update fps
            fps = getFps(t, timeBase);

            // Update band spectrum limits
            low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
            high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;

            // If valid signal is large enough: estimate
            if (s.rows >= fps * minSignalSize) {
                // Filtering
                switch (rPPGAlg) {
                    case g:
                        extractSignal_g();
                        break;
                    case pca:
                        extractSignal_pca();
                        break;
                    case xminay:
                        extractSignal_xminay();
                        break;
                }
                // HR estimation
                bpm = estimateHeartrate();
            }
        }
        rescanFlag = false;
        frameGray.copyTo(lastFrameGray);

        return bpm;
    };

    void exit() {;};

    typedef vector<Point2f> Contour2f;

    /**
     * This is the implementation of the Smoothed Z-Score Algorithm.
     * This is direction translation of https://stackoverflow.com/a/22640362/1461896.
     *
     * @param input - input signal
     * @param lag - the lag of the moving window
     * @param threshold - the z-score at which the algorithm signals
     * @param influence - the influence (between 0 and 1) of new signals on the mean and standard deviation
     * @return a hashmap containing the filtered signal and corresponding mean and standard deviation.
     * @note https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/46998001#46998001
     */
    unordered_map<string, vector<ld>> z_score_thresholding(
        vector<ld> input, int lag, 
        ld threshold, ld influence
    ) {
        unordered_map<string, vector<ld>> output;
        
        uint n = (uint) input.size();
        vector<ld> signals(input.size());
        vector<ld> filtered_input(input.begin(), input.end());
        vector<ld> filtered_mean(input.size());
        vector<ld> filtered_stddev(input.size());
        
        VectorStats lag_subvector_stats(input.begin(), input.begin() + lag);
        filtered_mean[lag - 1] = lag_subvector_stats.mean();
        filtered_stddev[lag - 1] = lag_subvector_stats.standard_deviation();
        
        for (int i = lag; i < n; i++) {
            if (abs(input[i] - filtered_mean[i - 1]) > threshold * filtered_stddev[i - 1]) {
                signals[i] = (input[i] > filtered_mean[i - 1]) ? 1.0 : -1.0;
                filtered_input[i] = influence * input[i] + (1 - influence) * filtered_input[i - 1];
            } else {
                signals[i] = 0.0;
                filtered_input[i] = input[i];
            }
            VectorStats lag_subvector_stats(filtered_input.begin() + (i - lag), filtered_input.begin() + i);
            filtered_mean[i] = lag_subvector_stats.mean();
            filtered_stddev[i] = lag_subvector_stats.standard_deviation();
        }
        
        output["signals"] = signals;
        output["filtered_mean"] = filtered_mean;
        output["filtered_stddev"] = filtered_stddev;
        
        return output;
    };

private:

    void detectFace(Mat &frameRGB, Mat &frameGray) {
        vector<Rect> boxes = {};

        switch (faceDetAlg) {
        case haar:
            // Detect faces with Haar classifier
            haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CASCADE_SCALE_IMAGE, minFaceSize);
            break;
        case deep:
            // Detect faces with DNN
            Mat resize300;
            cv::resize(frameRGB, resize300, Size(300, 300));
            Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
            dnnClassifier.setInput(blob);
            Mat detection = dnnClassifier.forward();
            Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
            float confidenceThreshold = 0.5;

            for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidenceThreshold) {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
                boxes.push_back(object);
            }
            }
            break;
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
    };

    void setNearestBox(vector<Rect> boxes) {
        int index = 0;
        Point p = box.tl() - boxes.at(0).tl();
        int min = p.x * p.x + p.y * p.y;
        for (int i = 1; i < boxes.size(); i++) {
            p = box.tl() - boxes.at(i).tl();
            int d = p.x * p.x + p.y * p.y;
            if (d < min) {
                min = d;
                index = i;
            }
        }
        box = boxes.at(index);
    };

    void detectCorners(Mat &frameGray) {
        // Define tracking region
        Mat trackingRegion = Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
        Point points[1][4];
        
        points[0][0] = Point(box.tl().x + 0.22 * box.width, box.tl().y + 0.21 * box.height);
        points[0][1] = Point(box.tl().x + 0.78 * box.width, box.tl().y + 0.21 * box.height);
        points[0][2] = Point(box.tl().x + 0.70 * box.width, box.tl().y + 0.65 * box.height);
        points[0][3] = Point(box.tl().x + 0.30 * box.width, box.tl().y + 0.65 * box.height);
        
        const Point *pts[1] = {points[0]};
        int npts[] = {4};
        fillPoly(trackingRegion, pts, npts, 1, WHITE);

        // Apply corner detection
        goodFeaturesToTrack(
            frameGray, corners, MAX_CORNERS, 
            QUALITY_LEVEL, MIN_DISTANCE, trackingRegion, 
            3, false, 0.04
        );
    };

    void trackFace(Mat &frameGray) {
        // Make sure enough corners are available
        if (corners.size() < MIN_CORNERS) {
            detectCorners(frameGray);
        }

        Contour2f corners_1;
        Contour2f corners_0;
        vector<uchar> cornersFound_1;
        vector<uchar> cornersFound_0;
        Mat err;

        // Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
        calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

        // Backtrack once to make it more robust
        calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

        // Exclude no-good corners
        Contour2f corners_1v;
        Contour2f corners_0v;
        for (size_t j = 0; j < corners.size(); j++) {
            if (cornersFound_1[j] && cornersFound_0[j]
                && norm(corners[j]-corners_0[j]) < 2) {
                corners_0v.push_back(corners_0[j]);
                corners_1v.push_back(corners_1[j]);
            };
        }

        if (corners_1v.size() >= MIN_CORNERS) {
            // Save updated features
            corners = corners_1v;

            // Estimate affine transform
            Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);

            if (transform.total() > 0) {
                // Update box
                Contour2f boxCoords;
                boxCoords.push_back(box.tl());
                boxCoords.push_back(box.br());
                Contour2f transformedBoxCoords;

                cv::transform(boxCoords, transformedBoxCoords, transform);
                box = Rect(transformedBoxCoords[0], transformedBoxCoords[1]);

                // Update roi
                Contour2f roiCoords;
                roiCoords.push_back(roi.tl());
                roiCoords.push_back(roi.br());
                Contour2f transformedRoiCoords;
                cv::transform(roiCoords, transformedRoiCoords, transform);
                roi = Rect(transformedRoiCoords[0], transformedRoiCoords[1]);

                updateMask(frameGray);
            }

        } else {
            invalidateFace();
        }
    };

    void updateMask(Mat &frameGray) {
        mask = Mat::zeros(frameGray.size(), frameGray.type());
        rectangle(mask, this->roi, WHITE, FILLED);
    };

    void updateROI() {
        this->roi = Rect(
            Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
            Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height)
        );    
    };

    void extractSignal_g() {
        // Denoise
        Mat s_den = Mat(s.rows, 1, CV_64F);
        
        denoise(s.col(1), re, s_den);
        
        // Normalise
        normalization(s_den, s_den);

        // Detrend
        Mat s_det = Mat(s_den.rows, s_den.cols, CV_64F);
        detrend(s_den, s_det, fps);

        // Moving average
        Mat s_mav = Mat(s_det.rows, s_det.cols, CV_64F);
        movingAverage(s_det, s_mav, 3, fmax(floor(fps/6), 2));

        s_mav.copyTo(s_f);
    };

    void extractSignal_pca() {
        // Denoise signals
        Mat s_den = Mat(s.rows, s.cols, CV_64F);
        denoise(s, re, s_den);

        // Normalize signals
        normalization(s_den, s_den);

        // Detrend
        Mat s_det = Mat(s.rows, s.cols, CV_64F);
        detrend(s_den, s_det, fps);

        // PCA to reduce dimensionality
        Mat s_pca = Mat(s.rows, 1, CV_32F);
        Mat pc = Mat(s.rows, s.cols, CV_32F);
        pcaComponent(s_det, s_pca, pc, low, high);

        // Moving average
        Mat s_mav = Mat(s.rows, 1, CV_32F);
        movingAverage(s_pca, s_mav, 3, fmax(floor(fps/6), 2));

        s_mav.copyTo(s_f);
    };

    void extractSignal_xminay() {
        // Denoise signals
        Mat s_den = Mat(s.rows, s.cols, CV_64F);
        denoise(s, re, s_den);

        // Normalize raw signals
        Mat s_n = Mat(s_den.rows, s_den.cols, CV_64F);
        normalization(s_den, s_n);

        // Calculate X_s signal
        Mat x_s = Mat(s.rows, s.cols, CV_64F);
        addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);

        // Calculate Y_s signal
        Mat y_s = Mat(s.rows, s.cols, CV_64F);
        addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
        addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);

        // Bandpass
        Mat x_f = Mat(s.rows, s.cols, CV_32F);
        bandpass(x_s, x_f, low, high);
        x_f.convertTo(x_f, CV_64F);
        Mat y_f = Mat(s.rows, s.cols, CV_32F);
        bandpass(y_s, y_f, low, high);
        y_f.convertTo(y_f, CV_64F);

        // Calculate alpha
        Scalar mean_x_f;
        Scalar stddev_x_f;
        meanStdDev(x_f, mean_x_f, stddev_x_f);
        Scalar mean_y_f;
        Scalar stddev_y_f;
        meanStdDev(y_f, mean_y_f, stddev_y_f);
        double alpha = stddev_x_f.val[0]/stddev_y_f.val[0];

        // Calculate signal
        Mat xminay = Mat(s.rows, 1, CV_64F);
        addWeighted(x_f, 1, y_f, -alpha, 0, xminay);

        // Moving average
        movingAverage(xminay, s_f, 3, fmax(floor(fps/6), 2));
    };

    float estimateHeartrate() {
        powerSpectrum = cv::Mat(s_f.size(), CV_32F);
        timeToFrequency(s_f, powerSpectrum, true);

        // band mask
        const int total = s_f.rows;
        Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
        bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);
        double bpm = 0.0;

        if (!powerSpectrum.empty()) {

            // grab index of max power spectrum
            double min, max;
            Point pmin, pmax;
            minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask);

            // calculate BPM
            bpm = pmax.y * fps / total * SEC_PER_MIN;
            bpms.push_back(bpm);
        }

        if ((time - lastSamplingTime) * timeBase >= 1/samplingFrequency) {
            lastSamplingTime = time;

            cv::sort(bpms, bpms, SORT_EVERY_COLUMN);

            // average calculated BPMs since last sampling time
            meanBpm = mean(bpms)(0);
            minBpm = bpms.at<double>(0, 0);
            maxBpm = bpms.at<double>(bpms.rows-1, 0);

            bpms.pop_back(bpms.rows);
        }

        return (float)(bpm);
    };

    void invalidateFace() {
        s = Mat1d();
        s_f = Mat1d();
        t = Mat1d();
        re = Mat1b();
        powerSpectrum = Mat1d();
        faceValid = false;
    };

    // The algorithm
    rPPGAlgorithm rPPGAlg;

    // The classifier
    faceDetAlgorithm faceDetAlg;
    CascadeClassifier haarClassifier;
    Net dnnClassifier;

    // Settings
    Size minFaceSize;
    int maxSignalSize;
    int minSignalSize;
    double rescanFrequency;
    double samplingFrequency;
    double timeBase;
    bool guiMode;

    // State variables
    int64_t time;
    double fps;
    int high;
    int64_t lastSamplingTime;
    int64_t lastScanTime;
    int low;
    
    // int64_t now;
    bool faceValid;
    bool rescanFlag;

    // Tracking
    Mat lastFrameGray;
    Contour2f corners;

    // Mask
    Rect box;
    Mat1b mask;
    Rect roi;

    // Raw signal
    Mat1d s;
    Mat1d t;
    Mat1b re;

    // Estimation
    Mat1d s_f;
    Mat1d bpms;
    Mat1d powerSpectrum;
    double bpm = 0.0;
    double meanBpm;
    double minBpm;
    double maxBpm;
};

#endif /* RPPG_hpp */
