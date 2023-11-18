#ifndef Heartbeat_hpp
#define Heartbeat_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

#include "RPPG.hpp"

#define DEFAULT_RPPG_ALGORITHM "g"
#define DEFAULT_FACEDET_ALGORITHM "haar"
#define DEFAULT_RESCAN_FREQUENCY 1
#define DEFAULT_SAMPLING_FREQUENCY 1
#define DEFAULT_MIN_SIGNAL_SIZE 5
#define DEFAULT_MAX_SIGNAL_SIZE 5
#define DEFAULT_DOWNSAMPLE 1 // x means only every xth frame is used

#define HAAR_CLASSIFIER_PATH "haarcascade_frontalface_alt.xml"
#define DNN_PROTO_PATH "opencv/deploy.prototxt"
#define DNN_MODEL_PATH "opencv/res10_300x300_ssd_iter_140000.caffemodel"

using namespace std;
using namespace cv;

class Heartbeat {
    
public:
    Heartbeat(int argc_, char* argv_[])
    {
        std::move(argv_, argv_ + argc, argv.begin());

        auto it1 = argv.begin();
        auto it2 = it1 + 1;

        while (it1 != argv.end() && it2 != argv.end()) {
            if ((*it1)[0] == '-') {
                switch_map[*it1] = *(it2);
            }
            ++it1;
            ++it2;
        }
    }

    float runScan(VideoCapture capture) {
        rPPGAlgorithm rPPGA;
        faceDetAlgorithm faceDetA;
        double rescanFreq;
        double samplingFreq;
        int maxSignSize;
        int minSignSize;

        rPPGA = to_rppgAlgorithm(DEFAULT_RPPG_ALGORITHM);
        faceDetA = to_faceDetAlgorithm(DEFAULT_FACEDET_ALGORITHM);
        rescanFreq = DEFAULT_RESCAN_FREQUENCY;
        samplingFreq = DEFAULT_SAMPLING_FREQUENCY;
        maxSignSize = DEFAULT_MAX_SIGNAL_SIZE;
        minSignSize = DEFAULT_MIN_SIGNAL_SIZE;

        if (minSignSize > maxSignSize) {
            cout << "Error: min signal size is greater than max signal size." << endl;
            return -1;
        }

        // Downsample
        int downsample;
        downsample = DEFAULT_DOWNSAMPLE;
        
        std::ifstream test1(HAAR_CLASSIFIER_PATH);
        if (!test1) {
            std::cout << "Face classifier xml not found!" << std::endl;
            exit(0);
        }

        std::ifstream test2(DNN_PROTO_PATH);
        if (!test2) {
            std::cout << "DNN proto file not found!" << std::endl;
            exit(0);
        }

        std::ifstream test3(DNN_MODEL_PATH);
        if (!test3) {
            std::cout << "DNN model file not found!" << std::endl;
            exit(0);
        }

        if (!capture.isOpened()) {
            std::cout << "[WARN] {VideoCapture} - Camera is not already open. Trying to open now\n" << std::endl;
            try {
                capture.open(0);
            } catch (Exception& e) {
                std::cout << "[ERROR] {VideoCapture} - Camera couldn't be opened. Please make sure the camera is properly connected/available.\n" << std::endl;
                std::cout << e.what() << std::endl;
                std::exit(-1);
            }
        }

        string title = "Webcam";

        const int WIDTH = capture.get(CAP_PROP_FRAME_WIDTH);
        const int HEIGHT = capture.get(CAP_PROP_FRAME_HEIGHT);
        const int FPS = capture.get(CAP_PROP_FPS);  
        const double TIME_BASE = 1.0 / FPS;

        ostringstream window_title;
        window_title << title << " - " << WIDTH << "x" << HEIGHT << " - " << FPS << "FPS";
    
        RPPG rppg = RPPG();
        rppg.load(
            rPPGA, faceDetA, // Algs
            WIDTH, HEIGHT, FPS, TIME_BASE, // Window
            samplingFreq, rescanFreq, // Scans
            minSignSize, maxSignSize, // Sizes
            HAAR_CLASSIFIER_PATH, DNN_PROTO_PATH, DNN_MODEL_PATH, // Models
            false // GUI
        );

        int i = 0;
        Mat frameRGB, frameGray;

        float bpm = 0.0;

        while (true) {
            capture.read(frameRGB);

            if (frameRGB.empty()) {
                break;
            }

            cvtColor(frameRGB, frameGray, COLOR_BGR2GRAY);
            equalizeHist(frameGray, frameGray);

            int time = (getTickCount() * 1000.0) / getTickFrequency();

            if (i % downsample) {
                bpm = rppg.processFrame(frameRGB, frameGray, time);
            }
            else {
                cout << "Skipping frame to downsample!" << endl;
            }

            if (waitKey(30) >= 0) {
                break;
            }

            i++;
        }

        return bpm;
    };

private:
    
    int argc;
    vector<string> argv;

    bool switches_on;
    map<string, string> switch_map;
    
    bool to_bool(string s) {
        bool res;
        
        transform(s.begin(), s.end(), s.begin(), ::tolower);
        istringstream is(s);
        is >> boolalpha >> res;

        return res;
    }


    string get_arg(int i) {
        if (i >= 0 && i < argc)
            return argv[i];
        return "";
    };

    string get_arg(string s) {
        if (!switches_on) return "";
        if (switch_map.find(s) != switch_map.end())
            return switch_map[s];
        return "";
    };
    
    rPPGAlgorithm to_rppgAlgorithm(string s) {
        rPPGAlgorithm res;
        // Switch-Case for rPPGAlgorithm
        switch (s[0]) {
            case 'g':
                res = g;
                break;
            case 'pca': 
                res = pca;
                break;
            case 'xminay':
                res = xminay;
                break;
            default:
                res = g;
                break;
        }

        return res;
    }

    faceDetAlgorithm to_faceDetAlgorithm(string s) {
        faceDetAlgorithm res;
        // Switch-Case for faceDetAlgorithm
        switch (s[0]) {
            case 'haar':
                res = haar;
                break;
            case 'deep':
                res = deep;
                break;
            default:
                res = haar;
                break;
        }

        return res;
    }
};

#endif /* Heartbeat_hpp */