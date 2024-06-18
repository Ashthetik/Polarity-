#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
 
using namespace cv;
using namespace std;

class MotionBlur {
    public:
    MotionBlur() {};

    Mat motionBlur(Mat src) {
        int len = 125;
        double theta = 0;
        int snr = 700;

        if (src.empty()) {
            cout << "Error loading src" << endl;
        }

        Mat out;

        Rect roi = Rect(0, 0, src.cols & -2, src.rows & -2);

        Mat Hw, h;
        
        this->calculatePSF(h, roi.size(), len, theta);
        this->calculateWnrFilter(h, Hw, 1.0 / double(snr));

        src.convertTo(src, CV_32F);
        this->edgeTaper(src, src);

        this->filter2DFreq(src(roi), out, Hw);

        out.convertTo(out, CV_8U);
        normalize(out, out, 0, 255, NORM_MINMAX);

        return out;
    }

    void calculatePSF(Mat& out, Size filter, int len, double theta) {
        Mat h(filter, CV_32F, Scalar(0));
        Point point(filter.width / 2, filter.height / 2);
        ellipse(h, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
        Scalar summa = sum(h);
        out = h / summa[0];
    }

    void FFTShift(const Mat& input, Mat& output) {
        output = input.clone();
        int cx = output.cols / 2;
        int cy = output.rows / 2;

        Mat q0(output, Rect(0, 0, cx, cy));
        Mat q1(output, Rect(cx, 0, cx, cy));
        Mat q2(output, Rect(0, cy, cx, cy));
        Mat q3(output, Rect(cx, cy, cx, cy));

        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    void filter2DFreq(const Mat& input, Mat& output, const Mat& H) {
        Mat planes[2] = {Mat_<float>(input.clone()), Mat::zeros(input.size(), CV_32F)};
        Mat complexI;
        merge(planes, 2, complexI);
        dft(complexI, complexI, DFT_SCALE);

        Mat planesH[2] = {Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F)};
        Mat complexH;
        merge(planesH, 2, complexH);
        Mat complexIH;
        mulSpectrums(complexI, complexH, complexIH, 0);

        idft(complexIH, complexIH);
        split(complexIH, planes);
        output = planes[0];
    }

    void calculateWnrFilter(const Mat& input, Mat& output, double nsr) {
        Mat h;
        FFTShift(input, h);

        Mat planes[2] = {
            Mat_<float>(h.clone()), Mat::zeros(h.size(), CV_32F)
        };
        Mat complexI;

        merge(planes, 2, complexI);
        dft(complexI, complexI);
        split(complexI, planes);

        Mat denom;
        pow(abs(planes[0]), 2, denom);
        denom += nsr;

        divide(planes[0], denom, output);
    }

    void edgeTaper(const Mat& input, Mat& output, double gamma = 5.0, double beta = 0.2) {
        int nx = input.cols;
        int ny = input.rows;
        Mat w1(1, nx, CV_32F, Scalar(0));
        Mat w2(1, ny, CV_32F, Scalar(0));

        float* p1 = w1.ptr<float>(0);
        float* p2 = w2.ptr<float>(0);

        float dx = float(2.0 * CV_PI / nx);
        float x = float(-CV_PI);

        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            p1[i] = float(
                0.5 * (tanh((
                    x + gamma / 2
                ) / beta) - tanh((
                    x - gamma / 2
                ) / beta ))
            );
            x += dx;
        }

        float dy = float(2.0 * CV_PI / ny);
        float y = float(-CV_PI);

        #pragma omp parallel for
        for (int i = 0; i < ny; i++) {
            p2[i] = float(
                0.5 * (tanh((
                    y + gamma / 2
                ) / beta) - tanh((
                    y - gamma / 2
                ) / beta ))
            );
            y += dy;
        }

        Mat w = w2 * w1;
        multiply(input, w, output);
    }

    private:
};