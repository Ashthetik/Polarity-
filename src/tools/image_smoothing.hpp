#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "motion_blur.hpp"

using namespace cv;
using namespace std;

int DELAY_CAPTION =  1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

class ImageSmoothing {
    public:
    static void smooth(Mat src) {
        Mat clone = src.clone(), dst;
        MotionBlur motionBlur;

        dst = motionBlur.motionBlur(clone);
        dst = Mat::zeros(src.size(), src.type());

        if (!src.data) {
            cout << "Error loading src" << endl;
        }

        // Guassian blur: Save to dst
        for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 ) {
            GaussianBlur( src, dst, Size( i, i ), 0, 0 );
        }
    }
};