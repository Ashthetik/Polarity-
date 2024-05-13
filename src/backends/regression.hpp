#include <iostream>
#include <stdio.h>
#include <vector>
#include <numeric>

using namespace std;

class Regression {
    private:
        vector<float> x;
        vector<float> y;
        float m, c;
        
        float coeff;
        float constTerm;

        float sumx;
        float sumy;
        float sumxy;
        float sumx_sqr;
        float sumy_sqr;

    public:
        Regression() {
            coeff = 0;
            constTerm = 0;
            sumy = 0;
            sumy_sqr = 0;
            sumx = 0;
            sumx_sqr = 0;
            sumxy = 0;
        }

        void calcCoeff() {
            float N = x.size();
            float num = (N * sumxy - sumx * sumy);
            float denom = (N * sumx_sqr - sumx * sumx);
            coeff = num / denom;
        }

        void calcCT() {
            float N = x.size();
            float num = (sumy * sumx_sqr - sumx * sumxy);
            float denom = (N * sumx_sqr - sumx * sumx);
            coeff = num / denom;
        }

        float sizeOfData() {
            return x.size();
        }
        
        float coefficient() {
            if (constTerm == 0) {
                calcCoeff();
            }
            return coeff;
        }

        float constant() {
            if (constTerm ==0) {
                calcCT();
            }
            return constTerm;
        }

        vector<float> bestFit() {
            if (coeff == 0 && constTerm == 0) {
                calcCoeff();
                calcCT();
            }
            return vector<float> { coeff, constTerm };
        }

        void takeIn(float x[5], float y[5]) {
            for (int i = 0; i < 5; i++) {
                sumxy += x[i] * y[i];
                sumx += x[i];
                sumy += y[i];
                sumx_sqr += x[i]*x[i];
                sumy_sqr += y[i]*y[i];
                this->x.push_back(x[i]);
                this->y.push_back(y[i]);
            }
        }

        float predict(float X) {
            float sumX = std::accumulate(x.begin(), x.end(), 0.0);
            float sumY = std::accumulate(y.begin(), y.end(), 0.0);
            float sumXY = 0;
            float sumXX = 0;
            int n = x.size();

            for (int i = 0; i < n; i++) {
                sumXY += x[i] * y[i];
                sumXX += x[i] * x[i];
            }

            m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
            c = (sumY - m * sumX) / n;

            return m * X + c;
        }

        float errorSquare() {
            float ans = 0;
            for (int i = 0;
                i < x.size(); i++) {
                ans += ((predict(x[i]) - y[i])
                        * (predict(x[i]) - y[i]));
            }
            return ans;
        }

        float errorIn(float num) {
            for (int i = 0;
                    i < x.size(); i++) {
                if (num == x[i]) {
                    return (y[i] - predict(x[i]));
                }
            }
            return 0;
        }

        float rSquared() {
            float ssr = errorSquare();
            float sst = 0;
            float meanY = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

            for (int i = 0; i < y.size(); i++) {
                sst += (y[i] - meanY) * (y[i] - meanY);
            }

            return 1 - (ssr / sst);
        }
};
