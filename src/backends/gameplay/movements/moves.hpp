#ifndef Moves_HPP
#define Moves_HPP
#include "backends/converter/matrix.hpp"
#include "backends/regression.hpp"
#include <vector>

class MovementProcessor {
    public:
        MovementProcessor();

        void appendToMatrix(Mat3D mat) {
            // Convert the matrix to something usable
            Matrix newMat = convertMat3D(mat);
            this->currentMatrix[1][5] = { newMat };

            // append are current matrix to the final matrix
            this->matrix.push_back(this->currentMatrix);
        }

        std::vector<Matrix[1][5]> getMatrix(void) {
            return this->matrix;
        }

        Matrix processMatrix(void) {
            Regression reg;
            std::vector<float> x, y;

            // Flatten the matrix into two 1D arrays
            for (const auto& mat : this->matrix) {
                for (int i = 0; i < 5; i++) {
                    x.push_back(mat[0][0].M[0][i]);
                    y.push_back(currentMatrix[0][0].M[0][i]);
                }
            }

            // Convert the vectors to arrays
            float* x_arr = &x[0];
            float* y_arr = &y[0];

            // Pass the arrays to takeIn
            reg.takeIn(x_arr, y_arr);
                      
            // predict expected values
            std::vector<float> pred;
            #pragma omp parallel for
            for (int i = 0; i < 5; i++) {
                pred.push_back(reg.predict(x_arr[i]));
            }

            // Convert the predicted values to a matrix
            Matrix predMat = convertMat3D(Mat3D(pred[0], pred[1], pred[2]));
            return predMat;
        }
    
    private:
        ///
        Matrix currentMatrix[1][5];

    public:
        std::vector<Matrix[1][5]> matrix;

};

#endif