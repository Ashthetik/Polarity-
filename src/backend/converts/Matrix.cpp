#include "Matrix.h"
#include <vector>
#include <iostream>

void Matrix::clear() {
    Matrix_.clear();
}

void Matrix::set(int i, int j, float value) {
    Matrix_[i][j] = value;
}

float Matrix::get(int i, int j) const {
    return Matrix_[i][j];
}

/**
 * @brief For pretty printing DEBUG statments 
 */
void Matrix::print() const {
    #pragma omp parallel for
    for (const auto & i : Matrix_) {
        for (const float j : i) {
            std::cout << j << " ";
        }
    }
}

Matrix Matrix::append(const Matrix &newMatrix, Matrix currentMatrix) {
    std::vector<Matrix> matrix;
    matrix.push_back(currentMatrix);
    matrix.push_back(newMatrix);

    currentMatrix = matrix[1];

    return currentMatrix;
}

Matrix convertMat3D(const Mat3D position) {
    Matrix matrix;

    matrix.Matrix_[0][0] = position.X;
    matrix.Matrix_[0][1] = position.Y;
    matrix.Matrix_[0][2] = position.Z;
    matrix.Matrix_[0][3] = 0.0;
    matrix.Matrix_[0][4] = 0.0;

    return matrix;
}

Matrix convertMat5D(const Mat5D position) {
    Matrix matrix;

    matrix.Matrix_[0][0] = position.X;
    matrix.Matrix_[0][1] = position.Y;
    matrix.Matrix_[0][2] = position.Z;
    matrix.Matrix_[0][3] = position.Yaw;
    matrix.Matrix_[0][4] = position.Pitch;

    return matrix;
}
