#ifndef Matrix_HPP
#define Matrix_HPP

#include <iostream>
#include <vector>

struct Mat3D {
    float X;
    float Y;
    float Z;

    public:
        Mat3D() = default;
        Mat3D(float x, float y, float z) 
            : X(x), Y(y), Z(z) {};
};

struct Mat5D {
    float X;
    float Y;
    float Z;
    float Yaw;
    float Pitch;

    public:
        Mat5D() = default;
        Mat5D(
            float x, float y, float z, 
            float yaw, float pitch
        ) 
            : X(x), Y(y), Z(z), Yaw(y), Pitch(pitch) {};
};

struct Matrix {
    float M[1][5];

    public:
        Matrix() = default;

        void clear() {
            for (int i = 0; i < 1; i++) {
                for (int j = 0; j < 5; j++) {
                    M[i][j] = 0;
                }
            }
        }

        void set(int i, int j, float value) {
            M[i][j] = value;
        }

        float get(int i, int j) {
            return M[i][j];
        }

        void print() {
            for (int i = 0; i < 1; i++) {
                for (int j = 0; j < 5; j++) {
                    std::cout << M[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }

        static inline Matrix append(
            Matrix newMat, Matrix current
        ) {
            // Append the new Matrix, to our current matrix
            std::vector<Matrix> matrix;
            matrix.push_back(current);
            matrix.push_back(newMat);
            
            // Update the current matrix
            current = matrix[1];

            return current;
        }
};


Matrix convertMat3D(Mat3D position) {
    Matrix converted = { { 
        position.X, position.Y, position.Y, 
        0.0, 0.0 // Pre-Define the Yaw/Pitch
    } };

    return converted;
};

Matrix convertMat5D(Mat5D position) {
    Matrix converted = { {
        position.X, position.Y, position.Z,
        position.Yaw, position.Pitch
    } };

    return converted;
};

#endif