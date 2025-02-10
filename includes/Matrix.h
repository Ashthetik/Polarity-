#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

struct Mat3D {
    float X, Y, Z;

public:
    Mat3D() = default;
    Mat3D(const float X, const float Y, const float Z) : X(X), Y(Y), Z(Z) {}
};

struct Mat5D {
    float X, Y, Z, Yaw, Pitch;

public:
    Mat5D() = default;
    Mat5D(
        const float X, const float Y, const float Z,
        const float Yaw, const float Pitch
    ): X(X), Y(Y), Z(Z), Yaw(Yaw), Pitch(Pitch) {};
};

struct Matrix {
public:
    std::vector<std::vector<float>> _Matrix;

    Matrix() = default;

    void clear();
    void set(int i, int j, float value);
    
    float get(int i, int j) const;

    void print() const;

    static inline Matrix append(const Matrix& newMatrix, Matrix currentMatrix);
};

inline Matrix convertMat3D(Mat3D position);

inline Matrix convertMat5D(const Mat5D &position);

#endif