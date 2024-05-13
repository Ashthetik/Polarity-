#ifndef Matrix_HPP
#define Matrix_HPP

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