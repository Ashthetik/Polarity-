#ifndef MOVEMENT_H
#define MOVEMENT_H

#include "regression.h"
#include "Matrix.h"
#include <opencv4/opencv2/opencv.hpp>

class MovementProcessor {
	public:
		MovementProcessor();
		
		void add_to_matrix(Mat3D matrix);

		Matrix get_matrix();

		Matrix process_matrix();

	private:
		Matrix _matrix;
};

#endif