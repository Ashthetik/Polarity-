#include <vector>
#include "movement.h"

MovementProcessor::MovementProcessor() {};

/**
 * @brief Add the new matrix to the existing matrix
 * 
 * @param matrix 
 */
void MovementProcessor::add_to_matrix(Mat3D matrix) {
	// Convert a 3D Matrix to an unbound matrix
	Matrix new_matrix = convertMat3D(matrix);

	_matrix.append(_matrix, new_matrix);
}

Matrix MovementProcessor::get_matrix() {
	return _matrix;
};

Matrix MovementProcessor::process_matrix() {
	Regression reg;
	std::vector<float> x, y;

	// Flatten the matrix into two 1D arrays
	for (const auto& mat : _matrix.Matrix_) {
		for (int i = 0; i < mat.size(); i++) {
			if (i < 3) {
				x.emplace_back(mat[i]);
			} else {
				y.emplace_back(mat[i]);
			}
		}
	}

	reg.add_data_points(x.data(), y.data());

	std::vector<float> prediction = {};

	#pragma omp parallel for
	for (int i = 0; i < 5; i++) {
		prediction.emplace_back(reg.predict(x.data()[i]));
	}

	// Convert the prediction out into our workable Matrix
	Matrix prediction_matrix = convertMat3D(
		Mat3D(
			prediction[0], 
			prediction[1], 
			prediction[2]
		)
	);

	return prediction_matrix;
}
