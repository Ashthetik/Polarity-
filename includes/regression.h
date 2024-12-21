#ifndef REGRESSION_H
#define REGRESSION_H

#include <numeric>
#include <vector>

class Regression {
	public:
		Regression(): _m(0), _c(0) {};

		void calculate_coefficients();

		void calculate_constterm();

		float get_coefficient();

		float get_constterm();

		std::vector<float> find_best_fit();

		void add_data_points(float x[5], float y[5]);

		float predict(float x);

		float error_square();

	private:
		std::vector<float> _x, _y;
		float _m, _c = 0;

		float _coeff = 0;
		float _constTerm = 0;

		float _SumX, _SumY = 0;
		float _SumXY = 0;
		float _SumX_Sq, _SumY_Sq = 0;
};

#endif