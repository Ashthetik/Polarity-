#include "regression.h"
#include <cmath>

void Regression::calculate_coefficients()
{
    float N = _x.size(); // Number of Data Points
    float num = (_SumY * _SumX_Sq) - (_SumX * _SumXY); // Numerator
    float denom = (N * _SumX_Sq) - (_SumX * _SumX); // Denominator

    _coeff = num / denom; // Coefficient of Determination
};

void Regression::calculate_constterm()
{
    float N = _x.size(); // Number of Data Points
    _constTerm = (_SumY - (_coeff * _SumX)) / N; // Constant Term
}

float Regression::get_coefficient()
{
    if (_coeff == 0) {
        calculate_coefficients();
    }
    return _coeff;
}

float Regression::get_constterm()
{
    if (_constTerm == 0) {
        calculate_constterm();
    }

    return _constTerm;
}

/**
 * @brief Find the Best Fit
 * @return std::vector<float>
 */
std::vector<float> Regression::find_best_fit()
{
    if (_coeff == 0 && _constTerm == 0) {
        calculate_coefficients();
        calculate_constterm();
    }

    return std::vector<float> { _coeff, _constTerm };
}

/**
 * @brief Add Data Points to the Regression
 * @param xVal X-Values
 * @param yVal Y-Values
 * @return void
 */
void Regression::add_data_points(float x[5], float y[5])
{
    for (int i = 0; i < 5; i++) {
        // Calculate the sums of X and Y
        _SumXY = x[i] * y[i];
        _SumX += x[i];
        _SumY += y[i];
        _SumX_Sq += x[i] * x[i];
        _SumY_Sq += y[i] * y[i];

        // Add the plots to the data points
        _x.push_back(x[i]);
        _y.push_back(y[i]);
    }
}

float Regression::predict(float x)
{
    // Calculate the sums of the Plot Points
    float sumx = std::accumulate(_x.begin(), _x.end(), 0.0);
    float sumy = std::accumulate(_y.begin(), _y.end(), 0.0);
    // These are bases for the regression line
    float sumxy = 0;
    float sumxx = 0;

    // Number of X-Points
    int n = _x.size();

    // Calculate the traversals of XY and XX
    // For the intercepts
    for (int i = 0; i < n; i++) {
        sumxy += _x[i] * _y[i];
        sumxx += _x[i] * _x[i];
    }

    // Calculate the Slope
    float m = (n * sumxy - sumx * sumy) / (n * sumxx - sumx * sumx);
    // Calculate the Y-Intercept
    float c = (sumy - m * sumx) / n;

    // Calculate the Y-Value
    return m * x + c; // Y = MX + C
}

float Regression::error_square()
{
    float ans = 0;
    for (int i = 0; i < _x.size(); i++) {
        ans += (_y[i] - predict(_x[i])) * (_y[i] - predict(_x[i]));
    }

    return ans;
}
