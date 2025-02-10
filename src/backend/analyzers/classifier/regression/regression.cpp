#include "regression.h"
#include <cmath>

/**
 * @brief Calculates the coefficient of determination for the regression model.
 *
 * This function computes the coefficient of determination (also known as R-squared)
 * using the formula: (ΣY * ΣX^2 - ΣX * ΣXY) / (N * ΣX^2 - (ΣX)^2),
 * where N is the number of data points.
 *
 * @pre The member variables _SumY, _SumX_Sq, _SumX, and _SumXY must be initialized.
 * @pre The vector _x must contain the x-coordinates of the data points.
 *
 * @post The calculated coefficient is stored in the _coeff member variable.
 */
void Regression::calculate_coefficients()
{
    float N = _x.size(); 
    float num = (_SumY * _SumX_Sq) - (_SumX * _SumXY);
    float denom = (N * _SumX_Sq) - (_SumX * _SumX);

    _coeff = num / denom; // Coefficient of Determination
};

/**
 * @brief Calculates the constant term (y-intercept) of the regression line.
 *
 * This function computes the constant term (also known as the y-intercept)
 * of the regression line using the formula: (ΣY - coefficient * ΣX) / N,
 * where N is the number of data points.
 *
 * @pre The member variables _SumY, _coeff, and _SumX must be initialized.
 * @pre The vector _x must contain the x-coordinates of the data points.
 *
 * @post The calculated constant term is stored in the _constTerm member variable.
 */
void Regression::calculate_constterm()
{
    float N = _x.size(); // Number of Data Points
    _constTerm = (_SumY - (_coeff * _SumX)) / N;
}

/**
 * @brief Retrieves the coefficient of the regression model.
 *
 * This function returns the coefficient (slope) of the regression line.
 * If the coefficient hasn't been calculated yet (i.e., it's zero),
 * it calls the calculate_coefficients() method to compute it first.
 *
 * @return float The coefficient (slope) of the regression line.
 */
float Regression::get_coefficient()
{
    if (_coeff == 0) {
        calculate_coefficients();
    }
    return _coeff;
}

/**
 * @brief Retrieves the constant term (y-intercept) of the regression model.
 *
 * This function returns the constant term (y-intercept) of the regression line.
 * If the constant term hasn't been calculated yet (i.e., it's zero),
 * it calls the calculate_constterm() method to compute it first.
 *
 * @return float The constant term (y-intercept) of the regression line.
 */
float Regression::get_constterm()
{
    if (_constTerm == 0) {
        calculate_constterm();
    }

    return _constTerm;
}

/**
 * @brief Finds the best fit line for the regression model.
 *
 * This function calculates the coefficient and constant term of the best fit line
 * if they haven't been calculated yet. It then returns these values as a vector.
 *
 * @return std::vector<float> A vector containing two elements:
 *         - The first element is the coefficient (slope) of the best fit line.
 *         - The second element is the constant term (y-intercept) of the best fit line.
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
 * @brief Adds data points to the regression model and calculates necessary sums.
 *
 * This function takes two arrays of 5 float values each, representing x and y coordinates
 * of data points. It adds these points to the regression model and calculates various sums
 * used in regression analysis.
 *
 * @param x An array of 5 float values representing the x-coordinates of the data points.
 * @param y An array of 5 float values representing the y-coordinates of the data points.
 *
 * @pre The arrays x and y must contain exactly 5 elements each.
 * @post The data points are added to the model, and the sums (_SumXY, _SumX, _SumY, _SumX_Sq, _SumY_Sq)
 *       are updated. The vectors _x and _y are also updated with the new data points.
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

/**
 * @brief Predicts the y-value for a given x-value using linear regression.
 *
 * This function calculates the predicted y-value for a given x-value using
 * the linear regression model based on the data points stored in the object.
 * It computes the slope and y-intercept of the regression line and then
 * applies the equation y = mx + c to make the prediction.
 *
 * @param x The x-value for which to predict the corresponding y-value.
 * @return The predicted y-value based on the linear regression model.
 */
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

    // Calculate the traversals of XY and XX for the intercepts
    for (int i = 0; i < n; i++) {
        sumxy += _x[i] * _y[i];
        sumxx += _x[i] * _x[i];
    }

    // Calculate the Slope, Y intercept, and return the Y value
    float m = (n * sumxy - sumx * sumy) / (n * sumxx - sumx * sumx);
    float c = (sumy - m * sumx) / n;

    return m * x + c; // Y = MX + C, also known as Linear Regression (y = mx + b)
}

/**
 * @brief Calculates the sum of squared errors for the regression model.
 *
 * This function computes the sum of squared differences between the actual y-values
 * and the predicted y-values for all data points in the regression model. It provides
 * a measure of the model's accuracy, with lower values indicating a better fit.
 *
 * @return float The sum of squared errors for the regression model.
 */
float Regression::error_square()
{
    float ans = 0;
    for (int i = 0; i < _x.size(); i++) {
        ans += (_y[i] - predict(_x[i])) * (_y[i] - predict(_x[i]));
    }

    return ans;
}
