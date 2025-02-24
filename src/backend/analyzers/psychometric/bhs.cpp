#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "Matrix.h"
#include "bhs.h"
#include <complex>
#include <limits>
#include <bitset>
#include <numeric>

/**
 * @brief Logarithmic Transformation for redistributing data representations through additive means. 
 * 
 * This function essentially "normalises" the data, whereas multiplicative functions would otherwise result in loss of
 * representation for the model.
 * 
 * @param X The array of data values in 64-bit integer format
 * @param a The shape of the object
 * @param h The horizontal shift/translation along the X axis
 * @param k The vertical shift/translation along the Y axis
 * @return
 */
std::vector<int> log_transform(std::vector<int> x, int32_t a, int h, int k)
{
	std::vector<int> replacer = {};

#pragma omp parallel for
	for (int i = 0; i < x.size(); i++)
	{
		replacer.emplace_back(
			(a * logb(x[i] - h) + k));
	}

	return replacer;
}

struct BehaviourTable
{
	std::vector<Mat3D> movement_occurrences = {};
	std::vector<float> bpm_fluctuation = {};
	std::vector<float> voice_fluctation = {};
};

struct DataTable
{
	std::vector<std::string> class_types = {};
	std::vector<int> class_data_points = {};
	std::vector<int> common_occurrences = {};
	std::vector<BehaviourTable> behaviour_data = {};
};

struct CleanTableData
{
    std::vector<int> voice = {};
    std::vector<int> bpm = {};
    std::vector<int> movement = {};
};

std::vector<int> convert_mat(std::vector<Mat3D> matrix) {
    std::vector<int> replacer = {};

    for (auto& value : matrix) {
        replacer.emplace_back(value.X);
        replacer.emplace_back(value.Y);
        replacer.emplace_back(value.Z);
    }

    return replacer;
	
}

/**
 * @brief Converts a vector of float values to a vector of int values.
 *
 * This function takes a vector of float values as input and returns a new vector
 * of int values. Each float value in the input vector is multiplied by 100,
 * and the result is cast to an int value. The resulting int values are
 * stored in the output vector.
 *
 * @param array The input vector of float values.
 *
 * @return A vector of int values, where each value is the result of
 *         multiplying the corresponding float value in the input vector by 100,
 *         and casting the result to an int value.
 */
std::vector<int> convert_float(std::vector<float> array) {
    std::vector<int> replacer = {};

    for (auto& value : array) {
        replacer.emplace_back(static_cast<int>(value * 100));
    }

    return replacer;
}

/**
 * @brief Cleans and transforms the data in the provided DataTable.
 *
 * This function processes the behaviour data in the input DataTable, converting
 * movement occurrences, BPM fluctuations, and voice fluctuations to cleaned
 * int vectors using logarithmic transformation.
 *
 * @param table The DataTable containing behaviour data to be cleaned.
 *
 * @return A CleanTableData object containing the cleaned and transformed data.
 */
CleanTableData clean_table_data(DataTable table)
{
    std::vector<int> movement_clean = {};
    std::vector<int> bpm_clean = {};
    std::vector<int> voice_clean = {};

    CleanTableData cleaned_data;

    if (table.behaviour_data.size() != 0) {
        #pragma omp parallel for
        for (auto it : table.behaviour_data) {
            auto movement_vector = convert_mat(it.movement_occurrences);
            auto bpm_vector = convert_float(it.bpm_fluctuation);
            auto voice_vector = convert_float(it.voice_fluctation);

            movement_clean = log_transform(movement_vector, movement_vector.size(), movement_vector.max_size(), static_cast<int>(std::numeric_limits<double>::epsilon()));
            bpm_clean = log_transform(bpm_vector, bpm_vector.size(), bpm_vector.max_size(), static_cast<int>(std::numeric_limits<double>::epsilon()));
            voice_clean = log_transform(voice_vector, voice_vector.size(), voice_vector.max_size(), static_cast<int>(std::numeric_limits<double>::epsilon()));
        }
    }

    cleaned_data.movement = movement_clean;
    cleaned_data.bpm = bpm_clean;
    cleaned_data.voice = voice_clean;

    return cleaned_data;
}


// TODO: Follow the cited paper 
behaviour_data calculate_stress_level(behaviour_data bpd) {

}

float heart_rate_trend(std::vector<float> prior_heart_rates, std::vector<float> curr_heart_rates)
{
    if (prior_heart_rates.size() <= 0 || curr_heart_rates.size() <= 0) {
        std::cout << "[ERROR] Heart Rate Data is Nil." << std::endl;
        return 0.0f;
    }

    float sum_of_a = std::accumulate(prior_heart_rates.begin(), prior_heart_rates.end(), 0);
    float sum_of_b = std::accumulate(curr_heart_rates.begin(), curr_heart_rates.end(), 0);

    return ((sum_of_b - sum_of_a) / sum_of_a) * 100;
}

float action_trend(std::vector<int> prior_choices, std::vector<int> curr_choices)
{
    if (prior_choices.size() <= 0 || curr_choices.size() <= 0) {
        std::cout << "[ERROR] Actions Trend Data is Nil." << std::endl;
        return 0.0f;
    }

    float sum_a = std::accumulate(prior_choices.begin(), prior_choices.end(), 0);
    float sum_b = std::accumulate(curr_choices.begin(), curr_choices.end(), 0);

    return ((sum_a - sum_b) / sum_a) * 100;
}

std::vector<float> vocal_strain_trend(
    std::vector<float> praat_hnr, std::vector<float> adsv, 
    std::vector<float> cpp, std::vector<float> csid, 
    std::vector<float> lh_ratio
) {
    return std::vector<float>();
}

std::vector<float> calculate_praat_hnr() {
    
}
