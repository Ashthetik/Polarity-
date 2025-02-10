#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "Matrix.h"
#include "bhs.h"
#include <complex>
#include <bitset>

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
std::vector<int64_t> log_transform(std::vector<int64_t> x, int32_t a, int64_t h, int64_t k)
{
	std::vector<int64_t> replacer = {};

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
	std::vector<int64_t> class_data_points = {};
	std::vector<int64_t> common_occurrences = {};
	std::vector<BehaviourTable> behaviour_data = {};
};

struct CleanTableData
{
    std::vector<std::int64_t> voice = {};
    std::vector<std::int64_t> bpm = {};
    std::vector<std::int64_t> movement = {};
};

std::vector<int64_t> convert_mat(std::vector<Mat3D> matrix) {
    std::vector<int64_t> replacer = {};

    for (auto& value : matrix) {
        replacer.emplace_back(value.X);
        replacer.emplace_back(value.Y);
        replacer.emplace_back(value.Z);
    }

    return replacer;
	
}

/**
 * @brief Converts a vector of float values to a vector of int64_t values.
 *
 * This function takes a vector of float values as input and returns a new vector
 * of int64_t values. Each float value in the input vector is multiplied by 100,
 * and the result is cast to an int64_t value. The resulting int64_t values are
 * stored in the output vector.
 *
 * @param array The input vector of float values.
 *
 * @return A vector of int64_t values, where each value is the result of
 *         multiplying the corresponding float value in the input vector by 100,
 *         and casting the result to an int64_t value.
 */
std::vector<int64_t> convert_float(std::vector<float> array) {
    std::vector<int64_t> replacer = {};

    for (auto& value : array) {
        replacer.emplace_back(static_cast<int64_t>(value * 100));
    }

    return replacer;
}

/**
 * @brief Cleans and transforms the data in the provided DataTable.
 *
 * This function processes the behaviour data in the input DataTable, converting
 * movement occurrences, BPM fluctuations, and voice fluctuations to cleaned
 * int64_t vectors using logarithmic transformation.
 *
 * @param table The DataTable containing behaviour data to be cleaned.
 *
 * @return A CleanTableData object containing the cleaned and transformed data.
 */
CleanTableData clean_table_data(DataTable table)
{
    std::vector<int64_t> movement_clean = {};
    std::vector<int64_t> bpm_clean = {};
    std::vector<int64_t> voice_clean = {};

    CleanTableData cleaned_data;

    if (table.behaviour_data.size() != 0) {
        #pragma omp parallel for
        for (auto it : table.behaviour_data) {
            auto movement_vector = convert_mat(it.movement_occurrences);
            auto bpm_vector = convert_float(it.bpm_fluctuation);
            auto voice_vector = convert_float(it.voice_fluctation);

            movement_clean = log_transform(movement_vector, movement_vector.size(), movement_vector.max_size(), static_cast<int64_t>(__DBL_EPSILON__));
            bpm_clean = log_transform(bpm_vector, bpm_vector.size(), bpm_vector.max_size(), static_cast<int64_t>(__DBL_EPSILON__));
            voice_clean = log_transform(voice_vector, voice_vector.size(), voice_vector.max_size(), static_cast<int64_t>(__DBL_EPSILON__));
        }
    }

    cleaned_data.movement = movement_clean;
    cleaned_data.bpm = bpm_clean;
    cleaned_data.voice = voice_clean;

    return cleaned_data;
}


/** -------------------- Data Processing ----------------------- */

enum ModelChoices {
    MODEL_A = 0, // Learnable Evolution Model
    MODEL_B = 1, // Learnable Hieararchical Variation Model
    MODEL_C = 2, // Recurrent Neural Network
};

void data_preprocessing(CleanTableData cleaned_data, ModelChoices model) {
    switch (model) {
        case MODEL_A:
            // Apply data processing for MODEL_A
            break;
        case MODEL_B:
            // Apply data processing for MODEL_B
            break;
        case MODEL_C:
            // Apply data processing for MODEL_C
            break;
        default:
            std::cerr << "Invalid model choice. Please choose a valid model." << std::endl;
    }

    // TODO: Add more model-specific data processing steps as needed
    // Example: Normalize data, feature extraction, etc.
}

// TOOD: Infer the model choice via a local variable or from OOP interface
void model_training(CleanTableData cleaned_data, ModelChoices model) {
    switch (model) {
        case MODEL_A:
            // Train MODEL_A
            break;
        case MODEL_B:
            // Train MODEL_B
            break;
        case MODEL_C:
            // Train MODEL_C
            break;
        default:
            std::cerr << "Invalid model choice. Please choose a valid model." << std::endl;
    }

    // TODO: Add more model-specific training steps as needed
    // Example: Split data into training and validation sets, set up hyperparameters, train the model
}

void model_evaluation(CleanTableData cleaned_data, ModelChoices model) {
    switch (model) {
        case MODEL_A:
            // Evaluate MODEL_A
            break;
        case MODEL_B:
            // Evaluate MODEL_B
            break;
        case MODEL_C:
            // Evaluate MODEL_C
            break;
        default:
            std::cerr << "Invalid model choice. Please choose a valid model." << std::endl;
    }

    // TODO: Add more model-specific evaluation steps as needed
    // Example: Calculate metrics like accuracy, precision, recall, F1-score, etc.
}