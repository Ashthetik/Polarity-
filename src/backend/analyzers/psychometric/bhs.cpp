/**
 * Structure of the Behaviour data:
 * Emotions (int8_bit), general movement (string), playtime (float .3f), Stressed (bool), Sex (int8_bit),
 *
 * [
 *    [ // Emotions
 *		0
 *    ]
 * ]
 */

#include <cstdint>
#include <string>
#include <vector>
#include "Matrix.h"
#include "bhs.h"
#include <complex>
#include <bitset>

/**
 * @brief
 *
 * @param x
 * @param a
 * @param h
 * @param k
 * @return std::vector<int64_t>
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

/**
 * Clear up the data
 */
void clean_table_data(DataTable table)
{
}
