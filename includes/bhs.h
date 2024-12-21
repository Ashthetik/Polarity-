#ifndef BHS_H
#define BHS_H

#include <cstdint>
#include <string>
#include <vector>

struct behaviour_data {
	int8_t emotions = 0;
	std::string playstyle = ""; // General Movements/Actions
	float playtime = 0.0; // We'll store their total playtime as a 0.3f
	bool stressed = false; // We derive this from the emotions, voice analysis, and current situation
	int32_t average_bpm = 0; // We'll encorporate the heart-rate as a backup verification metric
};

/**
 * @brief Logarithmic Transformation for redistributing data representations through additive means. 
 * Essentially "normalising" the data, whereas multiplicative would otherwise would result in loss of
 * representation
 * @param X The array of data values in 64-bit integer format
 * @param base The base of the logarithmic expression 
 * @param h The horizontal shift/translation along the X axis
 * @param k The vertical shift/translation along the Y axis
 * @return
 */
std::vector<int64_t> log_transform(std::vector<int64_t> x, int32_t base, int64_t h, int64_t k);

#endif