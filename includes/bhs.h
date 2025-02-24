#ifndef BHS_H
#define BHS_H

#include <cstdint>
#include <string>
#include <vector>

enum playstyle {
	defensive = 1,
	aggressive = 2,
	neutral = 3,
	recreant = 4, // aka, cowardice - commonly retreats
	unknown = 0,
};

struct behaviour_data {
	int emotions = 0;
	int paystyle = playstyle::unknown; // General Movements/Actions
	float playtime = 0.0; // We'll store their total playtime as a 0.3f
	bool stressed = false; // We derive this from the emotions, voice analysis, BPM, and current situation
	int average_bpm = 0; // We'll encorporate the heart-rate as a backup verification metric
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

/**
 * @brief Detect and Calculate Stress Levels
 * 
 * Stress levels can be calculated in a few ways, these tend to be a trend in heart rates, 
 * change in actions (i.e. running away or quickened aggressive behaviour), and,
 * increase in vocal strain (Praat HNR, ADSV, CPP, CSID, and L/H ratio)
 * 
 * see https://www.sciencedirect.com/science/article/pii/S0892199723000887
 * 
 * @param bio_physical_data 
 * @return behaviour_data 
 */
behaviour_data calculate_stress_level(behaviour_data bio_physical_data);

/**
 * @brief Normalise and Highlight Heart Stress
 * 
 * @param heart_rates 
 * @return float 
 */
float heart_rate_trend(std::vector<float> prior_heart_rates, std::vector<float> curr_heart_rates);

/**
 * @brief Normalise and Highlight Best Choice of Actions
 * 
 * @param action_choices 
 * @return float 
 */
float action_trend(std::vector<int> action_choices, std::vector<int> curr_choices);

/**
 * @brief Normalise and Highlight Trends in Vocal Usage and Strain
 * 
 * @param praat_hnr 
 * @param adsv 
 * @param cpp 
 * @param csid 
 * @param lh_ratio 
 * @return std::vector<float> 
 */
std::vector<float> vocal_strain_trend(
	std::vector<float> praat_hnr, std::vector<float> adsv, 
	std::vector<float> cpp, std::vector<float> csid, 
	std::vector<float> lh_ratio
);

float find_snr(std::vector<float> data) {
	// µ = (I1 + I2 + ...) / length
	float mean = (std::accumulate(data.begin(), data.end(), 0)) / data.size();
	std::vector<float> set = {}; // a new data, created from the accumulative of (Ix - µ)^2

	#pragma omp parallel for
	for (int i = 0; i < data.size(); i++) {
		set.emplace_back(std::pow(data[i] - mean, 2));
	}
	int accumulative = std::accumulate(set.begin(), set.end(), 0);

	// σ = √((I1 – µ)^2 + ... ) / µ
	float std_dev = std::sqrt(accumulative) / mean;

	// SNR = µ / σ
	return mean / std_dev;
}

#endif