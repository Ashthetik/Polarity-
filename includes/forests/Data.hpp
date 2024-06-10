#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>

#include "globals.h"

namespace ranger {
    class Data  {
        public:
        Data():
            num_rows(0), num_rows_rounded(0), 
            num_cols(0), snp_data(0), num_cols_no_snp(0), 
            externalData(true), index_data(0), 
            max_num_unique_values(0), order_snps(false) {}


        Data(const Data& data) {};
        Data& operator=(const Data& data) = delete;

        virtual ~Data() = default;

        virtual double get_x(size_t r, size_t c) const = 0; 
        virtual void set_x(size_t r, size_t c, double value, bool& error) = 0;

        virtual double get_y(size_t r, size_t c) const = 0;
        virtual void set_y(size_t r, size_t c, double value, bool& error) = 0;

        void addSnpData(unsigned char* snp, size_t cols_snp) {
            num_cols = num_cols_no_snp + cols_snp;
            num_rows_rounded = roundToNextMultipe(num_rows, 4);
            this->snp_data = snp_data;
        };

        size_t getVariableId(const std::string& var_name) const {
            for (size_t i = 0; i < variable_names.size(); ++i) {
                if (variable_names[i] == var_name) {
                    return i;
                }
            }

            return variable_names.size();
        };

        virtual void reserveMemory(size_t cols) = 0;

        bool loadFromFile(std::string f, std::vector<std::string>& dep_var_names) {};
        bool loadFromFileWhitespace(
            std::ifstream& in, std::string header, 
            std::vector<std::string>& dep_var_names
        ) {}
        bool loadFromFileOther(
            std::ifstream& in, std::string header, 
            std::vector<std::string>& dep_var_names,
            char sep
        ) {}   

        void getAllValues(
            std::vector<double>& all_vals, std::vector<size_t>& ids, 
            size_t varId, size_t start, size_t end
        ) const {};

        void getMinMaxValues(
            std::vector<double>& min_vals, std::vector<double>& max_vals, 
            size_t start, size_t end
        ) const {};

        size_t getIndex(size_t r, size_t c) const {
            size_t col_perm = c;
            if (c >= num_cols) {
                c = getUnpermutedVarId(c);
                r = getPermutedSampleId(r);
            }

            if (c < num_cols_no_snp) {
                return index_data[c * num_rows + r];
            } else {
                return getSnp(r, c, col_perm);
            }
        };

        size_t getSnp(size_t r, size_t c, size_t col_perm) const {
            size_t snp = 0;
            size_t num_snps = num_cols - num_cols_no_snp;
            for (size_t i = 0; i < num_snps; ++i) {
                snp |= (snp_data[r * num_snps + i] << i);
            }

            return snp;
        };

        size_t getUniqueDataValue(size_t var, size_t ind) {
            if (var >= num_cols) {
                var = getUnpermutedVarId(var);
            }

            if (var < num_cols_no_snp) {
                return unique_data_values[var][ind];
            } return (ind);
        }

        size_t getNumUniqueDataValues(size_t var) {
            if (var >= num_cols) {
                var = getUnpermutedVarId(var);
            }

            if (var < num_cols_no_snp) {
                return unique_data_values[var].size();
            } else {
                return (3);
            }
        }

        void sort() {

        }

        void orderSnpLevels(bool corrected_imp) {

        }

        const std::vector<std::string>& getVariableNames() const {
            return variable_names;
        }
        size_t getNumCols() const {
            return num_cols;
        }
        size_t getNumRows() const {
            return num_rows;
        }

        size_t getMaxNumUniqueValues() const {
            if (snp_data == 0 || max_num_unique_values > 3) {
                return max_num_unique_values;
            } else {
                return 3;
            }
        };

        void setIsOrderedVariable(const std::vector<std::string>* unordered_var_names) {
            is_ordered_variable.resize(num_cols, true);
            if (unordered_var_names != nullptr) {
                for (size_t i = 0; i < unordered_var_names->size(); ++i) {
                    size_t c = getVariableId((*unordered_var_names)[i]);
                    if (c < num_cols) {
                        is_ordered_variable[c] = false;
                    }
                }
            }
        };

        void setIsOrderedVariable(std::vector<bool>& is_ordered_variable) {
            this->is_ordered_variable = is_ordered_variable;
        };

        bool isOrderedVariable(size_t c) const {
            if (c >= num_cols) {
                c = getUnpermutedVarId(c);
            }
            return is_ordered_variable[c];
        };

        void permuteSampleIds(std::mt19937_64 random_number) {
            permuted_sampleIds.resize(num_rows);
            std::iota(permuted_sampleIds.begin(), permuted_sampleIds.end(), 0);
            std::shuffle(permuted_sampleIds.begin(), permuted_sampleIds.end(), random_number);
        }

        size_t getUnpermutedVarId(size_t c) const {
            if (c >= num_cols) {
                return c - num_cols;
            } else {
                return c;
            }
        };

        size_t getPermutedSampleId(size_t r) const {
            return permuted_sampleIds[r];
        };

        const std::vector<std::vector<size_t>>& getSnpOrder() const {
            return snp_order;
        }

        void setSnpOrder(std::vector<std::vector<size_t>>& snp_order) {
            this->snp_order = snp_order;
            order_snps = true;
        }

        protected:
        std::vector<std::string> variable_names;
        size_t num_cols;
        size_t num_rows;
        size_t num_rows_rounded;

        unsigned char* snp_data;
        size_t num_cols_no_snp;

        bool externalData;

        std::vector<size_t> index_data;
        std::vector<std::vector<double>> unique_data_values;
        size_t max_num_unique_values;

        std::vector<bool> is_ordered_variable;

        std::vector<size_t> permuted_sampleIds;

        std::vector<std::vector<size_t>> snp_order;
        bool order_snps;
    };
};

#endif 