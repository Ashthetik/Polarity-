#ifndef COLLECTOR_HPP
#define COLLECTOR_HPP

#include "backends/converter/matrix.hpp"
#include "backends/decisiontree.hpp"

/**
 * Matrix for the Decision Tree 4xInf:
 * Action (int), occurence (float), combo (boolean), target (int)
 * [
 *      [ 1, 0.5, true, 1 ],
 *      [ 2, 0.3, false, 2 ],    
 *      [ 3, 0.2, true, 3 ],
 *      ...
 * ]
*/

class Collector {
    public:
    int target = 0;   

    Collector(void) {
        // Initialize the matrix
        
        this->currentMatrix = { 
            {
                0, // Action
                0.0, // Occurence
                false, // Combo
                0 // Target
            } 
        };
    };

    ~Collector(void) {
        // Clear the matrix
        this->currentMatrix.clear();
    }

    void appendToMatrix(Matrix mat) {
        // append are current matrix to the final matrix
        this->currentMatrix = Matrix::append(mat, currentMatrix);
    }

    Matrix getMatrix(void) {
        return this->currentMatrix;
    }

    Table convertToTable(void) {
        if (this->table.data.size() == 0) {
            this->table.data.push_back({
                "Action",
                "Occurence",
                "Combo",
                "Target"
            });
        }

        // Convert the matrix to a table
        for (int i = 0; i < 1; i++) {
            this->table.data.push_back({
                std::to_string(this->currentMatrix.get(i, 0)),
                std::to_string(this->currentMatrix.get(i, 1)),
                std::to_string(this->currentMatrix.get(i, 2)),
                std::to_string(this->currentMatrix.get(i, 3))
            });
        }

        // Clear the matrix
        this->currentMatrix.clear();

        return this->table;
    }

    void processMatrix(void) {
        // Convert the matrix to a table;
        Table table = convertToTable();

        // Process the table
        DecisionTree tree = DecisionTree(table);

        tree.run(table, 0);

        /// Predict the target
        // Find the row with the highest occurence
        int maxOccurence = 0;

        #pragma omp parallel for
        for (int i = 1; i < table.data.size(); i++) {
            if (std::stof(table.data[i][1]) > maxOccurence) {
                maxOccurence = std::stof(table.data[i][1]);
                target = std::stoi(table.data[i][3]);
            }
        }
    };

    private:
    // Action (int), occurence (float), combo (boolean), target (int)
    Matrix currentMatrix;
    Table table;
};


#endif