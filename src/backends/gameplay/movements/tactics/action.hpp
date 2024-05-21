#ifndef ACTION_HPP
#define ACTION_HPP  

#include "backends/gameplay/movements/tactics/collector.hpp"
#include "backends/converter/matrix.hpp"

class ActionProcessor {
    public:
        ActionProcessor() {};
        ~ActionProcessor() {};

    void addDataToMatrix(int action, int occurence, bool combo, int target) {
        if (occurence == 0) {
            occurence = 1;
        }
        Matrix mat = {
            {
                action,
                occurence,
                combo,
                target
            }
        };

        this->collector = Collector();
        Matrix currentMatrix = this->collector.getMatrix();

        this->collector.appendToMatrix(mat);

        this->collector.processMatrix();
    };

    /**
     * @details Get the target from the matrix
     * @return int
     * @retval -1 No target found
     * @retval >= 1 Target found
    */
    int getTargeted() {
        if (this->collector.target == 0) {
            // No target found
            return -1;
        } else {
            return this->collector.target;
        }
    }

    private:
        Collector collector;

};

#endif