#ifndef GLOBALS_H
#define GLOBAL_H

namespace ranger {
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    typedef unsigned int uint;

    // Tree Types; Probability is not selected by ID
    enum TreeType {
        TREE_CLASSIFICATION = 1,
        TREE_REGRESSION = 3,
        TREE_SURVIVAL = 5,
        TREE_PROBABILITY = 9
    };

    // Memory Modes
    enum MemoryMode {
        MEM_DOUBLE = 0,
        MEM_FLOAT = 1,
        MEM_CHAR = 2
    };

    const uint MAX_MEM_MODE = 2;

    static const int mask[4] = { 192, 48, 12, 3 };
    static const int shift[4] = { 6, 4, 2, 0 };

    enum ImportanceMode {
        IMP_NONE = 0,
        IMP_GINI = 1,
        IMP_PERM_BREIMAN = 2,
        IMP_PERM_LIAW = 4,
        IMP_PERM_RAW = 3,
        IMP_GINI_CORRECTED = 5,
        IMP_PERM_CASEWISE = 6
    };

    const uint MAX_IMP_MODE = 6;

    enum SplitRule {
        LOGRANK = 1,
        AUC = 2,
        AUC_IGNORE_TIES = 3,
        MAXSTAT = 4,
        EXTRATREES = 5,
        BETA = 6,
        HELLINGER = 7   
    };

    enum PredictionType {
        RESPONSE = 1,
        TERMINALNODES = 2,
    };


    // Default values
    const uint DEFAULT_NUM_TREE = 500;
    const uint DEFAULT_NUM_THREADS = 0;
    const ImportanceMode DEFAULT_IMPORTANCE_MODE = IMP_NONE;

    const uint DEFAULT_MIN_NODE_SIZE_CLASSIFICATION = 1;
    const uint DEFAULT_MIN_NODE_SIZE_REGRESSION = 5;
    const uint DEFAULT_MIN_NODE_SIZE_SURVIVAL = 3;
    const uint DEFAULT_MIN_NODE_SIZE_PROBABILITY = 10;
    const uint DEFAULT_MIN_BUCKET = 1;
    const uint DEFAULT_MIN_BUCKET_SURVIVAL = 3;

    const SplitRule DEFAULT_SPLITRULE = LOGRANK;
    const double DEFAULT_ALPHA = 0.5;
    const double DEFAULT_MINPROP = 0.1;

    const uint DEFAULT_MAXDEPTH = 0;
    const PredictionType DEFAULT_PREDICTIONTYPE = RESPONSE;
    const uint DEFAULT_NUM_RANDOM_SPLITS = 1;

    const double DEFAULT_SAMPLE_FRACTION_REPLACE = 1;
    const double DEFAULT_SAMPLE_FRACTION_NOREPLACE = 0.632;

    // Interval to print progress in seconds
    const double STATUS_INTERVAL = 30.0;

    // Threshold for q value split method switch
    const double Q_THRESHOLD = 0.02;
};

#endif