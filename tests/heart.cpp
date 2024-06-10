#include <iostream>
#include "src/handlers/heartrate/rppg.hpp"

int main(int argc, char const *argv[])
{
    RPPG rppg = RPPG();
    rppg.load(g, deep, 640, 480, 1.0, 1, 30.0, 1.0, 100, 100, "path/to/dnnProto", "path/to/dnnModel");
    rppg.runDetection();    
    return 0;
}