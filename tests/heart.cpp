#include <iostream>


int main(const int argc, char const *argv[])
{
    if (argc > 0) {
        std::string dnnProto = argv[1];
        std::string dnnModel = argv[2];
    }

    auto rppg = RPPG();
    rppg.load(g, deep, 640, 480, 30.0, 1.0, 100, 100, "path/to/dnnProto", "path/to/dnnModel");
    rppg.runDetection();    
    
    return 0;
}