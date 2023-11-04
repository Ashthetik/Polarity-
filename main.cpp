#pragma once

#include "includes/afinn/afinn.hpp"
#include "includes/heartbeat/heartbeat.hpp"
#include "includes/emotions/NED.hpp"

#include <iostream>
#include <fstream>
#include <string>


using namespace std;

int main(int argc, char* argv[]) {
    Polarity polarity;
    Heartbeat heartbeat = Heartbeat(argc, argv);
    NED ned;

    std::cout << "Polarity: " << polarity.getText("Hello, World! Today I'm happy!") << std::endl;
    std::cout << "Heartbeat: " << heartbeat.runScan(argc, argv) << std::endl;
    std::cout << "NED: " << ned.detectEmotion() << std::endl;

    return 0;
}

