#include <includes/afinn.h>

#include <unordered_map>
#include <string>
#include <fstream>

int AFINN::getText(std::string text) {
    if (textPolarity.size() == 0) {
        std::ifstream file ("data/AFINN.txt");
        if (file.is_open()) {
            std::string line;
            while (getline(file, line)) {
                int pos = line.find("\t");
                std::string word = line.substr(0, pos);
                int value = std::stoi(line.substr(pos + 1));
                textPolarity[word] = value;
            }
        }
    }
};

int AFINN::getEmoji(std::string emoji) {

}