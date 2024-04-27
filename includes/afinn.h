#ifndef AFINN_H
#define AFINN_H

#include <unordered_map>
#include <string>
#include <fstream>

class AFINN {
public:
    int getEmoji(std::string);
    int getText(std::string);
private:
    std::unordered_map<std::string, int> textPolarity;
    std::unordered_map<std::string, int> emojiPolarity;
};

#endif