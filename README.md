# Polarity++

> [!IMPORTANT]
> This project is in **ALPHA** stages and __**IS NOT**__ expected to work and may cause potential damage in extreme cases.
> Proceed at your own risk.

## Description:
This project is a combination of facial emotion recognition (FER), sentiment analysis using the AFINN lexicon, and heart rate monitoring. The system will be integrated into NLP/LLM engine games and other applications to detect the player's emotional state and heart rate during gameplay. By incorporating these technologies, the system aims to create more reactive scenarios and NPCs to enhance the overall gaming experience.

## Features:
- Facial Emotion Recognition (FER): Detects the player's emotions through facial expressions using computer vision algorithms.
- Sentiment Analysis (AFINN): Analyzes the player's textual inputs to determine their emotional state based on the AFINN lexicon.
- Heart Rate Monitoring: Tracks the player's heart rate during gameplay using a heart rate monitor device.
- Integration with NLP/LLM engine games: Allows developers to incorporate the system into their games to create more immersive and dynamic gaming experiences.

## Installation:
1. Clone the repository: `git clone https://github.com/Ashthetik/Polarity-.git`
2. run **BOTH** the installers in `installers/`
3. create a `build/` directory and run `cmake ..`
4. now `make -j $nproc` or `ninja` (note for windows do `cmake --build .  --target install`)

> [!IMPORTANT]
> Please note, this won't actually run yet, so please don't open an issue regarding it not running.

## Usage: (Planned)
1. [Coming Soon]

## Authors And Contributors:
- Ashley (me@ashleyxir.tech)
- CallMeZombie ([Zonbi-san](https://github.com/Zonbi-san))

## Credits:
- Proaust for the base of the RPPG: [Proaust/heartbeat](https://github.com/prouast/heartbeat)
- Daniel Ruskin for the fundamental layout of the RNN: [DanielRuskin1/simple-recurrent-neural-network](https://github.com/DanielRuskin1/simple-recurrent-neural-network)
- Georgi Gerganov for Whisper.cpp: [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp)

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
