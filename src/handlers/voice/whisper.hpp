/**
 * TODO: Work on handling Whisper.cpp through this
 * too allow for Voice Recognition and Speech to Text
 * for AFINN-111 to operate as a Whisper Agent
 */
#ifndef Whisper_HPP
#define Whisper_HPP

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include "includes/Whisper/whisper.h"
#include "common-sdl.h"
#include <common.h>

using namespace std;

class Whisper {
    public:
    Whisper() {};
    ~Whisper() {};

    struct whisper_params {
        int32_t num_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
        int32_t steps_ms = 3000; // Steps in milliseconds (3 second default)
        int32_t length_ms = 10000; // Length in milliseconds (10 second default)
        int32_t keep_ms = 200; // Keep in milliseconds (0.2 second default)
        int32_t capture_id = -1; // Capture ID
        int32_t max_tokens = 32; // Maximum tokens
        int32_t audio_ctx = 0; // Audio context

        float vad_threshold = 0.6f; // VAD threshold
        float freq_threshold = 100.0f; // Frequency threshold

        bool translate = true; // Translate for compatibility to English
        bool no_fallback = false; // No fallback
        bool print_special = true; // Print special characters
        bool no_context = false; // No context
        bool no_timestamps = false; // Allow timestamps
        bool tinydiarize = false;
        bool save_audio = false; // Don't save audio, we don't need it
        
        // Now macro check for GPU
        #ifdef WHISPER_GPU
        bool use_gpu = true; // Use GPU
        bool flash_attn = true; // Flash attention
        #else
        bool use_gpu = false; // Don't use GPU
        bool flash_attn = false; // Don't flash attention
        #endif

        std::string lang = "en"; // Language
        std::string model = "ggml-medium.en.bin"; // Model
        std::string fname_out; // Output file name
    };

    void printd(const whisper_params& params) {
        fprintf(stderr, "num_threads: %d\n", params.num_threads);
        /// ...
    }

    std::vector<char> runWhisper() {
        whisper_params params;

        std::vector<char> output;

        params.keep_ms = std::min(params.keep_ms, params.steps_ms);
        params.length_ms = std::max(params.length_ms, params.steps_ms);

        const int n_sample_steps = (1e-3 * params.steps_ms) * WHISPER_SAMPLE_RATE;
        const int n_sample_length = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
        const int n_sample_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;
        const int n_samples_30s = (1e-3 * 30000) * WHISPER_SAMPLE_RATE;

        const bool use_vad = n_sample_steps <= 0; // Sliding Window for VAD
        const int n_new_line = !use_vad ? std::min(1, params.length_ms / params.steps_ms - 1) : 1;

        params.no_timestamps = !use_vad;
        params.no_context |= use_vad;
        params.max_tokens = 0;

        /**
         * @brief Initialize the Whisper Agent
         */
        audio_async audio(params.length_ms);

        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "Failed to initialize audio capture!\n");
        }

        audio.resume();

        struct whisper_context_params cparams = whisper_context_default_params();

        cparams.use_gpu = params.use_gpu;
        cparams.flash_attn = params.flash_attn;

        struct whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

        std::vector<float> pcmf32(n_samples_30s, 0.0f);
        std::vector<float> pcmf32_old;
        std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

        std::vector<whisper_token> prompt_tokens;

        int n_iter = 0;
        bool is_running = true;

        std::ofstream fout;

        printf("Running Whisper Agent...\n");
        fflush(stdout);

        auto t_last = std::chrono::high_resolution_clock::now();
        const auto t_start = t_last;


        while (is_running) {
            is_running = sdl_poll_events();

            if (!is_running) {
                break;
            }

            if (!use_vad) {
                while (true) {
                    audio.get(params.steps_ms, pcmf32_new);

                    if ((int)pcmf32_new.size() > (2 * n_sample_steps)) {
                        audio.clear();
                        continue;
                    }

                    if ((int)pcmf32_new.size() >= n_sample_steps) {
                        audio.clear();
                        break;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                const int n_samples_new = pcmf32_new.size();

                const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, 
                    n_sample_keep + n_sample_length - n_samples_new
                ));

                pcmf32.resize(n_samples_new + n_samples_take);

                for (int i = 0; i < n_samples_take; i++) {
                    pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + 1];
                }

                memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new * sizeof(float));

                pcmf32_old = pcmf32_new;
            } else {
                const auto t_now = std::chrono::high_resolution_clock::now();
                const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();  

                if (t_diff < 2000) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                audio.get(2000, pcmf32_new);

                if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_threshold, params.freq_threshold, false)) {
                    audio.get(params.length_ms, pcmf32);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    continue;
                }

                t_last = t_now;
            }

            // Run the inference
            {
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

                wparams.print_progress = false;
                wparams.print_special = params.print_special;
                wparams.print_realtime = false;
                wparams.print_timestamps = !params.no_timestamps;
                wparams.translate = params.translate;
                wparams.single_segment = !use_vad;
                wparams.max_tokens = params.max_tokens;
                wparams.language = params.lang.c_str();
                wparams.n_threads = params.num_threads;
                wparams.audio_ctx = params.audio_ctx;
                wparams.tdrz_enable = params.tinydiarize; // Tinydiarize
                wparams.temperature_inc = params.no_fallback ? 0.0f : 0.1f;
                wparams.prompt_tokens = params.no_context ? nullptr : prompt_tokens.data();
                wparams.prompt_n_tokens = params.no_context ? 0 : prompt_tokens.size();

                if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                    fprintf(stderr, "Failed to run Whisper Agent!\n");
                    break;
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; i++) {
                    const char* text = whisper_full_get_segment_text(ctx, i);

                    if (text) {
                        output.insert(output.end(), text, text + strlen(text));
                    } else {
                        fprintf(stderr, "Failed to get segment text!\n");
                    }
                }
            }

            ++n_iter;

            fflush(stdout);
        }

        audio.pause();

        whisper_free(ctx);;

        return output;
    }

};

#endif