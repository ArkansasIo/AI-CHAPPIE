// ---------------------------------------------------------------------------
// audio/audio.hpp -- WAV file I/O and core audio signal processing.
//
// Pure C++17, no external libraries. Implements:
//   * WAV read/write (PCM 8/16/24/32-bit, mono/stereo, any sample rate)
//   * a mono AudioBuffer abstraction
//   * DSP: windowing, FFT (radix-2), resampling, filtering, normalization
//   * MFCC feature extraction (for speech recognition)
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/math.hpp"

namespace chappie
{

    // ---------------------------------------------------------------------------
    // A mono audio buffer of floating-point samples in [-1, 1].
    // ---------------------------------------------------------------------------
    class AudioBuffer
    {
    public:
        AudioBuffer() = default;
        AudioBuffer(std::size_t num_samples, int sample_rate = 16000);

        std::size_t size() const { return samples_.size(); }
        bool empty() const { return samples_.empty(); }
        int sample_rate() const { return sample_rate_; }
        void set_sample_rate(int sr) { sample_rate_ = sr; }

        double &operator[](std::size_t i) { return samples_[i]; }
        double operator[](std::size_t i) const { return samples_[i]; }

        const std::vector<double> &samples() const { return samples_; }
        std::vector<double> &samples() { return samples_; }

        double duration_seconds() const
        {
            return sample_rate_ > 0
                       ? static_cast<double>(samples_.size()) / sample_rate_
                       : 0.0;
        }

        void resize(std::size_t n) { samples_.resize(n, 0.0); }
        void clear() { samples_.clear(); }

        void push_back(double v) { samples_.push_back(v); }
        void append(const AudioBuffer &other);

        // Peak / RMS
        double peak() const;
        double rms() const;
        double mean() const;

        // In-place transforms.
        void normalize(double target_peak = 0.95);
        void apply_gain(double gain);
        void fade_in(std::size_t samples);
        void fade_out(std::size_t samples);
        void reverse();

        // Slicing
        AudioBuffer slice(std::size_t start, std::size_t count) const;

        // Concatenate with silence between.
        AudioBuffer &operator+=(const AudioBuffer &other);

        static AudioBuffer silence(double seconds, int sample_rate = 16000);
        static AudioBuffer tone(double freq_hz, double seconds, int sample_rate = 16000,
                                double amplitude = 0.5);
        static AudioBuffer noise(double seconds, int sample_rate = 16000,
                                 double amplitude = 0.1, std::uint32_t seed = 7);

    private:
        std::vector<double> samples_;
        int sample_rate_ = 16000;
    };

    // ---------------------------------------------------------------------------
    // WAV file I/O.
    // ---------------------------------------------------------------------------
    struct WavInfo
    {
        int sample_rate = 16000;
        int channels = 1;
        int bits_per_sample = 16;
        std::size_t num_frames = 0;
    };

    // Load a .wav file into a mono buffer (channels are averaged).
    bool wav_load(const std::string &path, AudioBuffer &out, WavInfo *info,
                  std::string &error);

    // Save a buffer as 16-bit PCM mono .wav.
    bool wav_save(const std::string &path, const AudioBuffer &buffer,
                  std::string &error);

    // Save with explicit bit depth (8, 16, 24, 32).
    bool wav_save_ex(const std::string &path, const AudioBuffer &buffer,
                     int bits_per_sample, std::string &error);

    // ---------------------------------------------------------------------------
    // DSP utilities
    // ---------------------------------------------------------------------------
    // In-place radix-2 FFT. `re`/`im` must have power-of-two length.
    void fft(std::vector<double> &re, std::vector<double> &im, bool inverse = false);

    // Magnitude spectrum (returns size/2 + 1 bins).
    std::vector<double> magnitude_spectrum(const AudioBuffer &buffer,
                                           std::size_t fft_size = 512);

    // Mel filterbank + MFCC feature extraction.
    // Returns `num_frames` rows of `num_ceps` coefficients each.
    std::vector<std::vector<double>> mfcc(const AudioBuffer &buffer,
                                          int num_ceps = 13, int num_filters = 26,
                                          std::size_t frame_len = 400,
                                          std::size_t hop = 160);

    // Linear resampling to a target sample rate.
    AudioBuffer resample(const AudioBuffer &in, int target_rate);

    // Simple one-pole low-pass filter (in place).
    void low_pass(AudioBuffer &buffer, double cutoff_hz);

    // A moving-average smoothing pass.
    void smooth(AudioBuffer &buffer, std::size_t window);

    // Zero-crossing rate (a cheap voiced/unvoiced and VAD feature).
    double zero_crossing_rate(const AudioBuffer &buffer);

    // Short-time energy (per frame).
    std::vector<double> frame_energy(const AudioBuffer &buffer,
                                     std::size_t frame_len = 400,
                                     std::size_t hop = 160);

} // namespace chappie
