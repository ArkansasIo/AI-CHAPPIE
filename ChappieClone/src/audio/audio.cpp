// ---------------------------------------------------------------------------
// audio/audio.cpp -- WAV I/O and DSP implementation.
// ---------------------------------------------------------------------------
#include "audio/audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include "core/logger.hpp"

namespace chappie
{

    namespace
    {
        constexpr double kPi = 3.14159265358979323846;
    }

    // ===========================================================================
    // AudioBuffer
    // ===========================================================================
    AudioBuffer::AudioBuffer(std::size_t num_samples, int sample_rate)
        : samples_(num_samples, 0.0), sample_rate_(sample_rate) {}

    void AudioBuffer::append(const AudioBuffer &other)
    {
        samples_.insert(samples_.end(), other.samples_.begin(), other.samples_.end());
    }

    double AudioBuffer::peak() const
    {
        double p = 0.0;
        for (double v : samples_)
            p = std::max(p, std::fabs(v));
        return p;
    }

    double AudioBuffer::rms() const
    {
        if (samples_.empty())
            return 0.0;
        double s = 0.0;
        for (double v : samples_)
            s += v * v;
        return std::sqrt(s / static_cast<double>(samples_.size()));
    }

    double AudioBuffer::mean() const
    {
        if (samples_.empty())
            return 0.0;
        double s = 0.0;
        for (double v : samples_)
            s += v;
        return s / static_cast<double>(samples_.size());
    }

    void AudioBuffer::normalize(double target_peak)
    {
        const double p = peak();
        if (p <= 1e-12)
            return;
        apply_gain(target_peak / p);
    }

    void AudioBuffer::apply_gain(double gain)
    {
        for (double &v : samples_)
            v *= gain;
    }

    void AudioBuffer::fade_in(std::size_t n)
    {
        n = std::min(n, samples_.size());
        for (std::size_t i = 0; i < n; ++i)
            samples_[i] *= static_cast<double>(i) / static_cast<double>(n);
    }

    void AudioBuffer::fade_out(std::size_t n)
    {
        n = std::min(n, samples_.size());
        const std::size_t start = samples_.size() - n;
        for (std::size_t i = 0; i < n; ++i)
            samples_[start + i] *= 1.0 - static_cast<double>(i) / static_cast<double>(n);
    }

    void AudioBuffer::reverse()
    {
        std::reverse(samples_.begin(), samples_.end());
    }

    AudioBuffer AudioBuffer::slice(std::size_t start, std::size_t count) const
    {
        AudioBuffer out;
        out.set_sample_rate(sample_rate_);
        if (start >= samples_.size())
            return out;
        const std::size_t end = std::min(start + count, samples_.size());
        out.samples_.assign(samples_.begin() + start, samples_.begin() + end);
        return out;
    }

    AudioBuffer &AudioBuffer::operator+=(const AudioBuffer &other)
    {
        append(other);
        return *this;
    }

    AudioBuffer AudioBuffer::silence(double seconds, int sample_rate)
    {
        const std::size_t n = static_cast<std::size_t>(seconds * sample_rate);
        return AudioBuffer(n, sample_rate);
    }

    AudioBuffer AudioBuffer::tone(double freq_hz, double seconds, int sample_rate,
                                  double amplitude)
    {
        const std::size_t n = static_cast<std::size_t>(seconds * sample_rate);
        AudioBuffer b(n, sample_rate);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(i) / sample_rate;
            b[i] = amplitude * std::sin(2.0 * kPi * freq_hz * t);
        }
        return b;
    }

    AudioBuffer AudioBuffer::noise(double seconds, int sample_rate, double amplitude,
                                   std::uint32_t seed)
    {
        const std::size_t n = static_cast<std::size_t>(seconds * sample_rate);
        AudioBuffer b(n, sample_rate);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-amplitude, amplitude);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = dist(rng);
        return b;
    }

    // ===========================================================================
    // WAV I/O
    // ===========================================================================
    namespace
    {

        void put_u32(std::ofstream &os, std::uint32_t v)
        {
            const char b[4] = {static_cast<char>(v & 0xFF),
                               static_cast<char>((v >> 8) & 0xFF),
                               static_cast<char>((v >> 16) & 0xFF),
                               static_cast<char>((v >> 24) & 0xFF)};
            os.write(b, 4);
        }

        void put_u16(std::ofstream &os, std::uint16_t v)
        {
            const char b[2] = {static_cast<char>(v & 0xFF),
                               static_cast<char>((v >> 8) & 0xFF)};
            os.write(b, 2);
        }

        std::uint32_t read_u32(std::istream &is)
        {
            unsigned char b[4];
            is.read(reinterpret_cast<char *>(b), 4);
            return static_cast<std::uint32_t>(b[0]) |
                   (static_cast<std::uint32_t>(b[1]) << 8) |
                   (static_cast<std::uint32_t>(b[2]) << 16) |
                   (static_cast<std::uint32_t>(b[3]) << 24);
        }

        std::uint16_t read_u16(std::istream &is)
        {
            unsigned char b[2];
            is.read(reinterpret_cast<char *>(b), 2);
            return static_cast<std::uint16_t>(b[0]) |
                   static_cast<std::uint16_t>(b[1] << 8);
        }

        double clamp_unit(double v) { return v > 1.0 ? 1.0 : (v < -1.0 ? -1.0 : v); }

    } // namespace

    bool wav_save_ex(const std::string &path, const AudioBuffer &buffer,
                     int bits_per_sample, std::string &error)
    {
        if (bits_per_sample != 8 && bits_per_sample != 16 && bits_per_sample != 24 &&
            bits_per_sample != 32)
        {
            error = "unsupported bit depth (use 8, 16, 24 or 32)";
            return false;
        }
        std::ofstream os(path, std::ios::binary);
        if (!os)
        {
            error = "cannot open file for writing: " + path;
            return false;
        }

        const std::uint16_t channels = 1;
        const std::uint32_t sample_rate =
            static_cast<std::uint32_t>(buffer.sample_rate() > 0 ? buffer.sample_rate()
                                                                : 16000);
        const std::uint16_t block_align =
            static_cast<std::uint16_t>(channels * (bits_per_sample / 8));
        const std::uint32_t byte_rate = sample_rate * block_align;
        const std::uint32_t data_bytes = static_cast<std::uint32_t>(
            buffer.size() * (bits_per_sample / 8));

        // RIFF header
        os.write("RIFF", 4);
        put_u32(os, 36 + data_bytes);
        os.write("WAVE", 4);
        // fmt chunk
        os.write("fmt ", 4);
        put_u32(os, 16);
        put_u16(os, 1); // PCM
        put_u16(os, channels);
        put_u32(os, sample_rate);
        put_u32(os, byte_rate);
        put_u16(os, block_align);
        put_u16(os, static_cast<std::uint16_t>(bits_per_sample));
        // data chunk
        os.write("data", 4);
        put_u32(os, data_bytes);

        for (double s : buffer.samples())
        {
            const double v = clamp_unit(s);
            if (bits_per_sample == 8)
            {
                // 8-bit WAV is unsigned
                const unsigned char byte =
                    static_cast<unsigned char>((v * 127.5) + 128.0);
                os.write(reinterpret_cast<const char *>(&byte), 1);
            }
            else if (bits_per_sample == 16)
            {
                const std::int16_t sample = static_cast<std::int16_t>(v * 32767.0);
                const char b[2] = {static_cast<char>(sample & 0xFF),
                                   static_cast<char>((sample >> 8) & 0xFF)};
                os.write(b, 2);
            }
            else if (bits_per_sample == 24)
            {
                const std::int32_t sample = static_cast<std::int32_t>(v * 8388607.0);
                const char b[3] = {static_cast<char>(sample & 0xFF),
                                   static_cast<char>((sample >> 8) & 0xFF),
                                   static_cast<char>((sample >> 16) & 0xFF)};
                os.write(b, 3);
            }
            else
            {
                const std::int32_t sample = static_cast<std::int32_t>(v * 2147483647.0);
                const char b[4] = {static_cast<char>(sample & 0xFF),
                                   static_cast<char>((sample >> 8) & 0xFF),
                                   static_cast<char>((sample >> 16) & 0xFF),
                                   static_cast<char>((sample >> 24) & 0xFF)};
                os.write(b, 4);
            }
        }
        os.flush();
        if (!os)
        {
            error = "write error on " + path;
            return false;
        }
        error.clear();
        return true;
    }

    bool wav_save(const std::string &path, const AudioBuffer &buffer,
                  std::string &error)
    {
        return wav_save_ex(path, buffer, 16, error);
    }

    bool wav_load(const std::string &path, AudioBuffer &out, WavInfo *info,
                  std::string &error)
    {
        std::ifstream is(path, std::ios::binary);
        if (!is)
        {
            error = "cannot open file: " + path;
            return false;
        }

        char riff[4];
        is.read(riff, 4);
        if (std::strncmp(riff, "RIFF", 4) != 0)
        {
            error = "not a RIFF file";
            return false;
        }
        read_u32(is); // chunk size
        char wave[4];
        is.read(wave, 4);
        if (std::strncmp(wave, "WAVE", 4) != 0)
        {
            error = "not a WAVE file";
            return false;
        }

        int channels = 1;
        int bits = 16;
        int sample_rate = 16000;
        bool have_fmt = false;
        AudioBuffer buffer;

        // Walk chunks until we find (and consume) the data chunk.
        while (is && !is.eof())
        {
            char id[4];
            is.read(id, 4);
            if (!is || is.gcount() < 4)
                break;
            const std::uint32_t size = read_u32(is);

            if (std::strncmp(id, "fmt ", 4) == 0)
            {
                const std::uint16_t audio_format = read_u16(is);
                channels = read_u16(is);
                sample_rate = static_cast<int>(read_u32(is));
                read_u32(is); // byte rate
                read_u16(is); // block align
                bits = read_u16(is);
                // Skip any extra fmt bytes.
                if (size > 16)
                    is.seekg(size - 16, std::ios::cur);
                if (audio_format != 1 && audio_format != 0xFFFE)
                {
                    error = "unsupported WAV format (only PCM)";
                    return false;
                }
                have_fmt = true;
            }
            else if (std::strncmp(id, "data", 4) == 0)
            {
                if (!have_fmt)
                {
                    error = "data chunk before fmt chunk";
                    return false;
                }
                const int bytes_per_sample = bits / 8;
                const std::size_t frames =
                    bytes_per_sample > 0 ? size / bytes_per_sample : 0;
                buffer.set_sample_rate(sample_rate);
                buffer.resize(frames);

                for (std::size_t i = 0; i < frames; ++i)
                {
                    double acc = 0.0;
                    for (int c = 0; c < channels; ++c)
                    {
                        if (bits == 8)
                        {
                            unsigned char b;
                            is.read(reinterpret_cast<char *>(&b), 1);
                            acc += (static_cast<double>(b) - 128.0) / 128.0;
                        }
                        else if (bits == 16)
                        {
                            const std::int16_t s = static_cast<std::int16_t>(read_u16(is));
                            acc += static_cast<double>(s) / 32768.0;
                        }
                        else if (bits == 24)
                        {
                            unsigned char b[3];
                            is.read(reinterpret_cast<char *>(b), 3);
                            std::int32_t s = static_cast<std::int32_t>(
                                b[0] | (b[1] << 8) | (b[2] << 16));
                            if (s & 0x800000)
                                s |= ~0xFFFFFF; // sign extend
                            acc += static_cast<double>(s) / 8388608.0;
                        }
                        else
                        {
                            const std::uint32_t raw = read_u32(is);
                            const std::int32_t s = static_cast<std::int32_t>(raw);
                            acc += static_cast<double>(s) / 2147483648.0;
                        }
                    }
                    buffer[i] = acc / static_cast<double>(channels);
                }
                out = buffer;
                if (info)
                {
                    info->sample_rate = sample_rate;
                    info->channels = channels;
                    info->bits_per_sample = bits;
                    info->num_frames = frames;
                }
                error.clear();
                return true;
            }
            else
            {
                // Unknown chunk: skip it (chunks are word-aligned).
                is.seekg(size + (size & 1), std::ios::cur);
            }
        }

        error = "no data chunk found";
        return false;
    }

    // ===========================================================================
    // DSP
    // ===========================================================================
    void fft(std::vector<double> &re, std::vector<double> &im, bool inverse)
    {
        const std::size_t n = re.size();
        if (n == 0 || (n & (n - 1)) != 0)
            return; // must be a power of two

        // Bit-reversal permutation.
        for (std::size_t i = 1, j = 0; i < n; ++i)
        {
            std::size_t bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }

        // Cooley-Tukey butterflies.
        for (std::size_t len = 2; len <= n; len <<= 1)
        {
            const double ang = 2.0 * kPi / static_cast<double>(len) *
                               (inverse ? 1.0 : -1.0);
            const double wlen_re = std::cos(ang);
            const double wlen_im = std::sin(ang);
            for (std::size_t i = 0; i < n; i += len)
            {
                double w_re = 1.0, w_im = 0.0;
                for (std::size_t k = 0; k < len / 2; ++k)
                {
                    const std::size_t u = i + k;
                    const std::size_t v = i + k + len / 2;
                    const double vr = re[v] * w_re - im[v] * w_im;
                    const double vi = re[v] * w_im + im[v] * w_re;
                    re[v] = re[u] - vr;
                    im[v] = im[u] - vi;
                    re[u] += vr;
                    im[u] += vi;
                    const double nwr = w_re * wlen_re - w_im * wlen_im;
                    const double nwi = w_re * wlen_im + w_im * wlen_re;
                    w_re = nwr;
                    w_im = nwi;
                }
            }
        }

        if (inverse)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                re[i] /= static_cast<double>(n);
                im[i] /= static_cast<double>(n);
            }
        }
    }

    std::vector<double> magnitude_spectrum(const AudioBuffer &buffer,
                                           std::size_t fft_size)
    {
        std::vector<double> re(fft_size, 0.0), im(fft_size, 0.0);
        const std::size_t n = std::min(fft_size, buffer.size());
        for (std::size_t i = 0; i < n; ++i)
        {
            // Hann window to reduce spectral leakage.
            const double w = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                                   static_cast<double>(n - 1)));
            re[i] = buffer[i] * w;
        }
        fft(re, im);
        std::vector<double> mags(fft_size / 2 + 1);
        for (std::size_t i = 0; i < mags.size(); ++i)
            mags[i] = std::sqrt(re[i] * re[i] + im[i] * im[i]);
        return mags;
    }

    namespace
    {

        // Mel-scale helpers.
        double hz_to_mel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
        double mel_to_hz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

        // DCT-II (used to decorrelate the log-mel spectrum into cepstral coefficients).
        std::vector<double> dct(const std::vector<double> &in, int num_out)
        {
            const std::size_t n = in.size();
            std::vector<double> out(static_cast<std::size_t>(num_out), 0.0);
            for (int k = 0; k < num_out; ++k)
            {
                double acc = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    acc += in[i] * std::cos(kPi * static_cast<double>(k) *
                                            (static_cast<double>(i) + 0.5) /
                                            static_cast<double>(n));
                out[static_cast<std::size_t>(k)] = acc;
            }
            return out;
        }

    } // namespace

    std::vector<std::vector<double>> mfcc(const AudioBuffer &buffer, int num_ceps,
                                          int num_filters, std::size_t frame_len,
                                          std::size_t hop)
    {
        std::vector<std::vector<double>> result;
        if (buffer.empty() || num_ceps <= 0 || num_filters < 2)
            return result;

        const int sample_rate = buffer.sample_rate() > 0 ? buffer.sample_rate() : 16000;
        std::size_t fft_size = 1;
        while (fft_size < frame_len)
            fft_size <<= 1;

        // Mel filterbank: triangular filters on the mel scale.
        const double low_mel = hz_to_mel(50.0);
        const double high_mel = hz_to_mel(sample_rate * 0.5);
        std::vector<double> mel_points(static_cast<std::size_t>(num_filters + 2));
        for (int i = 0; i < num_filters + 2; ++i)
        {
            const double mel = low_mel + (high_mel - low_mel) *
                                             static_cast<double>(i) /
                                             static_cast<double>(num_filters + 1);
            const double hz = mel_to_hz(mel);
            mel_points[static_cast<std::size_t>(i)] =
                (fft_size / 2 + 1) * hz / (sample_rate * 0.5);
        }

        for (std::size_t start = 0; start + frame_len <= buffer.size(); start += hop)
        {
            // windowed frame
            std::vector<double> re(fft_size, 0.0), im(fft_size, 0.0);
            for (std::size_t i = 0; i < frame_len; ++i)
            {
                const double w = 0.54 - 0.46 * std::cos(2.0 * kPi *
                                                        static_cast<double>(i) /
                                                        static_cast<double>(frame_len - 1));
                re[i] = buffer[start + i] * w;
            }
            fft(re, im);

            const std::size_t bins = fft_size / 2 + 1;
            std::vector<double> power(bins);
            for (std::size_t i = 0; i < bins; ++i)
                power[i] = (re[i] * re[i] + im[i] * im[i]) / static_cast<double>(fft_size);

            // Apply the triangular filters.
            std::vector<double> log_mel(static_cast<std::size_t>(num_filters), 0.0);
            for (int f = 0; f < num_filters; ++f)
            {
                const double left = mel_points[static_cast<std::size_t>(f)];
                const double center = mel_points[static_cast<std::size_t>(f + 1)];
                const double right = mel_points[static_cast<std::size_t>(f + 2)];
                double acc = 0.0;
                for (std::size_t b = 0; b < bins; ++b)
                {
                    const double bd = static_cast<double>(b);
                    double w = 0.0;
                    if (bd >= left && bd <= center && center > left)
                        w = (bd - left) / (center - left);
                    else if (bd > center && bd <= right && right > center)
                        w = (right - bd) / (right - center);
                    acc += power[b] * w;
                }
                log_mel[static_cast<std::size_t>(f)] = std::log(acc + 1e-10);
            }

            result.push_back(dct(log_mel, num_ceps));
        }
        return result;
    }

    AudioBuffer resample(const AudioBuffer &in, int target_rate)
    {
        if (in.empty() || in.sample_rate() == target_rate || in.sample_rate() <= 0)
            return in;
        const double ratio = static_cast<double>(target_rate) /
                             static_cast<double>(in.sample_rate());
        const std::size_t out_n =
            static_cast<std::size_t>(static_cast<double>(in.size()) * ratio);
        AudioBuffer out(out_n, target_rate);
        for (std::size_t i = 0; i < out_n; ++i)
        {
            const double src = static_cast<double>(i) / ratio;
            const std::size_t i0 = static_cast<std::size_t>(src);
            const std::size_t i1 = std::min(i0 + 1, in.size() - 1);
            const double frac = src - static_cast<double>(i0);
            out[i] = in[i0] * (1.0 - frac) + in[i1] * frac; // linear interpolation
        }
        return out;
    }

    void low_pass(AudioBuffer &buffer, double cutoff_hz)
    {
        if (buffer.empty() || buffer.sample_rate() <= 0)
            return;
        const double dt = 1.0 / static_cast<double>(buffer.sample_rate());
        const double rc = 1.0 / (2.0 * kPi * cutoff_hz);
        const double alpha = dt / (rc + dt);
        double prev = buffer[0];
        for (std::size_t i = 1; i < buffer.size(); ++i)
        {
            prev = prev + alpha * (buffer[i] - prev);
            buffer[i] = prev;
        }
    }

    void smooth(AudioBuffer &buffer, std::size_t window)
    {
        if (window < 2 || buffer.size() < window)
            return;
        std::vector<double> out(buffer.size(), 0.0);
        double acc = 0.0;
        for (std::size_t i = 0; i < buffer.size(); ++i)
        {
            acc += buffer[i];
            if (i >= window)
                acc -= buffer[i - window];
            out[i] = acc / static_cast<double>(std::min(i + 1, window));
        }
        for (std::size_t i = 0; i < buffer.size(); ++i)
            buffer[i] = out[i];
    }

    double zero_crossing_rate(const AudioBuffer &buffer)
    {
        if (buffer.size() < 2)
            return 0.0;
        std::size_t crossings = 0;
        for (std::size_t i = 1; i < buffer.size(); ++i)
            if ((buffer[i - 1] >= 0.0) != (buffer[i] >= 0.0))
                ++crossings;
        return static_cast<double>(crossings) / static_cast<double>(buffer.size() - 1);
    }

    std::vector<double> frame_energy(const AudioBuffer &buffer, std::size_t frame_len,
                                     std::size_t hop)
    {
        std::vector<double> out;
        if (frame_len == 0)
            return out;
        for (std::size_t start = 0; start + frame_len <= buffer.size(); start += hop)
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < frame_len; ++i)
                acc += buffer[start + i] * buffer[start + i];
            out.push_back(acc / static_cast<double>(frame_len));
        }
        return out;
    }

} // namespace chappie
