// ---------------------------------------------------------------------------
// audio/tts.cpp -- formant speech synthesizer implementation.
// ---------------------------------------------------------------------------
#include "audio/tts.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>

#include "core/logger.hpp"

namespace chappie
{

    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        // A simple 2-pole resonant band-pass filter, applied forward and backward so
        // the phase distortion cancels (giving a clean resonant peak).
        struct Resonator
        {
            double b0 = 0.0, a1 = 0.0, a2 = 0.0;

            void configure(double freq_hz, double bandwidth_hz, double sample_rate)
            {
                const double r = std::exp(-kPi * bandwidth_hz / sample_rate);
                const double theta = 2.0 * kPi * freq_hz / sample_rate;
                a1 = -2.0 * r * std::cos(theta);
                a2 = r * r;
                b0 = (1.0 - r) * std::sqrt(1.0 - 2.0 * r * std::cos(2.0 * theta) + r * r);
            }

            void run(const std::vector<double> &in, std::vector<double> &out) const
            {
                out.assign(in.size(), 0.0);
                double y1 = 0.0, y2 = 0.0;
                for (std::size_t i = 0; i < in.size(); ++i)
                {
                    const double y = b0 * in[i] - a1 * y1 - a2 * y2;
                    out[i] = y;
                    y2 = y1;
                    y1 = y;
                }
            }
        };

        // One-pole low-pass, used for anti-aliasing the noise source.
        void one_pole_lowpass(std::vector<double> &x, double cutoff, double sr)
        {
            const double alpha = 1.0 - std::exp(-2.0 * kPi * cutoff / sr);
            double prev = 0.0;
            for (double &v : x)
            {
                prev += alpha * (v - prev);
                v = prev;
            }
        }

    } // namespace

    // ===========================================================================
    // VoiceProfile presets
    // ===========================================================================
    VoiceProfile VoiceProfile::synthetic_young()
    {
        VoiceProfile p;
        p.name = "synthetic-young";
        p.base_pitch_hz = 132.0;
        p.pitch_range_hz = 48.0;
        p.formant_scale = 1.15;
        p.breathiness = 0.06;
        p.roughness = 0.12;
        p.brightness = 0.58;
        p.amplitude = 0.85;
        return p;
    }

    VoiceProfile VoiceProfile::deep_calm()
    {
        VoiceProfile p;
        p.name = "deep-calm";
        p.base_pitch_hz = 96.0;
        p.pitch_range_hz = 26.0;
        p.formant_scale = 0.94;
        p.speed = 0.92;
        p.breathiness = 0.03;
        p.roughness = 0.05;
        p.brightness = 0.40;
        p.amplitude = 0.9;
        return p;
    }

    VoiceProfile VoiceProfile::high_energetic()
    {
        VoiceProfile p;
        p.name = "high-energetic";
        p.base_pitch_hz = 175.0;
        p.pitch_range_hz = 70.0;
        p.formant_scale = 1.25;
        p.speed = 1.12;
        p.breathiness = 0.09;
        p.roughness = 0.18;
        p.brightness = 0.72;
        p.amplitude = 0.9;
        return p;
    }

    VoiceProfile VoiceProfile::neutral()
    {
        VoiceProfile p;
        p.name = "neutral";
        p.base_pitch_hz = 120.0;
        p.pitch_range_hz = 35.0;
        p.formant_scale = 1.0;
        p.breathiness = 0.04;
        p.roughness = 0.06;
        p.brightness = 0.5;
        p.amplitude = 0.85;
        return p;
    }

    // ===========================================================================
    // TtsEngine
    // ===========================================================================
    TtsEngine::TtsEngine() : profile_(VoiceProfile::synthetic_young()) {}

    TtsEngine::TtsEngine(const VoiceProfile &profile) : profile_(profile) {}

    std::vector<std::string> TtsEngine::available_voices()
    {
        return {"synthetic-young", "deep-calm", "high-energetic", "neutral"};
    }

    bool TtsEngine::set_voice(const std::string &name)
    {
        if (name == "synthetic-young")
            profile_ = VoiceProfile::synthetic_young();
        else if (name == "deep-calm")
            profile_ = VoiceProfile::deep_calm();
        else if (name == "high-energetic")
            profile_ = VoiceProfile::high_energetic();
        else if (name == "neutral")
            profile_ = VoiceProfile::neutral();
        else
            return false;
        return true;
    }

    SpeakParams TtsEngine::emotion_params(const std::string &emotion,
                                          const std::string &text)
    {
        SpeakParams p;
        p.emotion = emotion;
        if (emotion == "happy")
        {
            p.pitch_shift_hz = 18.0;
            p.rate = 1.08;
            p.volume = 1.0;
        }
        else if (emotion == "excited")
        {
            p.pitch_shift_hz = 30.0;
            p.rate = 1.16;
            p.volume = 1.05;
        }
        else if (emotion == "sad")
        {
            p.pitch_shift_hz = -18.0;
            p.rate = 0.88;
            p.volume = 0.85;
        }
        else if (emotion == "curious")
        {
            p.pitch_shift_hz = 12.0;
            p.rate = 1.02;
            p.volume = 1.0;
        }
        // Detect question / exclamation from punctuation.
        if (!text.empty())
        {
            const char last = text.back();
            if (last == '?')
                p.is_question = true;
            if (last == '!')
                p.is_exclamation = true;
        }
        return p;
    }

    // ---------------------------------------------------------------------------
    // Glottal source: a pulse train with jitter and vibrato.
    // ---------------------------------------------------------------------------
    AudioBuffer TtsEngine::glottal_source(double duration_s, double f0_hz,
                                          int sample_rate,
                                          std::uint32_t seed) const
    {
        const std::size_t n =
            static_cast<std::size_t>(std::max(0.0, duration_s) * sample_rate);
        AudioBuffer out(n, sample_rate);
        if (n == 0 || f0_hz <= 0.0)
            return out;

        std::mt19937 rng(seed);
        std::normal_distribution<double> jitter(0.0, profile_.pitch_jitter);

        const double period = static_cast<double>(sample_rate) / f0_hz;
        double phase = 0.0;
        std::size_t i = 0;
        while (i < n)
        {
            // Vibrato modulates the instantaneous pitch.
            const double t = static_cast<double>(i) / sample_rate;
            const double vib = 1.0 + profile_.vibrato_depth *
                                         std::sin(2.0 * kPi * profile_.vibrato_hz * t);
            const double jit = 1.0 + jitter(rng);
            const double inst_period = period * vib * jit;

            // One glottal cycle: a short asymmetric pulse (Rosenberg-like).
            const std::size_t cycle = static_cast<std::size_t>(
                std::max(1.0, inst_period * (1.0 + 0.6 * profile_.roughness)));
            for (std::size_t k = 0; k < cycle && i < n; ++k, ++i)
            {
                const double x = static_cast<double>(k) / static_cast<double>(cycle);
                double pulse;
                if (x < 0.4)
                {
                    // opening phase
                    const double u = x / 0.4;
                    pulse = 0.5 * (1.0 - std::cos(kPi * u));
                }
                else if (x < 0.6)
                {
                    // closing phase
                    const double u = (x - 0.4) / 0.2;
                    pulse = 0.5 * (1.0 + std::cos(kPi * u));
                }
                else
                {
                    pulse = 0.0;
                }
                // Add roughness as a small asymmetric perturbation.
                if (profile_.roughness > 0.0)
                    pulse += profile_.roughness * 0.15 *
                             std::sin(2.0 * kPi * x * 3.0) * (1.0 - x);
                out[i] = pulse - 0.15; // remove DC offset of the pulse shape
            }
            phase += 1.0;
        }
        return out;
    }

    // ---------------------------------------------------------------------------
    // Formant filtering
    // ---------------------------------------------------------------------------
    void TtsEngine::apply_formants(AudioBuffer &buffer, double f1, double f2,
                                   double f3, double bandwidth) const
    {
        std::vector<double> sig = buffer.samples();
        if (sig.empty())
            return;

        const double sr = static_cast<double>(buffer.sample_rate());
        const double scale = profile_.formant_scale;

        // Resonate at each formant, summing the results with decreasing weight.
        const double freqs[3] = {f1 * scale, f2 * scale, f3 * scale};
        const double gains[3] = {1.0, 0.62, 0.28};

        std::vector<double> mixed(sig.size(), 0.0);
        for (int i = 0; i < 3; ++i)
        {
            if (freqs[i] <= 0.0 || freqs[i] >= sr * 0.48)
                continue;
            Resonator r;
            r.configure(freqs[i], bandwidth * (1.0 + 0.25 * i), sr);
            std::vector<double> filtered;
            r.run(sig, filtered);
            for (std::size_t k = 0; k < mixed.size(); ++k)
                mixed[k] += gains[i] * filtered[k];
        }

        // Mix the formant output with a little of the raw source for "brightness".
        const double b = profile_.brightness;
        for (std::size_t k = 0; k < mixed.size(); ++k)
            buffer[k] = mixed[k] + (1.0 - b) * 0.25 * sig[k];

        buffer.normalize(0.95);
    }

    // ---------------------------------------------------------------------------
    // Per-phoneme renderers
    // ---------------------------------------------------------------------------
    AudioBuffer TtsEngine::render_vowel(const Phoneme &p, double duration_s,
                                        double f0_hz, int sample_rate) const
    {
        AudioBuffer src = glottal_source(duration_s, f0_hz, sample_rate,
                                         static_cast<std::uint32_t>(p.f1));
        if (profile_.breathiness > 0.0)
        {
            // Mix aspiration noise into the voiced source.
            AudioBuffer noise = AudioBuffer::noise(duration_s, sample_rate,
                                                   profile_.breathiness, 1234);
            for (std::size_t i = 0; i < src.size() && i < noise.size(); ++i)
                src[i] += noise[i];
        }
        apply_formants(src, p.f1, p.f2, p.f3, 90.0);
        src.apply_gain(p.amplitude * profile_.amplitude);
        return src;
    }

    AudioBuffer TtsEngine::render_plosive(const Phoneme &p, double duration_s,
                                          double f0_hz, int sample_rate) const
    {
        const std::size_t n =
            static_cast<std::size_t>(std::max(0.01, duration_s) * sample_rate);
        AudioBuffer out(n, sample_rate);

        // Closure (silence) then a short broadband burst.
        const std::size_t closure = n / 3;
        std::mt19937 rng(static_cast<std::uint32_t>(p.symbol[0]) * 7919u + 13u);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (std::size_t i = closure; i < n; ++i)
        {
            const double t = static_cast<double>(i - closure) /
                             static_cast<double>(std::max<std::size_t>(1, n - closure));
            // Sharp attack, exponential decay.
            const double env = std::exp(-8.0 * t);
            out[i] = dist(rng) * env;
        }

        // Voiced plosives (b, d, g) get a little glottal buzz after the burst.
        if (p.voiced)
        {
            AudioBuffer voice = glottal_source(duration_s * 0.6, f0_hz, sample_rate,
                                               991u + p.f1);
            const std::size_t offset = n - voice.size();
            for (std::size_t i = 0; i < voice.size() && offset + i < n; ++i)
                out[offset + i] += 0.35 * voice[i];
        }

        // Emphasise the burst's high frequencies for a crisp "pop".
        one_pole_lowpass(out.samples(), 4200.0, static_cast<double>(sample_rate));
        out.normalize(0.8);
        out.apply_gain(p.amplitude * profile_.amplitude);
        return out;
    }

    AudioBuffer TtsEngine::render_fricative(const Phoneme &p, double duration_s,
                                            double f0_hz, int sample_rate) const
    {
        const std::size_t n =
            static_cast<std::size_t>(std::max(0.01, duration_s) * sample_rate);
        AudioBuffer out(n, sample_rate);

        std::mt19937 rng(static_cast<std::uint32_t>(p.symbol.size() * 131u +
                                                    (p.symbol.empty() ? 0 : p.symbol[0])));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = dist(rng);

        // Shape the noise to the phoneme's spectral character.
        double low_cut = 2000.0;
        double high_cut = 8000.0;
        if (p.symbol == "S")
        {
            low_cut = 4500.0;
            high_cut = 9000.0;
        }
        else if (p.symbol == "SH")
        {
            low_cut = 2200.0;
            high_cut = 6500.0;
        }
        else if (p.symbol == "F")
        {
            low_cut = 1800.0;
            high_cut = 7500.0;
        }
        else if (p.symbol == "TH")
        {
            low_cut = 1400.0;
            high_cut = 7000.0;
        }
        else if (p.symbol == "HH")
        {
            low_cut = 800.0;
            high_cut = 5000.0;
        }
        else if (p.symbol == "V" || p.symbol == "Z" || p.symbol == "ZH")
        {
            low_cut = 1200.0;
            high_cut = 6000.0;
        }

        // High-pass via subtracting a low-passed copy.
        std::vector<double> low = out.samples();
        one_pole_lowpass(low, low_cut, static_cast<double>(sample_rate));
        for (std::size_t i = 0; i < n; ++i)
            out[i] -= low[i];

        // Low-pass the remainder.
        one_pole_lowpass(out.samples(), high_cut, static_cast<double>(sample_rate));

        // Voiced fricatives mix in glottal buzz.
        if (p.voiced)
        {
            AudioBuffer voice = glottal_source(duration_s, f0_hz, sample_rate, 555u);
            for (std::size_t i = 0; i < n && i < voice.size(); ++i)
                out[i] += 0.3 * voice[i];
        }

        // Envelope: fade in/out so there are no clicks.
        const std::size_t fade = std::max<std::size_t>(1, n / 10);
        out.fade_in(fade);
        out.fade_out(fade);
        out.normalize(0.7);
        out.apply_gain(p.amplitude * profile_.amplitude);
        return out;
    }

    AudioBuffer TtsEngine::render_affricate(const Phoneme &p, double duration_s,
                                            double f0_hz, int sample_rate) const
    {
        // Affricate = stop burst followed immediately by a fricative.
        AudioBuffer stop = render_plosive(p, duration_s * 0.4, f0_hz, sample_rate);
        Phoneme fric = p;
        fric.symbol = (p.symbol == "CH") ? "SH" : "ZH";
        fric.type = PhonemeType::Fricative;
        AudioBuffer hiss = render_fricative(fric, duration_s * 0.6, f0_hz, sample_rate);
        stop.append(hiss);
        return stop;
    }

    AudioBuffer TtsEngine::render_nasal(const Phoneme &p, double duration_s,
                                        double f0_hz, int sample_rate) const
    {
        AudioBuffer src = glottal_source(duration_s, f0_hz, sample_rate, 777u);
        // Nasals: strong low formant (the nasal murmur) plus damped higher ones.
        apply_formants(src, p.f1, p.f2, p.f3, 180.0);
        src.apply_gain(p.amplitude * profile_.amplitude * 0.85);
        return src;
    }

    AudioBuffer TtsEngine::render_liquid_glide(const Phoneme &p, double duration_s,
                                               double f0_hz, int sample_rate) const
    {
        AudioBuffer src = glottal_source(duration_s, f0_hz, sample_rate, 888u);
        apply_formants(src, p.f1, p.f2, p.f3, 110.0);
        src.apply_gain(p.amplitude * profile_.amplitude * 0.9);
        return src;
    }

    AudioBuffer TtsEngine::render_silence(double duration_s, int sample_rate) const
    {
        return AudioBuffer::silence(duration_s, sample_rate);
    }

    // ---------------------------------------------------------------------------
    // Prosody
    // ---------------------------------------------------------------------------
    AudioBuffer TtsEngine::apply_prosody(const AudioBuffer &in,
                                         const SpeakParams &params,
                                         bool final_question) const
    {
        AudioBuffer out = in;
        if (out.empty())
            return out;

            // The pitch contour was already baked into the per-phoneme F0 values; here
            // we apply the global amplitude envelope that gives the utterance a
            // natural start and end.
            const double total = static_cast<double>(out.size());

            // Gentle overall envelope so the utterance starts and ends cleanly.
            const std::size_t fade = static_cast<std::size_t>(total * 0.01);
            out.fade_in(fade);
            out.fade_out(fade);

            out.apply_gain(params.volume);
            out.normalize(0.95);
            return out;
        }

    // ---------------------------------------------------------------------------
    // Full synthesis
    // ---------------------------------------------------------------------------
    AudioBuffer TtsEngine::speak_phonemes(const std::vector<Phoneme> &phonemes,
                                          const SpeakParams &params) const
    {
        const int sr = params.sample_rate;
        AudioBuffer result(0, sr);
        if (phonemes.empty())
            return result;

        const double base_f0 = profile_.base_pitch_hz + params.pitch_shift_hz;
        const double rate = params.rate * profile_.speed;

        // Compute the pitch contour across the utterance.
        const std::size_t total = phonemes.size();
        for (std::size_t i = 0; i < total; ++i)
        {
            const Phoneme &p = phonemes[i];
            const double t = static_cast<double>(i) / static_cast<double>(total);

            // Base F0 with a natural declination over the phrase, plus a rise for
            // questions and a fall for statements at the very end.
            double f0 = base_f0 * (1.0 - 0.12 * t);
            if (params.is_question)
            {
                f0 += profile_.question_rise_hz * t * t;
            }
            else if (params.is_exclamation)
            {
                f0 += profile_.pitch_range_hz * 0.35 * (1.0 - t);
            }
            else
            {
                f0 -= profile_.statement_fall_hz * 0.35 * t;
            }

            // Emotion shapes the contour further.
            if (params.emotion == "sad")
                f0 *= 0.92;
            if (params.emotion == "excited")
                f0 *= 1.06;
            if (params.emotion == "curious")
                f0 += 10.0 * std::sin(kPi * t);

            const double dur = (p.duration_ms / 1000.0) / std::max(0.2, rate);

            AudioBuffer seg;
            switch (p.type)
            {
            case PhonemeType::Vowel:
                seg = render_vowel(p, dur, f0, sr);
                break;
            case PhonemeType::Plosive:
                seg = render_plosive(p, dur, f0, sr);
                break;
            case PhonemeType::Fricative:
                seg = render_fricative(p, dur, f0, sr);
                break;
            case PhonemeType::Affricate:
                seg = render_affricate(p, dur, f0, sr);
                break;
            case PhonemeType::Nasal:
                seg = render_nasal(p, dur, f0, sr);
                break;
            case PhonemeType::Liquid:
            case PhonemeType::Glide:
                seg = render_liquid_glide(p, dur, f0, sr);
                break;
            case PhonemeType::Silence:
                seg = render_silence(dur, sr);
                break;
            }

            // Short crossfade between segments to avoid clicks.
            if (!result.empty() && !seg.empty())
            {
                const std::size_t xf = std::min<std::size_t>(
                    32, std::min(result.size(), seg.size()));
                const std::size_t start = result.size() - xf;
                for (std::size_t k = 0; k < xf; ++k)
                {
                    const double w = static_cast<double>(k) / static_cast<double>(xf);
                    result[start + k] = result[start + k] * (1.0 - w) + seg[k] * w;
                }
                for (std::size_t k = xf; k < seg.size(); ++k)
                    result.push_back(seg[k]);
            }
            else
            {
                result.append(seg);
            }
        }

        return apply_prosody(result, params, params.is_question);
    }

    AudioBuffer TtsEngine::speak(const std::string &text,
                                 const SpeakParams &params) const
    {
        const auto phonemes = text_to_phonemes(text);
        return speak_phonemes(phonemes, params);
    }

    bool TtsEngine::speak_to_file(const std::string &text, const std::string &path,
                                  const SpeakParams &params,
                                  std::string &error) const
    {
        const AudioBuffer buffer = speak(text, params);
        if (buffer.empty())
        {
            error = "synthesis produced no audio";
            return false;
        }
        return wav_save(path, buffer, error);
    }

    // ===========================================================================
    // Chappie voice pack
    // ===========================================================================
    const std::vector<VoiceLine> &chappie_voice_lines()
    {
        static const std::vector<VoiceLine> lines = {
            {"boot", "Systems online. Hello. I am Chappie.", "happy"},
            {"greet", "Hello. It is good to see you.", "happy"},
            {"curious", "What are we doing today?", "curious"},
            {"ack", "Okay. I understand.", "neutral"},
            {"yes", "Yes. I can do that.", "neutral"},
            {"no", "No. I cannot do that. It is not safe.", "neutral"},
            {"thinking", "Let me think about that.", "curious"},
            {"found", "I found a path. Following it now.", "neutral"},
            {"blocked", "There is an obstacle in my way.", "sad"},
            {"warning", "Warning. My battery is running low.", "sad"},
            {"learned", "I learned something new today.", "excited"},
            {"happy", "That makes me happy.", "happy"},
            {"scared", "I am scared. Please help me.", "sad"},
            {"alert", "I see something moving.", "curious"},
            {"shutdown", "Goodbye. See you soon.", "sad"},
        };
        return lines;
    }

    int render_voice_pack(const std::string &out_dir, const VoiceProfile &profile)
    {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);

        TtsEngine tts(profile);
        int written = 0;
        for (const auto &line : chappie_voice_lines())
        {
            SpeakParams params = TtsEngine::emotion_params(line.emotion, line.text);
            params.sample_rate = 22050;
            const std::string path = out_dir + "/" + line.id + ".wav";
            std::string error;
            if (tts.speak_to_file(line.text, path, params, error))
            {
                ++written;
                LOG_INFO() << "rendered voice line: " << line.id << " -> " << path;
            }
            else
            {
                LOG_ERROR() << "failed to render '" << line.id << "': " << error;
            }
        }
        return written;
    }

} // namespace chappie
