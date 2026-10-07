// ---------------------------------------------------------------------------
// audio/tts.hpp -- text-to-speech engine.
//
// A formant (source-filter) synthesizer:
//
//   text -> phonemes -> per-phoneme acoustic frames -> glottal source +
//   formant filtering + noise -> prosody (pitch/duration) -> AudioBuffer
//
// The glottal source is a pulse train at the target F0; formants are applied as
// a bank of resonant band-pass filters; fricatives mix in shaped noise. Prosody
// (pitch contour, phrase-final lengthening, stress) is layered on top from the
// VoiceProfile.
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "audio/phoneme.hpp"

namespace chappie
{

    // ---------------------------------------------------------------------------
    // VoiceProfile -- the "who is speaking" parameters.
    //
    // Chappie's default profile is a young, slightly synthetic male-ish voice with
    // a bright, curious pitch contour.
    // ---------------------------------------------------------------------------
    struct VoiceProfile
    {
        std::string name = "synthetic-young";

        // Pitch (F0) range in Hz.
        double base_pitch_hz = 130.0; // mean F0
        double pitch_range_hz = 45.0; // +/- variation for intonation
        double pitch_jitter = 0.012;  // random cycle-to-cycle variation
        double vibrato_hz = 5.0;      // subtle vibrato
        double vibrato_depth = 0.015;

        // Formant scaling (1.0 = average adult male).
        double formant_scale = 1.14; // higher = smaller vocal tract (younger)

        // Timing.
        double speed = 1.0; // >1 = faster speech
        double base_phrase_pause_ms = 120.0;

        // Voice quality.
        double breathiness = 0.06; // aspiration noise mixed into voicing
        double roughness = 0.10;   // slight rasp -> "synthetic" character
        double brightness = 0.55;  // high-frequency emphasis

        // Dynamic range.
        double amplitude = 0.85;

        // Prosody: how much pitch rises for questions, falls for statements.
        double question_rise_hz = 55.0;
        double statement_fall_hz = 35.0;

        // Presets.
        static VoiceProfile synthetic_young();
        static VoiceProfile deep_calm();
        static VoiceProfile high_energetic();
        static VoiceProfile neutral();
    };

    // ---------------------------------------------------------------------------
    // Synthesis parameters passed per call.
    // ---------------------------------------------------------------------------
    struct SpeakParams
    {
        double pitch_shift_hz = 0.0; // additive shift applied to base pitch
        double rate = 1.0;           // multiplicative speed (1 = normal)
        double volume = 1.0;         // output gain
        bool is_question = false;
        bool is_exclamation = false;
        int sample_rate = 22050;
        std::string emotion = "neutral"; // neutral, happy, sad, curious, excited
    };

    // ---------------------------------------------------------------------------
    // TtsEngine -- the synthesizer.
    // ---------------------------------------------------------------------------
    class TtsEngine
    {
    public:
        TtsEngine();
        explicit TtsEngine(const VoiceProfile &profile);

        // Synthesize text into an audio buffer.
        AudioBuffer speak(const std::string &text,
                          const SpeakParams &params = SpeakParams{}) const;

        // Synthesize directly from a phoneme sequence (skips G2P).
        AudioBuffer speak_phonemes(const std::vector<Phoneme> &phonemes,
                                   const SpeakParams &params = SpeakParams{}) const;

        // Render to a .wav file directly.
        bool speak_to_file(const std::string &text, const std::string &path,
                           const SpeakParams &params, std::string &error) const;

        // Voice management.
        const VoiceProfile &profile() const { return profile_; }
        void set_profile(const VoiceProfile &p) { profile_ = p; }

        // List available voice preset names.
        static std::vector<std::string> available_voices();
        // Load a preset by name; returns false if unknown.
        bool set_voice(const std::string &name);

        // Emotion presets modulate pitch/rate/volume.
        static SpeakParams emotion_params(const std::string &emotion,
                                          const std::string &text);

    private:
        VoiceProfile profile_;

        // Per-phoneme rendering.
        AudioBuffer render_vowel(const Phoneme &p, double duration_s,
                                 double f0_hz, int sample_rate) const;
        AudioBuffer render_plosive(const Phoneme &p, double duration_s,
                                   double f0_hz, int sample_rate) const;
        AudioBuffer render_fricative(const Phoneme &p, double duration_s,
                                     double f0_hz, int sample_rate) const;
        AudioBuffer render_affricate(const Phoneme &p, double duration_s,
                                     double f0_hz, int sample_rate) const;
        AudioBuffer render_nasal(const Phoneme &p, double duration_s, double f0_hz,
                                 int sample_rate) const;
        AudioBuffer render_liquid_glide(const Phoneme &p, double duration_s,
                                        double f0_hz, int sample_rate) const;
        AudioBuffer render_silence(double duration_s, int sample_rate) const;

        // Glottal pulse train with jitter + vibrato.
        AudioBuffer glottal_source(double duration_s, double f0_hz,
                                   int sample_rate, std::uint32_t seed) const;

        // Apply a bank of resonant formant filters (in place).
        void apply_formants(AudioBuffer &buffer, double f1, double f2, double f3,
                            double bandwidth = 80.0) const;

        // Apply the pitch contour for the whole utterance.
        AudioBuffer apply_prosody(const AudioBuffer &in, const SpeakParams &params,
                                  bool final_question) const;
    };

    // ---------------------------------------------------------------------------
    // Convenience: the classic Chappie lines, pre-rendered to WAV files.
    // ---------------------------------------------------------------------------
    struct VoiceLine
    {
        std::string id;
        std::string text;
        std::string emotion;
    };

    const std::vector<VoiceLine> &chappie_voice_lines();

    // Render every built-in line to `out_dir` as .wav. Returns count written.
    int render_voice_pack(const std::string &out_dir,
                          const VoiceProfile &profile = VoiceProfile::synthetic_young());

} // namespace chappie
