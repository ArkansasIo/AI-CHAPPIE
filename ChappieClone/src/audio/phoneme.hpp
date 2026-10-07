// ---------------------------------------------------------------------------
// audio/phoneme.hpp -- grapheme-to-phoneme (G2P) conversion and the phoneme
// inventory used by the speech synthesizer.
//
// English spelling is irregular, so the converter combines:
//   1. a lexicon of common irregular words (exact lookup)
//   2. ordered letter-to-sound rules (longest match first)
//   3. vowel/consonant context adjustments
//
// The output is a list of Phoneme tokens with ARPAbet-style symbols, each
// carrying the acoustic parameters the synthesizer needs (formant targets,
// voicing, duration).
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

namespace chappie
{

    // Phoneme categories drive the synthesis strategy.
    enum class PhonemeType
    {
        Vowel,     // voiced, formant-based (has F1/F2 targets)
        Plosive,   // stop burst: p b t d k g
        Fricative, // noisy: f v s z sh th h
        Affricate, // ch j (stop + fricative)
        Nasal,     // m n ng
        Liquid,    // l r
        Glide,     // w y
        Silence    // pause / word boundary
    };

    // ---------------------------------------------------------------------------
    // A single phoneme with its acoustic description.
    // ---------------------------------------------------------------------------
    struct Phoneme
    {
        std::string symbol; // ARPAbet-ish, e.g. "AE", "T", "SH"
        PhonemeType type = PhonemeType::Silence;
        double duration_ms = 80.0;
        double f1 = 500.0;      // first formant (Hz) -- vowel quality
        double f2 = 1500.0;     // second formant -- vowel place
        double f3 = 2500.0;     // third formant
        bool voiced = true;     // does the glottal source run?
        double amplitude = 1.0; // relative loudness
        double noise = 0.0;     // amount of fricative noise to mix in

        bool is_vowel() const { return type == PhonemeType::Vowel; }
        bool is_silence() const { return type == PhonemeType::Silence; }
    };

    // ---------------------------------------------------------------------------
    // The phoneme inventory (ARPAbet subset used by this synthesizer).
    // ---------------------------------------------------------------------------
    const std::vector<Phoneme> &phoneme_inventory();
    const Phoneme *find_phoneme(const std::string &symbol);

    // ---------------------------------------------------------------------------
    // Grapheme-to-phoneme conversion.
    // ---------------------------------------------------------------------------
    // Convert a single word to a phoneme sequence.
    std::vector<Phoneme> word_to_phonemes(const std::string &word);

    // Convert a sentence (handles punctuation -> pauses) to a phoneme sequence.
    std::vector<Phoneme> text_to_phonemes(const std::string &text);

    // A readable string form, e.g. "CH AE P IY" for "chappie".
    std::string phonemes_to_string(const std::vector<Phoneme> &phonemes);

    // Stress marks: which syllable of a word carries primary stress (0-based).
    int primary_stress_syllable(const std::string &word);

    // Syllable count estimate (vowel-group counting with silent-e correction).
    int count_syllables(const std::string &word);

    // Irregular word pronunciations (exact overrides for the rule engine).
    const std::vector<std::pair<std::string, std::vector<std::string>>> &
    irregular_lexicon();

} // namespace chappie
