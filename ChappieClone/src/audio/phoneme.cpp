// ---------------------------------------------------------------------------
// audio/phoneme.cpp -- phoneme inventory and grapheme-to-phoneme rules.
//
// Formant data is based on published average adult-male vowel formants; the
// synthesizer shifts them for the "synthetic-young" voice profile.
// ---------------------------------------------------------------------------
#include "audio/phoneme.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>

namespace chappie
{

    namespace
    {

        Phoneme make(const std::string &sym, PhonemeType type, double dur, double f1,
                     double f2, double f3, bool voiced, double amp, double noise)
        {
            Phoneme p;
            p.symbol = sym;
            p.type = type;
            p.duration_ms = dur;
            p.f1 = f1;
            p.f2 = f2;
            p.f3 = f3;
            p.voiced = voiced;
            p.amplitude = amp;
            p.noise = noise;
            return p;
        }

        std::vector<Phoneme> build_inventory()
        {
            using PT = PhonemeType;
            std::vector<Phoneme> v;

            // --- vowels (F1, F2, F3 in Hz, average male) -------------------------
            v.push_back(make("IY", PT::Vowel, 110, 270, 2290, 3010, true, 1.0, 0));  // see
            v.push_back(make("IH", PT::Vowel, 85, 400, 1990, 2550, true, 0.9, 0));   // sit
            v.push_back(make("EY", PT::Vowel, 130, 450, 2100, 2700, true, 1.0, 0));  // say
            v.push_back(make("EH", PT::Vowel, 95, 530, 1840, 2480, true, 0.95, 0));  // bed
            v.push_back(make("AE", PT::Vowel, 120, 660, 1720, 2410, true, 1.0, 0));  // cat
            v.push_back(make("AA", PT::Vowel, 130, 730, 1090, 2440, true, 1.0, 0));  // father
            v.push_back(make("AO", PT::Vowel, 130, 570, 840, 2410, true, 0.95, 0));  // thought
            v.push_back(make("OW", PT::Vowel, 135, 500, 900, 2400, true, 1.0, 0));   // go
            v.push_back(make("UH", PT::Vowel, 90, 440, 1020, 2240, true, 0.9, 0));   // book
            v.push_back(make("UW", PT::Vowel, 120, 300, 870, 2240, true, 0.95, 0));  // boot
            v.push_back(make("AH", PT::Vowel, 90, 640, 1190, 2390, true, 0.9, 0));   // but
            v.push_back(make("ER", PT::Vowel, 120, 490, 1350, 1690, true, 0.95, 0)); // bird
            v.push_back(make("AY", PT::Vowel, 150, 660, 1700, 2400, true, 1.0, 0));  // my
            v.push_back(make("AW", PT::Vowel, 150, 640, 1230, 2400, true, 1.0, 0));  // cow
            v.push_back(make("OY", PT::Vowel, 150, 570, 840, 2400, true, 1.0, 0));   // boy

            // --- plosives (short burst, voicing for b/d/g) -----------------------
            v.push_back(make("P", PT::Plosive, 70, 0, 0, 0, false, 0.6, 1.0));
            v.push_back(make("B", PT::Plosive, 60, 0, 0, 0, true, 0.7, 0.7));
            v.push_back(make("T", PT::Plosive, 65, 0, 0, 0, false, 0.6, 1.0));
            v.push_back(make("D", PT::Plosive, 55, 0, 0, 0, true, 0.7, 0.7));
            v.push_back(make("K", PT::Plosive, 70, 0, 0, 0, false, 0.6, 1.0));
            v.push_back(make("G", PT::Plosive, 60, 0, 0, 0, true, 0.7, 0.7));

            // --- fricatives (mostly noise) ---------------------------------------
            v.push_back(make("F", PT::Fricative, 90, 0, 0, 0, false, 0.4, 1.0));
            v.push_back(make("V", PT::Fricative, 70, 0, 0, 0, true, 0.5, 0.6));
            v.push_back(make("TH", PT::Fricative, 85, 0, 0, 0, false, 0.4, 1.0));
            v.push_back(make("DH", PT::Fricative, 65, 0, 0, 0, true, 0.5, 0.6));
            v.push_back(make("S", PT::Fricative, 100, 0, 0, 0, false, 0.5, 1.0));
            v.push_back(make("Z", PT::Fricative, 80, 0, 0, 0, true, 0.6, 0.7));
            v.push_back(make("SH", PT::Fricative, 105, 0, 0, 0, false, 0.55, 1.0));
            v.push_back(make("ZH", PT::Fricative, 85, 0, 0, 0, true, 0.6, 0.8));
            v.push_back(make("HH", PT::Fricative, 60, 0, 0, 0, false, 0.35, 0.9));

            // --- affricates -------------------------------------------------------
            v.push_back(make("CH", PT::Affricate, 110, 0, 0, 0, false, 0.6, 1.0));
            v.push_back(make("JH", PT::Affricate, 95, 0, 0, 0, true, 0.65, 0.8));

            // --- nasals -----------------------------------------------------------
            v.push_back(make("M", PT::Nasal, 75, 250, 1100, 2200, true, 0.8, 0));
            v.push_back(make("N", PT::Nasal, 70, 250, 1700, 2600, true, 0.8, 0));
            v.push_back(make("NG", PT::Nasal, 80, 250, 2300, 2750, true, 0.8, 0));

            // --- liquids / glides -------------------------------------------------
            v.push_back(make("L", PT::Liquid, 70, 360, 1300, 2700, true, 0.8, 0));
            v.push_back(make("R", PT::Liquid, 75, 310, 1060, 1380, true, 0.8, 0));
            v.push_back(make("W", PT::Glide, 65, 300, 610, 2200, true, 0.8, 0));
            v.push_back(make("Y", PT::Glide, 60, 260, 2070, 3020, true, 0.8, 0));

            // --- silence ----------------------------------------------------------
            v.push_back(make("_", PT::Silence, 120, 0, 0, 0, false, 0.0, 0));

            return v;
        }

        // Ordered letter-to-sound rules: longest patterns first so "ough" is tried
        // before "ou" before "o".
        struct Rule
        {
            std::string pattern; // lowercase letter sequence
            std::vector<std::string> phonemes;
        };

        const std::vector<Rule> &rules()
        {
            static const std::vector<Rule> r = {
                // 4+ letter patterns
                {"ough", {"AO", "F"}}, // rough/tough family (approx)
                {"tion", {"SH", "AH", "N"}},
                {"sion", {"ZH", "AH", "N"}},
                {"ture", {"CH", "ER"}},
                {"ough", {"OW"}},
                {"eigh", {"EY"}},
                // 3-letter patterns
                {"tch", {"CH"}},
                {"dge", {"JH"}},
                {"igh", {"AY"}},
                {"ing", {"IH", "NG"}},
                {"air", {"EH", "R"}},
                {"ear", {"IY", "R"}},
                {"eer", {"IY", "R"}},
                {"our", {"AO", "R"}},
                {"oor", {"AO", "R"}},
                {"all", {"AO", "L"}},
                {"alk", {"AO", "K"}},
                {"old", {"OW", "L", "D"}},
                {"ind", {"AY", "N", "D"}},
                {"ild", {"AY", "L", "D"}},
                {"sch", {"S", "K"}},
                {"shr", {"SH", "R"}},
                {"spl", {"S", "P", "L"}},
                {"str", {"S", "T", "R"}},
                {"thr", {"TH", "R"}},
                // 2-letter consonant digraphs
                {"ch", {"CH"}},
                {"sh", {"SH"}},
                {"th", {"TH"}},
                {"ph", {"F"}},
                {"wh", {"W"}},
                {"ck", {"K"}},
                {"ng", {"NG"}},
                {"qu", {"K", "W"}},
                {"gh", {"_"}}, // often silent (night)
                {"kn", {"N"}}, // silent k
                {"wr", {"R"}}, // silent w
                {"mb", {"M"}}, // silent b (climb)
                // 2-letter vowel digraphs
                {"ee", {"IY"}},
                {"ea", {"IY"}},
                {"ai", {"EY"}},
                {"ay", {"EY"}},
                {"oa", {"OW"}},
                {"oo", {"UW"}},
                {"ou", {"AW"}},
                {"ow", {"AW"}},
                {"oi", {"OY"}},
                {"oy", {"OY"}},
                {"au", {"AO"}},
                {"aw", {"AO"}},
                {"ie", {"IY"}},
                {"ue", {"UW"}},
                {"ui", {"UW"}},
                {"ey", {"EY"}},
                // r-controlled vowels
                {"ar", {"AA", "R"}},
                {"er", {"ER"}},
                {"ir", {"ER"}},
                {"ur", {"ER"}},
                {"or", {"AO", "R"}},
                // single consonants
                {"b", {"B"}},
                {"c", {"K"}},
                {"d", {"D"}},
                {"f", {"F"}},
                {"g", {"G"}},
                {"h", {"HH"}},
                {"j", {"JH"}},
                {"k", {"K"}},
                {"l", {"L"}},
                {"m", {"M"}},
                {"n", {"N"}},
                {"p", {"P"}},
                {"q", {"K"}},
                {"r", {"R"}},
                {"s", {"S"}},
                {"t", {"T"}},
                {"v", {"V"}},
                {"w", {"W"}},
                {"x", {"K", "S"}},
                {"y", {"Y"}},
                {"z", {"Z"}},
                // single vowels (default values)
                {"a", {"AE"}},
                {"e", {"EH"}},
                {"i", {"IH"}},
                {"o", {"AA"}},
                {"u", {"AH"}},
            };
            return r;
        }

        const std::map<std::string, std::vector<std::string>> &irregular()
        {
            static const std::map<std::string, std::vector<std::string>> table = {
                {"a", {"AH"}},
                {"an", {"AE", "N"}},
                {"the", {"DH", "AH"}},
                {"to", {"T", "UW"}},
                {"two", {"T", "UW"}},
                {"too", {"T", "UW"}},
                {"do", {"D", "UW"}},
                {"does", {"", "AH", "Z"}},
                {"of", {"AH", "V"}},
                {"off", {"AO", "F"}},
                {"one", {"W", "AH", "N"}},
                {"once", {"W", "AH", "N", "S"}},
                {"said", {"S", "EH", "D"}},
                {"says", {"S", "EH", "Z"}},
                {"are", {"AA", "R"}},
                {"was", {"W", "AH", "Z"}},
                {"were", {"W", "ER"}},
                {"have", {"HH", "AE", "V"}},
                {"has", {"HH", "AE", "Z"}},
                {"who", {"HH", "UW"}},
                {"what", {"W", "AH", "T"}},
                {"where", {"W", "EH", "R"}},
                {"when", {"W", "EH", "N"}},
                {"why", {"W", "AY"}},
                {"how", {"HH", "AW"}},
                {"you", {"Y", "UW"}},
                {"your", {"Y", "AO", "R"}},
                {"my", {"M", "AY"}},
                {"i", {"AY"}},
                {"he", {"HH", "IY"}},
                {"she", {"SH", "IY"}},
                {"we", {"W", "IY"}},
                {"me", {"M", "IY"}},
                {"be", {"B", "IY"}},
                {"see", {"S", "IY"}},
                {"here", {"HH", "IY", "R"}},
                {"there", {"DH", "EH", "R"}},
                {"their", {"DH", "EH", "R"}},
                {"they", {"DH", "EY"}},
                {"them", {"DH", "EH", "M"}},
                {"this", {"DH", "IH", "S"}},
                {"that", {"DH", "AE", "T"}},
                {"these", {"DH", "IY", "Z"}},
                {"those", {"DH", "OW", "Z"}},
                {"think", {"TH", "IH", "NG", "K"}},
                {"know", {"N", "OW"}},
                {"no", {"N", "OW"}},
                {"go", {"G", "OW"}},
                {"so", {"S", "OW"}},
                {"come", {"K", "AH", "M"}},
                {"some", {"S", "AH", "M"}},
                {"done", {"D", "AH", "N"}},
                {"love", {"L", "AH", "V"}},
                {"move", {"M", "UW", "V"}},
                {"prove", {"P", "R", "UW", "V"}},
                {"great", {"G", "R", "EY", "T"}},
                {"break", {"B", "R", "EY", "K"}},
                {"friend", {"F", "R", "EH", "N", "D"}},
                {"again", {"AH", "G", "EH", "N"}},
                {"against", {"AH", "G", "EH", "N", "S", "T"}},
                {"many", {"M", "EH", "N", "IY"}},
                {"any", {"EH", "N", "IY"}},
                {"woman", {"W", "UH", "M", "AH", "N"}},
                {"women", {"W", "IH", "M", "AH", "N"}},
                {"chappie", {"CH", "AE", "P", "IY"}},
                {"chappi", {"CH", "AE", "P", "IY"}},
                {"hello", {"HH", "EH", "L", "OW"}},
                {"robot", {"R", "OW", "B", "AA", "T"}},
                {"name", {"N", "EY", "M"}},
                {"please", {"P", "L", "IY", "Z"}},
                {"sorry", {"S", "AA", "R", "IY"}},
                {"yes", {"Y", "EH", "S"}},
                {"okay", {"OW", "K", "EY"}},
                {"ok", {"OW", "K", "EY"}},
            };
            return table;
        }

        std::string lower(const std::string &s)
        {
            std::string out = s;
            std::transform(out.begin(), out.end(), out.begin(),
                           [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            return out;
        }

        bool is_vowel_letter(char c)
        {
            const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return l == 'a' || l == 'e' || l == 'i' || l == 'o' || l == 'u';
        }

    } // namespace

    // ===========================================================================
    // Inventory access
    // ===========================================================================
    const std::vector<Phoneme> &phoneme_inventory()
    {
        static const std::vector<Phoneme> inv = build_inventory();
        return inv;
    }

    const Phoneme *find_phoneme(const std::string &symbol)
    {
        for (const auto &p : phoneme_inventory())
            if (p.symbol == symbol)
                return &p;
        return nullptr;
    }

    const std::vector<std::pair<std::string, std::vector<std::string>>> &
    irregular_lexicon()
    {
        static std::vector<std::pair<std::string, std::vector<std::string>>> flat;
        if (flat.empty())
            for (const auto &kv : irregular())
                flat.emplace_back(kv.first, kv.second);
        return flat;
    }

    // ===========================================================================
    // Syllables and stress
    // ===========================================================================
    int count_syllables(const std::string &word)
    {
        const std::string w = lower(word);
        if (w.empty())
            return 1;

        int count = 0;
        bool prev_vowel = false;
        for (std::size_t i = 0; i < w.size(); ++i)
        {
            const bool v = is_vowel_letter(w[i]);
            if (v && !prev_vowel)
                ++count;
            // A silent trailing 'e' does not form its own syllable.
            if (v && i == w.size() - 1 && w[i] == 'e' && count > 1)
                --count;
            prev_vowel = v;
        }
        if (count == 0)
            count = 1;
        return count;
    }

    int primary_stress_syllable(const std::string &word)
    {
        const int n = count_syllables(word);
        if (n <= 1)
            return 0;
        // English heuristic: nouns/adjectives stress the first syllable,
        // longer words often stress the antepenult. Approximate with n/3.
        if (n == 2)
            return 0;
        if (n == 3)
            return 0;
        return std::max(0, n / 3);
    }

    // ===========================================================================
    // Grapheme-to-phoneme
    // ===========================================================================
    namespace
    {

        std::vector<std::string> apply_rules(const std::string &word)
        {
            std::vector<std::string> out;
            const std::string w = lower(word);
            std::size_t i = 0;

            while (i < w.size())
            {
                // Only letters get rules applied; anything else becomes a pause.
                if (!std::isalpha(static_cast<unsigned char>(w[i])))
                {
                    ++i;
                    continue;
                }

                bool matched = false;
                // Try the longest pattern first (up to 4 letters).
                for (int len = 4; len >= 1 && !matched; --len)
                {
                    if (i + static_cast<std::size_t>(len) > w.size())
                        continue;
                    const std::string chunk = w.substr(i, static_cast<std::size_t>(len));
                    for (const auto &rule : rules())
                    {
                        if (rule.pattern != chunk)
                            continue;

                        // Context tweaks for single letters.
                        std::vector<std::string> ph = rule.phonemes;

                        if (chunk == "c" && i + 1 < w.size() &&
                            (w[i + 1] == 'e' || w[i + 1] == 'i' || w[i + 1] == 'y'))
                        {
                            ph = {"S"}; // soft c
                        }
                        else if (chunk == "g" && i + 1 < w.size() &&
                                 (w[i + 1] == 'e' || w[i + 1] == 'i' ||
                                  w[i + 1] == 'y'))
                        {
                            ph = {"JH"}; // soft g
                        }
                        else if (chunk == "s" && i + 1 < w.size() &&
                                 (w[i + 1] == 'i' || w[i + 1] == 'e'))
                        {
                            // 'si'/'se' often /ZH/ (vision, measure) -- approximate /SH/
                            ph = {"SH"};
                        }
                        else if (chunk == "y" && i == 0)
                        {
                            ph = {"Y"}; // initial y is a glide
                        }
                        else if (chunk == "y" && i == w.size() - 1)
                        {
                            ph = {"IY"}; // final y is a vowel (happy)
                        }
                        else if (chunk == "e" && i == w.size() - 1 && w.size() > 2)
                        {
                            ph = {}; // silent final e
                        }

                        out.insert(out.end(), ph.begin(), ph.end());
                        i += static_cast<std::size_t>(len);
                        matched = true;
                        break;
                    }
                }
                if (!matched)
                    ++i;
            }
            return out;
        }

    } // namespace

    std::vector<Phoneme> word_to_phonemes(const std::string &word)
    {
        std::vector<Phoneme> result;
        const std::string w = lower(word);

        // Strip surrounding punctuation but remember it as a pause.
        std::string core;
        bool trailing_pause = false;
        bool strong_pause = false;
        for (char c : w)
        {
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                core += c;
            }
            else if (!core.empty())
            {
                trailing_pause = true;
                if (c == '.' || c == '!' || c == '?' || c == ',' || c == ';' ||
                    c == ':')
                    strong_pause = true;
            }
        }
        if (core.empty())
        {
            if (const Phoneme *sil = find_phoneme("_"))
            {
                Phoneme p = *sil;
                p.duration_ms = strong_pause ? 250.0 : 120.0;
                result.push_back(p);
            }
            return result;
        }

        // 1. exact lexicon lookup
        std::vector<std::string> symbols;
        auto it = irregular().find(core);
        if (it != irregular().end())
        {
            symbols = it->second;
        }
        else
        {
            // 2. rule-based conversion
            symbols = apply_rules(core);
        }

        // 3. map symbols to Phoneme structs
        for (const auto &s : symbols)
        {
            if (const Phoneme *p = find_phoneme(s))
                result.push_back(*p);
        }

        if (trailing_pause)
        {
            if (const Phoneme *sil = find_phoneme("_"))
            {
                Phoneme p = *sil;
                p.duration_ms = strong_pause ? 220.0 : 90.0;
                result.push_back(p);
            }
        }
        return result;
    }

    std::vector<Phoneme> text_to_phonemes(const std::string &text)
    {
        std::vector<Phoneme> out;
        std::istringstream iss(text);
        std::string token;
        bool first = true;

        while (iss >> token)
        {
            if (!first)
            {
                // Inter-word gap.
                if (const Phoneme *sil = find_phoneme("_"))
                {
                    Phoneme p = *sil;
                    p.duration_ms = 60.0;
                    out.push_back(p);
                }
            }
            const auto word = word_to_phonemes(token);
            out.insert(out.end(), word.begin(), word.end());
            first = false;
        }
        return out;
    }

    std::string phonemes_to_string(const std::vector<Phoneme> &phonemes)
    {
        std::ostringstream os;
        for (std::size_t i = 0; i < phonemes.size(); ++i)
        {
            if (phonemes[i].is_silence())
                continue;
            if (i)
                os << " ";
            os << phonemes[i].symbol;
        }
        return os.str();
    }

} // namespace chappie
