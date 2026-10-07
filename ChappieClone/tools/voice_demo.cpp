// Render the Chappie voice pack + a few demo lines.
#include <cstdio>
#include <string>
#include "audio/tts.hpp"
#include "core/logger.hpp"

using namespace chappie;

int main(int argc, char **argv)
{
    Logger::instance().configure(LogLevel::Info, true, "");

    const std::string out = (argc > 1) ? argv[1] : "assets/voice";

    // 1. Built-in voice pack (15 lines).
    const int n = render_voice_pack(out, VoiceProfile::synthetic_young());
    std::printf("\nRendered %d voice lines into '%s'\n", n, out.c_str());

    // 2. Demonstrate the voice profiles on one sentence.
    const std::string demo = "Hello. I am Chappie. I am alive.";
    TtsEngine tts;
    for (const auto &name : TtsEngine::available_voices())
    {
        tts.set_voice(name);
        SpeakParams p;
        p.sample_rate = 22050;
        const std::string path = out + "/demo_" + name + ".wav";
        std::string err;
        if (tts.speak_to_file(demo, path, p, err))
            std::printf("  voice '%s' -> %s\n", name.c_str(), path.c_str());
        else
            std::printf("  voice '%s' FAILED: %s\n", name.c_str(), err.c_str());
    }

    // 3. Emotion demo.
    const std::string line = "I found something amazing!";
    for (const std::string emo : {"happy", "sad", "excited", "curious"})
    {
        SpeakParams p = TtsEngine::emotion_params(emo, line);
        p.sample_rate = 22050;
        const std::string path = out + "/emotion_" + emo + ".wav";
        std::string err;
        if (tts.speak_to_file(line, path, p, err))
            std::printf("  emotion '%s' -> %s\n", emo.c_str(), path.c_str());
    }

    // 4. Phoneme breakdown for a couple of words.
    for (const std::string w : {"chappie", "robot", "hello", "thinking"})
    {
        const auto ph = word_to_phonemes(w);
        std::printf("  %-10s -> %s\n", w.c_str(),
                    phonemes_to_string(ph).c_str());
    }
    return 0;
}
