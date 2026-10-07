# AI-CHAPPIE

A software CHAPPIE-style robot/AI system — an autonomous agent with a neural-network
brain, persistent memory, a personality engine, computer vision, a voice, and a
virtual robotic body — plus a from-scratch machine-learning and language library.

> **This is an original software project.** It is *not* the film's proprietary
> implementation and contains no material from the film. "CHAPPIE" here refers to
> the *style* of robot/agent being simulated: a curious, learning, semi-autonomous
> machine. All code, audio and data in this repository were written for this project.

---

## What's in here

Two cooperating parts:

| Part | Language | What it is |
|------|----------|------------|
| **`ChappieClone/`** | C++17 | The robot runtime: brain, memory, personality, vision, speech, body, navigation, sensors, security, database, plugins, API |
| **`nn/`, `db/`, `docs/`** | Python 3 | A from-scratch ML + language library: neural networks with autograd, Boolean/quantum logic, coordinates, English grammar & conjugation, and SQL/Excel databases |

The C++ engine is the product; the Python library is the research/teaching layer
that the algorithms were developed and verified against.

---

## ChappieClone — the robot

### Architecture

```
ChappieClone/
├── CMakeLists.txt
├── config/chappie.json          # all tunables, hot-reloadable
├── src/
│   ├── core/                    # math (Vec/Mat), JSON, config, logger
│   ├── ai/                      # NeuralNetwork, Brain, Memory, Personality, DecisionEngine
│   ├── audio/                   # WAV I/O, DSP, phonemizer (G2P), formant TTS
│   ├── vision/                  # frame buffer, edge/blob detection, feature extraction
│   ├── speech/                  # recognition (MFCC + DTW + NN), VAD, wake word
│   ├── robotics/                # virtual body: joints, battery, servos, kinematics
│   ├── navigation/              # A*, Dijkstra, BFS/DFS, occupancy grid
│   ├── sensors/                 # range finder, IMU, temperature, fusion
│   ├── networking/              # local API server, heartbeat, remote control
│   ├── security/                # permissions, autonomy levels, denied zones, audit log
│   ├── database/                # SQLite persistence (memory, episodes, runs, chat)
│   ├── plugins/                 # plugin manager + manifest loading
│   └── api_server.cpp           # JSON control surface
├── tools/voice_demo.cpp         # renders the voice pack
├── tests/                       # assert-based test suite
├── assets/voice/                # generated .wav voice lines
└── database/chappie.db          # generated SQLite database
```

### Core systems

- 🤖 **Autonomous robot AI** — a sense → think → act loop with a real-time tick
- 🧠 **Neural-network brain** — fully-connected net with hand-written backprop,
  gradient clipping, checkpointing (`NeuralNetwork`)
- 💾 **Persistent memory** — episodic (timestamped events), semantic (facts with
  confidence), and associative (LSH approximate recall) (`Memory`)
- 🎭 **Personality & behaviour** — Big-Five style traits, mood model with
  baseline/volatility, and behaviour modulation (`Personality`)
- 👁️ **Computer vision** — frame buffers, Sobel edges, connected-component blob
  detection, feature vectors (`VisionSystem`)
- 🎙️ **Speech recognition** — MFCC features + DTW template matching + NN classifier,
  voice-activity detection, wake-word spotting (`SpeechRecognizer`)
- 🔊 **Text-to-speech** — a real formant synthesizer: grapheme→phoneme rules →
  glottal pulse source → resonant formant filters → prosody. Writes actual `.wav`
  audio (`TtsEngine`)
- 🗣️ **Conversation** — intent matching over the memory store, reply generation
- 🧭 **Navigation** — A*, Dijkstra, BFS/DFS over an occupancy grid (`Navigator`)
- 🦾 **Virtual body** — joints, servos, battery drain, differential drive (`RobotBody`)
- 📡 **Sensors** — range beams, IMU with noise, temperature (`SensorSuite`)
- 🛡️ **Safety & permissions** — autonomy levels, confirmation-required actions,
  denied zones, full audit log (`SecurityManager`)
- 🔌 **Plugins** — manifest-based loading with hooks
- 🌐 **API** — a JSON control surface for external tools / the GUI

### The voice system (fully working)

The TTS is a **source-filter formant synthesizer**, not a wrapper around a system
voice:

1. **G2P** — `audio/phoneme.cpp`: a lexicon of irregular words plus ordered
   letter-to-sound rules (longest-match first), with context adjustments
   (soft `c`/`g`, silent `e`, final `y`), producing ARPAbet-style phonemes.
2. **Source** — a Rosenberg-like glottal pulse train with jitter, vibrato and
   adjustable roughness for a synthetic character.
3. **Filter** — three 2-pole resonators at F1/F2/F3 per vowel, plus shaped noise
   for fricatives and burst+decay for plosives.
4. **Prosody** — pitch declination, question rise, statement fall, stress,
   emotion presets (`happy`, `sad`, `excited`, `curious`).

Voices: `synthetic-young` (default), `deep-calm`, `high-energetic`, `neutral`.

**23 real `.wav` files** are generated into `assets/voice/` (15 scripted lines +
4 voice demos + 4 emotion demos).

---

## Building ChappieClone

Requirements: a C++17 compiler and CMake 3.16+.

```bash
cd ChappieClone
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/chappie            # run the robot
./build/chappie_tests      # run the tests
```

With MinGW / LLVM-MinGW on Windows:

```powershell
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Rendering the voice pack

```powershell
g++ -std=c++17 -O2 -Isrc tools/voice_demo.cpp `
    src/audio/tts.cpp src/audio/phoneme.cpp src/audio/audio.cpp `
    src/core/math.cpp src/core/logger.cpp -o voice_demo.exe
./voice_demo.exe assets/voice
```

---

## The Python library (`nn/`, `db/`)

A dependency-light (NumPy-only) library where every algorithm is written from
first principles and documented mathematically.

| Module | Contents |
|--------|----------|
| `nn/linalg.py` | Linear algebra + numeric gradients, PCA, SVD, stable softmax |
| `nn/activations.py` | Activation functions with analytic derivatives |
| `nn/losses.py` | MSE, MAE, BCE, cross-entropy (fused softmax/sigmoid variants) |
| `nn/layers.py` | Dense, Activation, Dropout, BatchNorm — explicit fwd/bwd |
| `nn/optimizers.py` | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW |
| `nn/model.py` | `Sequential` container, `fit`, `predict`, save/load |
| `nn/autograd.py` | Reverse-mode automatic differentiation (`Tensor`) |
| `nn/gates.py` | Boolean algebra, truth tables, ANF, differentiable logic |
| `nn/qubits.py` | Quantum logic: gates, entanglement, Bell/GHZ, teleportation |
| `nn/coordinates.py` | 3-D signed-coordinate geometry and value fields |
| `nn/grammar.py` | English morphology, tagging, parsing, agreement |
| `nn/conjugation.py` | Full verb paradigm (tense × aspect × mood × voice) |
| `nn/algorithms.py` | Every algorithm in one place, with complexity metadata |
| `db/database.py` | SQLite schema for models, runs, gates, metrics |
| `db/dictionary.py` | Full lexical database (senses, etymology, anagrams) |
| `db/excel.py` | Excel import/export + SQL⇄Excel round-tripping |

### Using the Python library

```python
from nn.model import MLP
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
Y = np.array([[0],[1],[1],[0]], float)          # XOR

net = MLP([2, 16, 16, 1], activations="tanh",
          out_activation="sigmoid", loss="sigmoid_bce")
net.fit(X, Y, epochs=3000, optimizer="adam", lr=0.05)
print(net.predict_classes(X))                    # [0 1 1 0]
```

```python
from db.lexicon import build

with build() as d:
    print(d.anagrams("listen"))     # ['enlist', 'silent', 'tinsel']
    print(d.define("bank")["senses"])
```

### Documentation

`docs/` contains the mathematics behind every module:

`01_theory` · `02_algebra` · `03_calculus` · `04_optimization` · `05_autodiff` ·
`06_architectures` · `07_gate_logic` · `08_database` · `11_dictionary` ·
`12_grammar` · `13_algorithms`

### Running the examples

```bash
python examples/01_xor_manual.py
python examples/06_database.py
python -m tests.test_math
```

---

## Status

| Area | State |
|------|-------|
| C++ core (math, JSON, config, logger) | ✅ builds |
| C++ neural network + backprop | ✅ verified — solves XOR |
| C++ memory (episodic/semantic/associative) | ✅ builds |
| C++ audio (WAV I/O, FFT, MFCC) | ✅ builds |
| C++ phonemizer (G2P) | ✅ builds |
| C++ TTS (formant synthesizer) | ✅ verified — 23 real `.wav` files |
| C++ vision / speech / robotics / nav / sensors / security / db / plugins / API | 🚧 in progress |
| C++ `main.cpp` runtime | 🚧 in progress |
| Python library | ✅ tested |
| Python GUI control room | 🚧 planned |

---

## License

MIT — see `LICENSE`.

---

## Credits

Built as an original educational/engineering project. The name and concept are
inspired by the *style* of a well-known fictional robot; no film assets, code, or
proprietary material are included.
