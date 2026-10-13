# AI-CHAPPIE

A software robot/AI research project with a C++17 robot runtime and a Python
machine-learning and language library. This is an original implementation; it
does not contain proprietary film code or assets.

## Layout

- `ChappieClone/`: C++ math/JSON/config/logging, neural network, memory, audio
  synthesis, and the hardware-independent robot runtime.
- `nn/`: NumPy-based neural networks, automatic differentiation, logic,
  coordinates, grammar, and optimization.
- `db/`: SQLite and lexical/database helpers.
- `docs/`, `examples/`, `tests/`: technical notes, examples, and tests.

## Build the C++ runtime

Requirements: CMake 3.16+ and a C++17 compiler.

```sh
cmake -S ChappieClone -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/chappie
```

For Visual Studio's multi-configuration generator on Windows, use
`cmake --build build --config Release` and
`ctest --test-dir build -C Release --output-on-failure`.

## Python library

```sh
python -m pip install -r requirements.txt
python -m tests.test_math
python examples/01_xor_manual.py
```

## Runtime simulator

The CLI supports `status`, `drive <value>`, `turn <value>`, `stop`,
`tick <seconds>`, `remember <key> <value>`, `recall <key>`, and `quit`.
Motion values are normalized to [-1, 1]. Motion commands require explicit
confirmation in the runtime API; forward motion is stopped when an obstacle is
within 0.25 m. See [Robot Runtime documentation](docs/ROBOT_RUNTIME.md).

## Safety

This is a simulator, not a certified safety controller. It does not directly
drive real motors, robot arms, or GPIO. Physical deployment requires validated
device adapters, command timeouts, watchdogs, independent emergency-stop wiring,
and safety interlocks. The in-process memory store is not durable storage.

## License

MIT. See [LICENSE](LICENSE).
