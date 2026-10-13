# Build and Test

Requirements: CMake 3.16+ and a C++17 compiler.

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/chappie_demo
```

Windows/Visual Studio: `cmake -S . -B build`, `cmake --build build --config Release`, then `ctest --test-dir build -C Release --output-on-failure`.

These commands validate portable software only. They do not compile Arduino sketches, validate wiring, or prove real-time deadlines. Firmware requires the exact board package and vendor libraries.
