# Build and Test

Requirements: CMake 3.16 or newer and a C++17 compiler.

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/chappie_demo
```

On Windows with Visual Studio, use `cmake -S . -B build` then `cmake --build build --config Release` and `ctest --test-dir build -C Release --output-on-failure`.

These commands validate the portable software only. They do not compile the Arduino sketches, validate electrical design, or prove real-time deadlines. Firmware requires the exact selected board package and vendor libraries.
