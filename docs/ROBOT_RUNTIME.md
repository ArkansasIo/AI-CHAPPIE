# Robot Runtime Simulator

The C++ runtime is a hardware-independent simulation interface. It is intended for development and testing—not direct control of a physical robot.

## Build

```sh
cmake -S ChappieClone -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/chappie
```

On Windows with Visual Studio, append `--config Release` to build and `-C Release` to CTest.

## CLI command summary

The command-line application supports:

| Command | Purpose |
|---|---|
| `status` | Show simulator state |
| `drive <value>` | Request normalized forward/reverse motion |
| `turn <value>` | Request normalized turning |
| `stop` | Request zero motion |
| `tick <seconds>` | Advance simulated time |
| `remember <key> <value>` | Store a value in the in-process memory |
| `recall <key>` | Retrieve a stored value |
| `quit` | Exit the simulator |

Motion values are normalized to [-1, 1]. Use the CLI only in the simulator. Hardware integration requires a separately reviewed device adapter and its own safety case.

## Safety boundary

- The runtime is not a certified robot safety controller.
- Do not connect its output directly to real motors, actuators, GPIO, or a robot arm.
- Physical systems need independent emergency-stop circuitry, watchdogs, safe torque/power removal, validated sensors, bounded command timeouts, and risk assessment.
- The in-memory store is volatile and is not suitable for safety-critical state.
- Test every adapter in a disconnected or hardware-in-the-loop environment before considering physical deployment.

## Troubleshooting

If CMake cannot find a compiler, install a C++17 toolchain and CMake 3.16 or newer. If CTest reports no tests, confirm that configuration completed successfully and that the test executable built.
