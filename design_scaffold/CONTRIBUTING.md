# Contributing

1. Open an issue describing the change and its safety impact.
2. Keep policy and hardware boundaries explicit.
3. Add or update tests for behavior changes.
4. Run CMake build and CTest before submitting.
5. Update documentation and diagrams when architecture or interfaces change.
6. Never include secrets, private keys, personal data, or production credentials.
7. Hardware-facing changes require a test plan and review by a qualified robotics/electrical engineer.

## C++ conventions
- C++17, RAII, explicit types, no hidden actuator side effects in constructors.
- Fail closed on unknown or invalid inputs.
- Keep hardware interfaces behind abstractions so tests can run without physical devices.
