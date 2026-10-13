# AI-CHAPPIE — Complete Design and Source Scaffold

**Project:** AI-CHAPPIE bipedal humanoid robot  
**Version:** 0.2.0 design scaffold  
**Language:** C++17 core; Arduino-compatible firmware templates; JSON configuration; Markdown, Mermaid and PlantUML design sources.

AI-CHAPPIE is organized as a layered robotics system: high-level cognition, policy and authorization, boot/configuration, real-time motion, local controllers, sensors, power telemetry, and independent hardware safety.

> **Scope:** This repository is a development scaffold. It is not a certified humanoid robot, a complete production OS, a BIOS/UEFI replacement, a human-brain emulator, a consciousness implementation, or medical/human-body integration. Do not connect actuator power or test bipedal walking until the complete hardware design, actuator sizing, safety circuit, and risk assessment have been reviewed by qualified engineers.

## Start here
1. `docs/PROJECT_OVERVIEW.md`
2. `docs/SYSTEM_ARCHITECTURE.md`
3. `docs/SAFETY_CASE.md`
4. `docs/HARDWARE_DESIGN.md`
5. `docs/SOFTWARE_DESIGN.md`
6. `docs/BUILD_AND_TEST.md`
7. `docs/IMPLEMENTATION_ROADMAP.md`

## Build the portable C++ core
Requirements: CMake 3.16+ and a C++17 compiler.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Repository map
```text
AI-CHAPPIE/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── REQUIREMENTS.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── SOFTWARE_DESIGN.md
│   ├── HARDWARE_DESIGN.md
│   ├── SAFETY_CASE.md
│   ├── THREAT_MODEL.md
│   ├── DATA_AND_PRIVACY.md
│   ├── TEST_STRATEGY.md
│   ├── BUILD_AND_TEST.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   ├── BOM.md
│   ├── WIRING_AND_BUSES.md
│   ├── BOOT_AND_CONFIGURATION.md
│   ├── API_CONTRACT.md
│   ├── OPERATIONS.md
│   ├── GLOSSARY.md
│   ├── decisions/
│   └── uml/
├── diagrams/
│   ├── system_context.mmd
│   ├── component_architecture.mmd
│   ├── command_sequence.mmd
│   ├── safety_state_machine.mmd
│   └── plantuml/
├── config/
│   ├── robot_profile.example.json
│   ├── law_policy.json
│   ├── command_permissions.json
│   └── schemas/
├── src/
│   ├── brain/
│   ├── boot/
│   ├── safety/
│   ├── model/
│   ├── hardware/
│   └── common/
├── firmware/
│   ├── common/
│   ├── safety_mcu/
│   ├── locomotion_mcu/
│   ├── joint_mcu/
│   ├── hand_mcu/
│   ├── head_neck_mcu/
│   ├── sensor_hub_mcu/
│   └── power_monitor_mcu/
├── tests/
└── scripts/
```

## Design rules
- The LLM proposes intent; it never grants its own permission or directly drives motor PWM.
- Unknown commands, invalid configuration, stale commands, missing safety status, and invalid numeric values fail closed.
- The physical E-stop and appropriate drive safety inputs are independent of the AI computer and network.
- Startup is actuator-disabled. Reboot/reconnect never starts motion automatically.
- The provided “CMOS-style” configuration is a versioned file model, not actual motherboard CMOS/NVRAM.
- Actuator limits in examples are test placeholders, not real-world safe values.
- Jurisdiction-specific legal policy requires review for the actual operating location and use.

## Current implementation status
The portable C++ core has deterministic policy gates, boot self-checks, a small joint model, and a simulated hardware interface. Unit tests exercise core rejection paths. Firmware files are scaffolds only and are not validated against any physical board or drive.
