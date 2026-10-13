# System Architecture

## Layering
1. Human/operator interface and authenticated API.
2. Brain OS: intent interpretation, task coordination and runtime state.
3. Policy layer: authorization, legal/safety rules, risk classification and confirmation.
4. Safety supervisor: command freshness, limits and interlock state; fail closed.
5. Motion host and per-axis or group controllers; these are not yet implemented as hardware drivers.
6. Sensors and state estimation with timestamps and validity flags.
7. Independent drive safety, physical E-stop, power protection and emergency stop path.

## Authority boundaries
The AI layer may propose a task but cannot grant permission to itself, alter hard safety limits, or bypass independent hardware safety. High-rate loops belong on validated real-time controllers, not the language-model process. Loss of heartbeat, stale sensors, invalid configuration or a fault must take the system to an actuator-disabled safe state.

## Example data path
Operator request → authentication/permission → intent/risk classification → policy decision → motion-layer validation → simulated/hardware adapter → feedback and audit. Each layer can reject a command.

The supplied application is a portable C++ scaffold and simulation. It does not implement a production gait controller, real motor drivers, EtherCAT master, certified safety system, or a validated dynamic model.
