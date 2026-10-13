# System Architecture

## Layers
1. Authenticated operator UI/API.
2. Brain OS for task coordination and runtime state.
3. Policy, permissions, risk classification and confirmation.
4. Motion safety checks for command freshness, limits and interlocks.
5. Real-time motion host and per-axis controllers.
6. Sensors/state estimation with timestamps and validity flags.
7. Independent hardware E-stop, drive STO/enable and power protection.

AI may propose intent but cannot authorize itself, change hard safety limits or bypass independent safety. High-rate loops belong on validated real-time controllers, not the language-model process. Loss of heartbeat, stale sensors, invalid config or any fault must lead to actuator-disabled state.

Command path: operator request → trusted authentication/authorization → intent/risk classification → policy → motion-layer validation → simulated/hardware adapter → feedback/audit. Each layer can reject commands. Current code is a software scaffold; no production gait controller, motor driver, EtherCAT master or certified safety system is included.
