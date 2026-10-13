# Software Design

C++17 core components:
- `BrainOS`: runtime state and task-request gate.
- `BootManager`: application startup readiness checks.
- `LawEngine`: default-deny policy rules and confirmation requirements.
- `SafetySupervisor`: command lease, input state and simulation bound checks.
- `RobotModel`: named axes and placeholder position/velocity limits.
- `SimulatedHardware`: bounded low-speed motion simulation.

The core keeps hardware access behind interfaces and deliberately contains no production motor-driver implementation. MCU examples are starter templates, not validated control firmware. The next step is dependency-injected interfaces, structured logging, configuration parsing and exhaustive negative-path tests before adding real-time hardware adapters.
