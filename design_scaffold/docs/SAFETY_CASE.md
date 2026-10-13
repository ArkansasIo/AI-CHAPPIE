# Safety Case (Preliminary, Not Certification)

## Claim
The software scaffold is designed to reject obviously invalid, unauthorized, stale or high-risk commands and to default to actuator-disabled behavior.

## Supporting design evidence
- High-level command policy is separate from motion validation.
- E-stop, heartbeat, interlock, time lease and finite numeric values are checked in the simulated safety supervisor.
- MCU examples keep enable outputs LOW.
- Tests cover a subset of policy and simulation rejection paths.

## Limitations and required evidence
This is not a complete safety case for a physical robot. No certified hardware, safety integrity level, performance level, validated braking distance, FMEA/FTA, formal verification, EMC testing, gait/fall test, or independent review is supplied. Physical E-stop/STO wiring and hazard analysis must be designed and approved for the actual assembly. Do not enable load-bearing actuators based solely on this repository.
