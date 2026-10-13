# Preliminary Safety Case — Not Certification

Claim: the scaffold is designed to reject invalid, unauthorized, stale or high-risk commands and remain actuator-disabled by default.

Design evidence: separate policy and motion validation; simulated checks for E-stop, heartbeat, interlocks, command expiry and finite values; MCU templates hold enable outputs LOW; unit tests exercise a subset of rejection paths.

Limitations: no certified hardware, validated braking distance, SIL/PL assessment, complete FMEA/FTA, formal verification, EMC tests, gait/fall tests, or independent review. The physical E-stop/STO design and hazard analysis must be qualified for the actual assembly. Do not enable load-bearing actuators based only on this repository.
