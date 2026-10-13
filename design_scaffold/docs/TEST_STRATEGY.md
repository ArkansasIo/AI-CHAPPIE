# Test Strategy

## Software tests
- Policy defaults to deny and rejects unauthenticated, unauthorized, ambiguous and high-risk commands.
- Configuration parsing validates schema version, types, bounds and required keys.
- Fault paths test missing E-stop status, lost heartbeat, expired commands, invalid floating-point values and stale sensor data.
- Simulation tests exercise startup disabled, bounded motion requests and stop behavior.
- MCU protocol tests must cover byte order, length bounds, CRC, sequence rollover, replay, timeout and malformed messages.

## Hardware-in-the-loop sequence
1. Test power rails and grounds with actuators disconnected.
2. Verify E-stop and drive STO/enable behavior independently of the AI computer.
3. Validate each drive with a current-limited bench supply and no load.
4. Add one axis at a time with mechanical stops, guarding and a reviewed risk assessment.
5. Conduct tethered, constrained, low-energy motion tests before any balance or walking test.

Never use this scaffold as evidence of safety certification. Every physical configuration requires its own validation and sign-off.
