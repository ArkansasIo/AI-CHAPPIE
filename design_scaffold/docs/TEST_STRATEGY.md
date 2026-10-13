# Test Strategy

## Software
- Policy defaults to deny and rejects unauthenticated/unauthorized/high-risk/ambiguous commands.
- Validate configuration schema, types, bounds and required keys.
- Inject missing E-stop status, lost heartbeat, expired commands, invalid floats and stale data.
- Verify actuator-disabled startup and stop behavior in simulation.
- Protocol tests should cover byte order, length, CRC, sequence rollover, replay, timeout and malformed messages.

## Hardware-in-the-loop
1. Test rails and grounds with actuators disconnected.
2. Verify E-stop and drive STO/enable independently of AI/network.
3. Validate each drive on a current-limited bench setup without load.
4. Add axes incrementally with guarding and a reviewed risk assessment.
5. Do tethered, constrained low-energy tests before balance/walking trials.

This scaffold is not a certification or safety claim. Every physical configuration needs specific validation and qualified sign-off.
