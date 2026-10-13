# Requirements (Initial)

## Functional
- Provide a bounded command envelope and deterministic policy result.
- Keep intent planning separate from trusted authorization.
- Provide health reporting and latched fault state.
- Model a configurable robot axis list and simulated hardware adapter.
- Version persistent configuration and protocol messages.
- Record decisions and support repeatable tests.

## Safety and security
- Actuator outputs disabled at startup and after any fault.
- E-stop and safety enable independent of AI host/network.
- Default-deny for unknown actions, unknown permissions and malformed commands.
- Validate command lease, numeric finiteness, bounds, authentication and authorization.
- No secrets in checked-in configurations.

## Unresolved product inputs
Robot dimensions, total mass, payload, center of gravity, workspace, operating speed, environment, exact axes and drives, target runtime, compute budget, selected safety standard and target jurisdiction remain unconfirmed. Hardware design must not proceed on implicit assumptions.
