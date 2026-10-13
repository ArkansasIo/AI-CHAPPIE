# Requirements (Initial)

Functional: bounded command envelope; deterministic policy; trusted authorization distinct from intent; health/fault reporting; configurable axis model; simulated hardware adapter; versioned config/protocol.

Safety/security: disabled actuator outputs at boot/fault; independent E-stop/STO; default-deny on unknown actions and malformed commands; validate lease, finite numbers, bounds, identity/permissions; never check in secrets.

Unresolved: robot dimensions/mass/payload/CoG/workspace/speed/environment/axes/drives/runtime/budget/standards/jurisdiction. Hardware selection must not rely on implicit assumptions.
