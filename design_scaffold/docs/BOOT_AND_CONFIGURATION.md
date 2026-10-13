# Boot and Configuration

Startup state is actuator-disabled. The boot manager checks required configuration fields, policy availability and self-test results before entering safe idle. A failure remains faulted or degraded and requires explicit recovery after diagnosis.

Configuration is represented by versioned JSON files and schema validation. “CMOS-style settings” means a persistent configuration file in this project, not motherboard CMOS/NVRAM or platform firmware. BootManager is application-level initialization; it is not BIOS/UEFI.

Production systems should use atomic updates, schema migration, signed configuration where appropriate, bounded values, audit logging and rollback. Runtime profile changes affecting motion or safety require a maintenance workflow and cannot relax independent hardware safety.
