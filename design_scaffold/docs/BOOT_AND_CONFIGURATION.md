# Boot and Configuration

Startup is actuator-disabled. Application-level boot checks confirm configuration, policy availability and self-tests before entering safe idle. Critical failures leave the runtime faulted/degraded pending diagnosis.

Settings use versioned JSON and schema validation. “CMOS-style settings” means file-backed application configuration, not motherboard CMOS/NVRAM. BootManager is app initialization, not BIOS/UEFI.

Production updates require atomic writes, migration/rollback strategy, bounded values, audit logging and authorized maintenance. Runtime profile edits must never relax independent hardware safety.
