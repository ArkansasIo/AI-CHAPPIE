# ADR-003: Versioned Configuration

**Status:** Proposed

## Decision
Use schema-versioned configuration files with safe defaults and validation. Invalid or unknown critical settings block readiness. Keep example actuator enables false.

## Rationale
File-backed configuration is portable and testable. It is not platform CMOS/NVRAM or BIOS/UEFI firmware.

## Consequences
A schema migration and rollback strategy is needed. Safety-critical values should be locked behind a qualified maintenance process, not remote model-controlled edits.
