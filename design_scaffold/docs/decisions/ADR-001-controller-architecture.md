# ADR-001: Controller Architecture

**Status:** Proposed

## Decision
Separate high-level AI and perception from deterministic motion control and independent hardware safety. Use local controllers for bounded peripheral tasks and vendor-supported drive controllers wherever practical.

## Rationale
Language-model/runtime latency and failure modes are incompatible with direct high-rate actuator control. Isolation limits the consequences of software, network and compute failures.

## Consequences
More integration complexity and hardware cost, but clearer fault containment and independently testable interfaces. Exact compute and fieldbus selection depends on component requirements.
