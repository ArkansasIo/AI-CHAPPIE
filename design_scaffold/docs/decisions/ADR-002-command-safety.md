# ADR-002: Command Authorization and Safety

**Status:** Proposed

## Decision
Use default-deny policy; authentication and authorization are separate from intent; require explicit confirmation for ambiguous or high-risk operations; revalidate commands at the motion boundary. Hardware E-stop/STO must remain independent and non-overridable by software policy.

## Rationale
Natural-language text or model output is not sufficient evidence of permission, intent or safe physical conditions.

## Consequences
Some actions require an additional operator workflow. This is intentional. Policies must be reviewed and adapted for actual use and jurisdiction.
