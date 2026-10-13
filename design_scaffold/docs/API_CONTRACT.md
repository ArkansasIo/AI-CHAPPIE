# API Contract (Proposed)

This is a design contract, not a deployed network server.

## Request envelope
`command_id`, `action`, `requester_id`, `permission`, monotonic expiry/lease, declared risk, authentication result, authorization result and any required confirmation token.

## Response
Decision (`ALLOW`, `DENY`, `REQUIRE_CONFIRMATION`), stable rule identifier, human-readable reason and audit/correlation ID.

## Rules
- Deny unknown message versions, actions and permissions.
- Validate length, encoding, integer overflow, finite floats, command age, sequence and replay protection.
- Authentication and authorization must come from trusted code and credentials, never from user text or a model-generated claim.
- Motion controller independently validates command freshness and physical limits.
- No API endpoint should enable motion by default. Emergency-stop handling must remain local and independent of network availability.
