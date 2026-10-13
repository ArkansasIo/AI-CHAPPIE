# API Contract (Proposed)

Design contract only; no production network API is deployed.

Request envelope: `command_id`, `action`, `requester_id`, `permission`, expiry/lease, declared risk, trusted authentication/authorization results and any separate confirmation token.

Response: `ALLOW`, `DENY` or `REQUIRE_CONFIRMATION`, stable rule ID, reason and correlation ID.

Rules: deny unknown versions/actions/permissions; validate size, encoding, overflow, finite floats, freshness, sequence and replay; trust authentication only from trusted credential-handling code; revalidate at motion boundary. The network API must never be the sole emergency-stop path, and no endpoint should enable motion by default.
