# Threat Model

Assets: safe motion, operator safety, configuration integrity, identity/authorization data, firmware and audit history.

Threats include prompt injection, forged API requests, stolen credentials, replayed/stale commands, malformed numeric values, sensor spoofing, bus faults, update tampering, denial of service and accidental unsafe configuration.

Mitigations: least privilege, independent authorization, bounded message parsers, authenticated transport, nonces/sequence numbers, timestamps/leases, secure boot/update, signed artifacts, fail-closed inputs, rate limiting, physical interlocks and tamper-evident audit logs.

The current scaffold implements only a small subset of these controls. Network authentication, secure update, cryptographic message authentication, production audit storage and sensor spoofing detection are not implemented.
