# Security Policy

Report vulnerabilities privately to the project maintainers before public disclosure.

## Security boundaries
- Authentication and authorization are separate from natural-language intent.
- The AI process must not hold unrestricted actuator or safety-controller credentials.
- Use signed firmware, secure update procedures, least-privilege service accounts, and tamper-evident logs for production.
- Never store passwords, tokens, private keys, or certificates in example JSON configuration.
- Validate message version, length, CRC, sequence, timestamp/lease, numeric finiteness and target limits.
- Treat camera, microphone, network, and operator text as untrusted inputs.
- A software watchdog supplements but does not replace a physical E-stop or safety-rated circuit.
