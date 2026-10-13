# Wiring and Buses (Planning Only)

Potential links include Ethernet for the AI host, a deterministic real-time fieldbus if supported by selected drives, CAN/CAN-FD for suitable local controllers, and UART/SPI/I²C for board-local peripherals. Protocol and topology depend on selected hardware; no bus is implicitly supported by the current source.

Document pinout, voltage level, connector, pin numbering, cable type, shielding, termination, bandwidth, update period, timeout, CRC, sequence counter and fault response for every link. Separate high-current motor and battery wiring from sensitive signal wiring. Validate grounds, isolation and EMC. Do not use pin assignments from these conceptual notes as a wiring harness plan.
