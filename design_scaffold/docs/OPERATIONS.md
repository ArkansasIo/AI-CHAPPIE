# Operations

## Before power-on
Inspect frame fasteners, joint couplings, cable routing, guards, battery retention, fuses, grounding and E-stop access. Ensure the workspace is clear and a trained operator is present.

## Startup
Start with actuator power isolated. Validate configuration, sensor status, heartbeat, drive diagnostics and emergency-stop circuit. Enable any actuator only through a reviewed hardware-specific procedure.

## Fault response
On unexpected motion, communication loss, overheating, sensor disagreement or mechanical noise: stop through the independent emergency-stop path, isolate power as appropriate, secure the robot against falling, preserve logs and inspect before reset. Never automatically resume motion after a fault or reboot.

## Maintenance
Record firmware/software versions, calibration values, replaced parts, checks performed and approvals. Do not use the MCU sketches as replacements for vendor drive safety configuration.
