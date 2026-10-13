# Bill of Materials (Planning Worksheet)

This is a categories list, not a purchase-ready BOM. No exact part numbers or vendor claims are locked.

| Subsystem | Candidate category | Decisions still needed |
|---|---|---|
| AI compute | Embedded GPU module or industrial x86 | Inference load, thermals, power, software support |
| Motion compute | Real-time x86/ARM controller | Scheduling, fieldbus support, watchdog and isolation |
| Joint drives | Vendor servo/BLDC drives | Torque/speed curve, encoder, STO, bus, regenerative handling |
| Local controllers | MCU boards or vendor controllers | I/O, isolation, timers, ADCs, CAN/Ethernet and firmware support |
| Sensors | IMU, encoders, force/contact, depth/range | Range, sample rate, latency, calibration and failure modes |
| Safety | E-stop, safety relay/controller, drive STO chain | Risk assessment and applicable safety standard |
| Power | Battery, BMS, fuses, contactors, DC/DC | Peak/continuous draw, fault current, thermal enclosure |
| Mechanical | Frame, bearings, joints, wiring chain, feet | Loads, fatigue, impact, serviceability |

Obtain qualified engineering review before ordering high-current power components or enabling load-bearing actuators.
