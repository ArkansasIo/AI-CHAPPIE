# Hardware Design (Conceptual)

## Suggested partition
- High-level AI computer: industrial x86 or an embedded GPU module; runs perception, user interaction and planning.
- Real-time motion computer: isolated from the AI runtime; runs validated control loops and state estimation.
- Local MCU/controllers: one or more vendor-supported controllers for joints, hands, head/neck and sensor acquisition.
- Independent safety subsystem: safety-rated relay/controller, E-stop, drive STO/enable path, guarded power distribution and fault monitoring.
- Sensors: joint encoders, IMU, foot contact/force sensing, range/depth cameras, temperature and battery telemetry.

## Design constraints
Select all components against torque-speed envelopes, peak and continuous current, thermal limits, regenerative energy, encoder precision, bus timing, structural loads, and serviceability. Battery chemistry, fusing, wire gauge, connectors, grounding, EMC and cooling require qualified design review.

No real bill of materials is fixed by this document. Vendor selection and exact ratings are open engineering decisions. Example joint angle/speed limits in code are simulation placeholders, not safe physical values.
