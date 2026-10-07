"""
Example 7 -- the X / Y / Z coordinate and value system (signed + / - axes).

Demonstrates:
  1. the three axes and the sign convention (+X, -X, +Y, -Y, +Z, -Z)
  2. handedness (right-handed X x Y = Z) and building signed frames
  3. `Vec3`: signed accessors, octant classification, spherical coords, rotation
  4. all 8 octants
  5. `Values3D`: a scalar field on a signed grid, half-space analysis, parity,
     octant means, and numerical gradients (calculus on the grid)

    python examples/07_coordinates.py
"""

import numpy as np

from nn.coordinates import (
    AXES, OCTANTS, Vec3, Values3D, X_AXIS, Y_AXIS, Z_AXIS,
    all_octants, axis_sign, frame_of, handedness, octant_name, octant_of,
)


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def demo_axes():
    banner("1. The signed axes")
    print("Positive half  ->  negative half is the opposite direction:\n")
    for name in ("X", "Y", "Z"):
        pos = axis_sign(name, +1)
        neg = axis_sign(name, -1)
        print(f"  {pos!r:<9} vector {pos.vector}      "
              f"{neg!r:<9} vector {neg.vector}")
    print("\nRight-handed check:  X x Y =", np.cross(X_AXIS.vector, Y_AXIS.vector),
          "== Z", Z_AXIS.vector)


def demo_frames():
    banner("2. Handedness and signed frames")
    print("Standard (+,+,+) :", handedness(X_AXIS.vector, Y_AXIS.vector, Z_AXIS.vector))
    print("Swapped Y and Z  :", handedness(X_AXIS.vector, Z_AXIS.vector, Y_AXIS.vector))
    print("\nframe_of(+X, -Y, +Z) -- note Z flips to stay right-handed:")
    print(frame_of(+1, -1, +1))
    print("\nframe_of(+X, -Y, +Z, handed='left'):")
    print(frame_of(+1, -1, +1, handed="left"))


def demo_vec3():
    banner("3. Vec3 with signed components")
    v = Vec3(3, -4, 12)
    print(f"v = {v}")
    print(f"  norm (L2)   = {v.norm():.4f}")
    print(f"  norm (L1)   = {v.norm_l1():.4f}")
    print(f"  norm (Linf) = {v.norm_linf():.4f}")
    print(f"  +X part = {v.plus('X')},  -X part = {v.minus('X')}")
    print(f"  +Y part = {v.plus('Y')},  -Y part = {v.minus('Y')}")
    print(f"  sign triple = {v.sign_triple()}  -> octant {v.octant()} ({v.octant_label()})")

    r, theta, phi = v.to_spherical()
    print(f"\n  spherical: r={r:.4f}, theta={theta:.4f}, phi={phi:.4f}")
    back = Vec3.from_spherical(r, theta, phi)
    print(f"  round-trip -> {back}  (matches: {v == back})")

    print(f"\n  rotate 90 deg about +Z: {v.rotate('Z', np.pi / 2)}")
    print(f"  rotate 90 deg about -Z: {v.rotate('Z', -np.pi / 2)}")

    a, b = Vec3(1, 0, 0), Vec3(0, 1, 0)
    print(f"\n  {a} . {b} = {a.dot(b)}    {a} x {b} = {a.cross(b)}")
    print(f"  angle between {a} and {b} = {np.degrees(a.angle_to(b)):.1f} deg")


def demo_octants():
    banner("4. The 8 octants of signed space")
    for i, signs, name in all_octants():
        print(f"  octant {i}:  signs {signs}   {name}")
    print(f"\n  point (0.5, -0.2, 0.9) -> octant "
          f"{octant_of(0.5, -0.2, 0.9)} = {octant_name(octant_of(0.5, -0.2, 0.9))}")


def demo_values():
    banner("5. Values3D -- a value field on the signed grid")
    # A field that rises toward +X, falls toward +Y, and is symmetric in Z.
    field = Values3D.from_function(lambda x, y, z: 2 * x - 3 * y, res=5)
    print(field)
    print()
    print(field.summary())

    print("\n  value at (+1, -2, 0) =", field.get(1, -2, 0))
    print("  half-space means    =", {k: round(v, 3) for k, v in field.half_means().items()})
    print("  asymmetry (+/-)     =", field.asymmetry())
    print("  parity per axis     =", {k: round(v, 3) for k, v in field.parity().items()})

    print("\n  octant means (only the X/Y sign quadrants matter for 2x - 3y):")
    for name, value in field.octant_means().items():
        print(f"    {name:<12} {value: .4f}")

    # --- calculus on the grid ---
    cx = cy = cz = field.half  # the origin cell
    print("\n  analytic field f = 2x - 3y  =>  grad = (2, -3, 0)")
    print(f"    numerical d/dX at origin = {field.gradient('X')[cx, cy, cz]:.4f}")
    print(f"    numerical d/dY at origin = {field.gradient('Y')[cx, cy, cz]:.4f}")
    print(f"    numerical d/dZ at origin = {field.gradient('Z')[cx, cy, cz]:.4f}")

    # directional derivative toward (+1, +1, 0)/sqrt(2)
    d = np.array([1, 1, 0]) / np.sqrt(2)
    dd = field.directional_derivative(d)
    print(f"    directional deriv toward (1,1,0)/sqrt2 = "
          f"{dd[cx, cy, cz]:.4f}  (expect {(2 - 3) / np.sqrt(2):.4f})")

    print(f"\n  cells with value > 0: {len(field.coords_where(lambda v: v > 0))} of "
          f"{len(field.coords)}")

    # A radial field to show the Laplacian.
    radial = Values3D.from_function(
        lambda x, y, z: -(x * x + y * y + z * z), res=5)
    lap = radial.laplacian()
    print(f"\n  radial field f = -(x^2+y^2+z^2):  Laplacian at origin = "
          f"{lap[cx, cy, cz]:.4f}  (analytic -6)")


if __name__ == "__main__":
    demo_axes()
    demo_frames()
    demo_vec3()
    demo_octants()
    demo_values()
    print("\nDone.")
