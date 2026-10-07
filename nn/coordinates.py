"""
nn.coordinates -- a 3-D coordinate and value system on signed (+, -) axes.

Two cooperating pieces:

1. **Geometry** -- points/vectors in 3-D space with an explicit sign convention
   for each axis. The three orthonormal axes `X`, `Y`, `Z` each carry a signed
   direction, so every one of the 8 *octants* is addressable.

        Z +                     yaw/pitch/roll, right-handed by default
        |  Y +
        | /
        |/
        o------- X +

   * `Axis`        : a named, signed basis direction (unit vector + sign).
   * `Vec3`        : a 3-D vector with +X/-X, +Y/-Y, +Z/-Z accessors, full
                     vector algebra, and conversions to/from octant codes.
   * `OCTANTS`     : the 8 sign triples (+,-,+ ...) and their index 0..7.
   * frame helpers : `right_handed_frame`, `left_handed_frame`, `octant_of`,
                     `axis_sign`, `project`, `reflect`.

2. **Values** -- a *scalar/vector field* sampled on a signed integer grid:

        Values3D(res=5).set(X=+1, Y=-2, Z=0, value=3.5)

   * `Values3D`    : an `(n, n, n)` array indexed by signed coordinates, with
                     `get/set`, slicing by planes, symmetry/parity queries,
                     and numeric gradients along each signed axis.

Sign convention
---------------
Every axis has a **positive (+)** and **negative (-)** half:

    +X right, -X left
    +Y up,    -Y down
    +Z out,   -Z in          (right-handed: X x Y = Z)

A coordinate triple `(x, y, z)` may be any real number; its *octant* is the
sign pattern `(sign(x), sign(y), sign(z))`, one of 8 values.

All arrays are NumPy; the module has no external dependencies.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "Axis", "X_AXIS", "Y_AXIS", "Z_AXIS", "AXES",
    "Vec3", "OCTANTS", "octant_of", "octant_name", "octant_center",
    "all_octants", "right_handed_frame", "left_handed_frame", "frame_of",
    "handedness", "axis_sign", "project", "reflect",
    "Values3D", "signed_grid", "gradient3d",
]


# ===========================================================================
# 1. Axes
# ===========================================================================
class Axis:
    """A named basis direction with a sign (+ / -) and a unit vector."""

    def __init__(self, name: str, unit: tuple, sign: int = +1):
        if sign not in (-1, +1):
            raise ValueError("sign must be +1 or -1")
        self.name = name
        self.unit = np.array(unit, dtype=float)
        self.sign = sign

    @property
    def vector(self) -> np.ndarray:
        """Signed unit vector: `sign * unit`."""
        return self.sign * self.unit

    def __neg__(self) -> "Axis":
        return Axis(self.name, tuple(self.unit), -self.sign)

    def __repr__(self):
        s = "+" if self.sign > 0 else "-"
        return f"Axis({s}{self.name})"


X_AXIS = Axis("X", (1, 0, 0))
Y_AXIS = Axis("Y", (0, 1, 0))
Z_AXIS = Axis("Z", (0, 0, 1))
AXES = {"X": X_AXIS, "Y": Y_AXIS, "Z": Z_AXIS}


def axis_sign(axis: str, sign: int = +1) -> Axis:
    """Return `Axis` for name in {'X','Y','Z'} with the given sign (+1/-1)."""
    if axis.upper() not in AXES:
        raise KeyError(f"Unknown axis '{axis}'. Use X, Y or Z.")
    return Axis(axis.upper(), tuple(AXES[axis.upper()].unit), sign)


# ===========================================================================
# 2. Octants and handedness
# ===========================================================================
def octant_of(x: float, y: float, z: float) -> int:
    """Return the octant index `0..7` of `(x, y, z)`.

    Bit layout: `bit0 = x>=0`, `bit1 = y>=0`, `bit2 = z>=0`.
    So `0` is (-,-,-) and `7` is (+,+,+).
    """
    return int((x >= 0) | ((y >= 0) << 1) | ((z >= 0) << 2))


def octant_name(index: int) -> str:
    """Human label like '+X -Y +Z' for an octant index."""
    x, y, z = OCTANTS[index]
    return f"{'+' if x > 0 else '-'}X {'+' if y > 0 else '-'}Y {'+' if z > 0 else '-'}Z"


# (x_sign, y_sign, z_sign) for each octant index 0..7.
OCTANTS = [tuple(1 if (i >> b) & 1 else -1 for b in range(3)) for i in range(8)]


def octant_center(index: int) -> np.ndarray:
    """A representative interior point (unit diagonal) of octant `index`."""
    return np.array(OCTANTS[index], dtype=float)


def all_octants() -> list:
    """Return `[(index, (sx, sy, sz), name), ...]` for all 8 octants."""
    return [(i, OCTANTS[i], octant_name(i)) for i in range(8)]


def handedness(x_hat, y_hat, z_hat) -> str:
    """Return 'right', 'left', or 'degenerate' from three basis vectors.

    Right-handed iff `x_hat . (y_hat x z_hat) > 0` (i.e. `X x Y = Z`).
    """
    triple = float(np.dot(np.asarray(x_hat, dtype=float),
                          np.cross(np.asarray(y_hat, dtype=float),
                                   np.asarray(z_hat, dtype=float))))
    if abs(triple) < 1e-12:
        return "degenerate"
    return "right" if triple > 0 else "left"


def right_handed_frame(x=None, y=None, z=None) -> np.ndarray:
    """Build a 3x3 frame (rows = axes). Defaults to the standard +X,+Y,+Z."""
    x = X_AXIS.vector if x is None else np.asarray(x, dtype=float)
    y = Y_AXIS.vector if y is None else np.asarray(y, dtype=float)
    z = Z_AXIS.vector if z is None else np.asarray(z, dtype=float)
    # Enforce right-handedness by flipping Z if necessary.
    if handedness(x, y, z) != "right":
        z = -z
    return np.vstack([x, y, z])


def left_handed_frame(x=None, y=None, z=None) -> np.ndarray:
    """Like `right_handed_frame` but with `X x Y = -Z` (left-handed)."""
    x = X_AXIS.vector if x is None else np.asarray(x, dtype=float)
    y = Y_AXIS.vector if y is None else np.asarray(y, dtype=float)
    z = Z_AXIS.vector if z is None else np.asarray(z, dtype=float)
    if handedness(x, y, z) != "left":
        z = -z
    return np.vstack([x, y, z])


def frame_of(x_sign=+1, y_sign=+1, z_sign=+1, handed="right") -> np.ndarray:
    """Frame whose axes carry the given signs, with the requested handedness."""
    x = axis_sign("X", x_sign).vector
    y = axis_sign("Y", y_sign).vector
    z = axis_sign("Z", z_sign).vector
    return (right_handed_frame(x, y, z) if handed == "right"
            else left_handed_frame(x, y, z))


# ===========================================================================
# 3. Vec3
# ===========================================================================
class Vec3:
    """A 3-D vector with signed-axis semantics and full vector algebra."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float).reshape(-1)
            if arr.size != 3:
                raise ValueError("Vec3 needs exactly 3 components")
            x, y, z = arr
        self.v = np.array([x, y, z], dtype=float)

    # -- signed-axis accessors --------------------------------------------
    @property
    def x(self) -> float:
        return float(self.v[0])

    @property
    def y(self) -> float:
        return float(self.v[1])

    @property
    def z(self) -> float:
        return float(self.v[2])

    def plus(self, axis: str) -> float:
        """Magnitude along the positive half of `axis` (0 if on the other side)."""
        c = self.v["XYZ".index(axis.upper())]
        return float(c) if c > 0 else 0.0

    def minus(self, axis: str) -> float:
        """Magnitude along the negative half of `axis` (positive number)."""
        c = self.v["XYZ".index(axis.upper())]
        return float(-c) if c < 0 else 0.0

    def sign_triple(self) -> tuple:
        """`(sign(x), sign(y), sign(z))` each in {-1, +1}."""
        return tuple(1 if c >= 0 else -1 for c in self.v)

    def octant(self) -> int:
        """Octant index 0..7 of this point."""
        return octant_of(*self.v)

    def octant_label(self) -> str:
        return octant_name(self.octant())

    # -- magnitude ---------------------------------------------------------
    def norm(self) -> float:
        """Euclidean length."""
        return float(np.linalg.norm(self.v))

    def norm_l1(self) -> float:
        return float(np.sum(np.abs(self.v)))

    def norm_linf(self) -> float:
        return float(np.max(np.abs(self.v)))

    def normalized(self) -> "Vec3":
        n = self.norm()
        return Vec3(self.v / n) if n > 0 else Vec3(0, 0, 0)

    # -- algebra -----------------------------------------------------------
    def __add__(self, o): return Vec3(self.v + _vec(o))
    def __sub__(self, o): return Vec3(self.v - _vec(o))
    def __neg__(self): return Vec3(-self.v)
    def __mul__(self, s): return Vec3(self.v * s)
    __rmul__ = __mul__
    def __truediv__(self, s): return Vec3(self.v / s)

    def dot(self, o) -> float:
        """Dot product `<self, o>`."""
        return float(np.dot(self.v, _vec(o)))

    def cross(self, o) -> "Vec3":
        """Cross product `self x o` (right-handed)."""
        return Vec3(np.cross(self.v, _vec(o)))

    def angle_to(self, o) -> float:
        """Angle in radians between this and `o`."""
        a, b = self.v, _vec(o)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        c = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
        return float(np.arccos(c))

    def angle_with_axis(self, axis: str) -> float:
        """Angle in radians to a named signed + axis."""
        return self.angle_to(AXES[axis.upper()].vector)

    def cosines(self) -> np.ndarray:
        """Direction cosines w.r.t. +X, +Y, +Z."""
        n = self.norm() or 1.0
        return self.v / n

    def distance_to(self, o) -> float:
        return float(np.linalg.norm(self.v - _vec(o)))

    def manhattan_to(self, o) -> float:
        return float(np.sum(np.abs(self.v - _vec(o))))

    # -- conversions -------------------------------------------------------
    def to_array(self) -> np.ndarray:
        return self.v.copy()

    def to_tuple(self) -> tuple:
        return tuple(float(c) for c in self.v)

    def to_spherical(self) -> tuple:
        """`(r, theta, phi)` with `theta` from +Z, `phi` from +X in the XY plane."""
        x, y, z = self.v
        r = float(np.sqrt(x * x + y * y + z * z))
        if r == 0:
            return 0.0, 0.0, 0.0
        theta = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
        phi = float(np.arctan2(y, x))
        return r, theta, phi

    @staticmethod
    def from_spherical(r: float, theta: float, phi: float) -> "Vec3":
        """Inverse of `to_spherical`."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return Vec3(x, y, z)

    def rotate(self, axis: str, angle: float) -> "Vec3":
        """Rotate about a signed cardinal axis by `angle` radians (Rodrigues)."""
        k = AXES[axis.upper()].vector
        c, s = np.cos(angle), np.sin(angle)
        v = self.v
        return Vec3(v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c))

    def __iter__(self):
        return iter(self.v)

    def __eq__(self, o):
        return isinstance(o, Vec3) and np.allclose(self.v, _vec(o))

    def __repr__(self):
        return f"Vec3({self.x:g}, {self.y:g}, {self.z:g})"


def _vec(o) -> np.ndarray:
    """Coerce Vec3 / sequence to a length-3 float array."""
    return o.v if isinstance(o, Vec3) else np.asarray(o, dtype=float).reshape(3)


def project(a, b) -> "Vec3":
    """Orthogonal projection of `a` onto `b`."""
    bv = _vec(b)
    bb = float(np.dot(bv, bv))
    if bb == 0:
        return Vec3(0, 0, 0)
    return Vec3(_vec(a) * (np.dot(_vec(a), bv) / bb))


def reflect(a, normal) -> "Vec3":
    """Reflect vector `a` across the plane with unit `normal`."""
    n = _vec(normal)
    n = n / (np.linalg.norm(n) or 1.0)
    return Vec3(_vec(a) - 2 * np.dot(_vec(a), n) * n)


# ===========================================================================
# 4. Values on a signed 3-D grid
# ===========================================================================
def signed_grid(res: int = 5):
    """Return `(coords, index_map)` for a signed integer grid.

    `res` must be odd so that 0 is the center. Values run from
    `-(res//2)` to `+(res//2)` on each axis.

    Returns
    -------
    coords    : array of shape `(res, 3)` listing (x, y, z) for every cell,
                ordered so that index `i` maps to `coords[i]`.
    index_map : dict `{(x, y, z): i}` for O(1) lookup.
    """
    if res % 2 == 0:
        raise ValueError("res must be odd so the origin is a grid point")
    half = res // 2
    rng = range(-half, half + 1)
    coords = np.array(list(itertools.product(rng, rng, rng)), dtype=int)
    index_map = {tuple(int(c) for c in row): i for i, row in enumerate(coords)}
    return coords, index_map


class Values3D:
    """A scalar field sampled on a signed 3-D integer grid.

    Each grid point `(x, y, z)` (each in `[-h, +h]`) holds a float value.
    Axes are signed: `x>0` is +X, `x<0` is -X, and likewise for Y and Z.
    """

    def __init__(self, res: int = 5, fill: float = 0.0):
        self.res = res
        self.half = res // 2
        self.coords, self.index = signed_grid(res)
        shape = (self.half * 2 + 1,) * 3
        self.data = np.full(shape, float(fill))

    # -- index helpers -----------------------------------------------------
    def _idx(self, x, y, z) -> tuple:
        for c in (x, y, z):
            if not (-self.half <= c <= self.half):
                raise IndexError(f"coordinate {c} outside [{-self.half}, {self.half}]")
        return (int(x + self.half), int(y + self.half), int(z + self.half))

    def get(self, x, y, z) -> float:
        """Value at signed coordinate `(x, y, z)`."""
        return float(self.data[self._idx(x, y, z)])

    def set(self, x, y, z, value: float) -> None:
        """Assign `value` at signed coordinate `(x, y, z)`."""
        self.data[self._idx(x, y, z)] = float(value)

    def __getitem__(self, key):
        return self.get(*key)

    def __setitem__(self, key, value):
        self.set(*key, value)

    # -- whole-axis views --------------------------------------------------
    def axis_values(self, axis: str, index: int) -> np.ndarray:
        """The 2-D plane of values at a fixed signed coordinate on `axis`.

        E.g. `axis_values('Z', 0)` returns the XY plane at z = 0.
        """
        a = "XYZ".index(axis.upper())
        sl = [slice(None)] * 3
        sl[a] = index + self.half
        return self.data[tuple(sl)]

    def positive_half(self, axis: str) -> np.ndarray:
        """All values strictly on the +half of `axis`."""
        a = "XYZ".index(axis.upper())
        sl = [slice(None)] * 3
        sl[a] = slice(self.half + 1, None)
        return self.data[tuple(sl)]

    def negative_half(self, axis: str) -> np.ndarray:
        """All values strictly on the -half of `axis`."""
        a = "XYZ".index(axis.upper())
        sl = [slice(None)] * 3
        sl[a] = slice(0, self.half)
        return self.data[tuple(sl)]

    def half_means(self) -> dict:
        """Mean value on each signed half of each axis, e.g. `'+X'`, `'-Z'`."""
        out = {}
        for axis in "XYZ":
            out[f"+{axis}"] = float(self.positive_half(axis).mean()) if self.half else 0.0
            out[f"-{axis}"] = float(self.negative_half(axis).mean()) if self.half else 0.0
        return out

    def asymmetry(self) -> dict:
        """`(+half_mean - -half_mean)` per axis: how lopsided the field is."""
        m = self.half_means()
        return {a: round(m[f"+{a}"] - m[f"-{a}"], 6) for a in "XYZ"}

    # -- octants -----------------------------------------------------------
    def octant_mean(self, index: int) -> float:
        """Mean value within one octant (index 0..7)."""
        sx, sy, sz = OCTANTS[index]
        xs = slice(self.half + 1, None) if sx > 0 else slice(0, self.half)
        ys = slice(self.half + 1, None) if sy > 0 else slice(0, self.half)
        zs = slice(self.half + 1, None) if sz > 0 else slice(0, self.half)
        block = self.data[xs, ys, zs]
        return float(block.mean()) if block.size else 0.0

    def octant_means(self) -> dict:
        """`{octant_name: mean}` for all 8 octants."""
        return {octant_name(i): self.octant_mean(i) for i in range(8)}

    def parity(self) -> dict:
        """Correlation of the field with each axis's sign function.

        A large positive value means the field tends to be higher on the
        positive half of that axis; negative means it favours the negative half.
        This measures the field's *signed-axis* structure.
        """
        out = {}
        for a, axis in enumerate("XYZ"):
            sign = np.sign(np.arange(-self.half, self.half + 1)).astype(float)
            shape = [1, 1, 1]
            shape[a] = self.data.shape[a]
            s = sign.reshape(shape)
            out[axis] = float(np.mean(self.data * s))
        return out

    # -- calculus on the grid ---------------------------------------------
    def gradient(self, axis: str) -> np.ndarray:
        """Central-difference derivative along a signed axis."""
        a = "XYZ".index(axis.upper())
        return np.gradient(self.data, axis=a)

    def directional_derivative(self, direction) -> np.ndarray:
        """Directional derivative along an arbitrary unit vector."""
        d = _vec(direction)
        d = d / (np.linalg.norm(d) or 1.0)
        gx = self.gradient("X")
        gy = self.gradient("Y")
        gz = self.gradient("Z")
        return d[0] * gx + d[1] * gy + d[2] * gz

    def laplacian(self) -> np.ndarray:
        """Discrete Laplacian (sum of second differences)."""
        out = np.zeros_like(self.data)
        for a in range(3):
            out += np.gradient(np.gradient(self.data, axis=a), axis=a)
        return out

    def divergence(self, field) -> np.ndarray:
        """Divergence of a vector field defined by three `Values3D` components."""
        return field[0].gradient("X") + field[1].gradient("Y") + field[2].gradient("Z")

    # -- construction helpers ---------------------------------------------
    @classmethod
    def from_function(cls, fn, res: int = 5) -> "Values3D":
        """Build a field by evaluating `fn(x, y, z)` at every signed grid point."""
        obj = cls(res, 0.0)
        for (x, y, z) in obj.coords:
            obj.set(x, y, z, fn(int(x), int(y), int(z)))
        return obj

    def coords_where(self, predicate) -> list:
        """All signed coordinates `(x, y, z)` whose value satisfies `predicate(value)`."""
        out = []
        for (x, y, z) in self.coords:
            if predicate(self.get(x, y, z)):
                out.append((int(x), int(y), int(z)))
        return out

    def summary(self) -> str:
        """Human-readable overview of the field and its signed halves."""
        hm = self.half_means()
        lines = [f"Values3D(res={self.res}, origin-centered, axes +/-X +/-Y +/-Z)"]
        lines.append(f"  min={self.data.min():.3f} max={self.data.max():.3f} "
                     f"mean={self.data.mean():.3f}")
        for axis in "XYZ":
            lines.append(f"  {axis}: +half {hm['+'+axis]: .3f} | "
                         f"-half {hm['-'+axis]: .3f} | "
                         f"asym {self.asymmetry()[axis]: .3f}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"Values3D(res={self.res}, shape={self.data.shape}, "
                f"range=[{self.data.min():.2f}, {self.data.max():.2f}])")


def gradient3d(values: Values3D) -> tuple:
    """Return `(d/dX, d/dY, d/dZ)` arrays for a `Values3D` field."""
    return values.gradient("X"), values.gradient("Y"), values.gradient("Z")
