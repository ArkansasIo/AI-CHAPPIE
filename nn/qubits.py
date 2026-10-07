"""
nn.qubits -- a quantum logic (qubit) system.

This module implements the mathematics of quantum computation in pure NumPy:

States
------
* `Qubit`       : a single-qubit state `|psi> = alpha|0> + beta|1>` with
                  normalization, Bloch-sphere coordinates, and measurement.
* `QuantumState`: an n-qubit register living in a `2^n`-dimensional Hilbert
                  space, with tensor products, partial traces, and measurement.

Gates (unitary matrices)
------------------------
Single-qubit: `I`, `X`, `Y`, `Z` (Pauli), `H` (Hadamard), `S`, `T`,
              `RX/RY/RZ(theta)`, `PHASE(phi)`
Multi-qubit : `CNOT`, `CZ`, `SWAP`, `CPHASE`, `TOFFOLI` (CCX), `FREDKIN` (CSWAP)

Algebra
-------
* tensor products (Kronecker), matrix multiplication, unitary checks
* commutators `[A, B] = AB - BA` and anticommutators `{A, B} = AB + BA`
* expectation values `<psi|A|psi>`
* entanglement: concurrence, entanglement entropy, Schmidt decomposition
* Bell and GHZ states, quantum teleportation, Deutsch-Jozsa

The sign/phase convention
-------------------------
    |0> = [1, 0]^T ,   |1> = [0, 1]^T
    |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
with `theta` the polar angle from +Z and `phi` the azimuth from +X.

Only NumPy is required.
"""

from __future__ import annotations

import cmath
import itertools
from math import cos, pi, sin, sqrt

import numpy as np

__all__ = [
    "I", "X", "Y", "Z", "H", "S", "T",
    "RX", "RY", "RZ", "PHASE",
    "CNOT", "CZ", "SWAP", "CPHASE", "TOFFOLI", "FREDKIN",
    "SINGLE_QUBIT_GATES", "MULTI_QUBIT_GATES", "gate",
    "Qubit", "QuantumState",
    "bell_state", "ghz_state", "w_state",
    "concurrence", "entanglement_entropy", "schmidt_coefficients",
    "partial_trace", "expectation", "commutator", "anticommutator",
    "is_unitary", "tensor", "QuantumCircuit",
    "teleportation_circuit", "deutsch_jozsa", "phase_oracle",
    "classical_gate_unitary", "reversible_and", "reversible_or",
    "reversible_nand", "reversible_xor", "reversible_not",
    "REVERSIBLE_GATES", "classical_to_quantum",
]

_EPS = 1e-12


# ===========================================================================
# 1. Single-qubit gates
# ===========================================================================
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)          # NOT / bit-flip
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)       # bit+phase flip
Z = np.array([[1, 0], [0, -1]], dtype=complex)         # phase flip

H = np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)   # Hadamard
S = np.array([[1, 0], [0, 1j]], dtype=complex)             # phase pi/2
T = np.array([[1, 0], [0, cmath.exp(1j * pi / 4)]], dtype=complex)  # phase pi/4


def RX(theta: float) -> np.ndarray:
    """Rotation about the X axis by `theta` radians.

        RX(t) = exp(-i t X/2) = [[cos(t/2), -i sin(t/2)],
                                 [-i sin(t/2), cos(t/2)]]
    """
    c, s = cos(theta / 2), sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def RY(theta: float) -> np.ndarray:
    """Rotation about the Y axis by `theta` radians."""
    c, s = cos(theta / 2), sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def RZ(theta: float) -> np.ndarray:
    """Rotation about the Z axis by `theta` radians."""
    return np.array([[cmath.exp(-1j * theta / 2), 0],
                     [0, cmath.exp(1j * theta / 2)]], dtype=complex)


def PHASE(phi: float) -> np.ndarray:
    """Phase gate: `diag(1, e^{i phi})` (generalizes S, T, Z)."""
    return np.array([[1, 0], [0, cmath.exp(1j * phi)]], dtype=complex)


SINGLE_QUBIT_GATES = {
    "I": I, "X": X, "Y": Y, "Z": Z, "H": H, "S": S, "T": T,
}


# ===========================================================================
# 2. Multi-qubit gates (built by embedding single-qubit gates)
# ===========================================================================
def _kron(*mats) -> np.ndarray:
    """Kronecker (tensor) product of matrices, left to right."""
    out = np.array([[1]], dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out


def tensor(*states) -> np.ndarray:
    """Tensor product of states/vectors (public alias of Kronecker product)."""
    out = np.array([1.0], dtype=complex)
    for s in states:
        out = np.kron(out, np.asarray(s, dtype=complex))
    return out


def _embed(gate_2x2: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Place a single-qubit gate on `target` in an `n_qubits` register.

    Qubit 0 is the most-significant factor, so the full operator is
    `I^{target} (x) gate (x) I^{n-qubits-1-target}`.
    """
    mats = [I] * n_qubits
    mats[target] = gate_2x2
    return _kron(*mats)


def controlled(U: np.ndarray, n_qubits: int = 2, control: int = 0,
               target: int = 1) -> np.ndarray:
    """Build a controlled-U operator on `n_qubits`.

    Applies `U` to `target` only on the subspace where `control` is `|1>`.
    """
    dim = 1 << n_qubits
    op = np.eye(dim, dtype=complex)
    # Iterate over all basis indices; apply U when the control bit is 1.
    for i in range(dim):
        # bit position: qubit 0 is the most significant bit
        cbit = (i >> (n_qubits - 1 - control)) & 1
        if cbit == 1:
            # gather the target bit
            t_bit = (i >> (n_qubits - 1 - target)) & 1
            # the paired index with the target bit flipped
            j = i ^ (1 << (n_qubits - 1 - target))
            op[i, i] = U[t_bit, t_bit]
            op[i, j] = U[t_bit, 1 - t_bit]
    return op


CNOT = controlled(X, 2, 0, 1)
CZ = controlled(Z, 2, 0, 1)
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)
CPHASE = controlled(S, 2, 0, 1)
TOFFOLI = controlled(X, 3, 0, 2)          # CCX (control 0, control 1, target 2)
FREDKIN = controlled(SWAP, 3, 0, 1)       # CSWAP (control 0, swap 1 and 2)

MULTI_QUBIT_GATES = {
    "CNOT": CNOT, "CZ": CZ, "SWAP": SWAP, "CPHASE": CPHASE,
    "TOFFOLI": TOFFOLI, "FREDKIN": FREDKIN,
}


def gate(name: str, *args) -> np.ndarray:
    """Look up a gate by name (parametric gates take their angle arguments)."""
    key = name.upper()
    if key in SINGLE_QUBIT_GATES:
        return SINGLE_QUBIT_GATES[key]
    if key in MULTI_QUBIT_GATES:
        return MULTI_QUBIT_GATES[key]
    if key == "RX":
        return RX(*args)
    if key == "RY":
        return RY(*args)
    if key == "RZ":
        return RZ(*args)
    if key == "PHASE":
        return PHASE(*args)
    raise KeyError(f"Unknown gate '{name}'.")


# ===========================================================================
# 3. Algebra helpers
# ===========================================================================
def is_unitary(u: np.ndarray, tol: float = 1e-9) -> bool:
    """True if `U^dagger U = I`."""
    u = np.asarray(u, dtype=complex)
    return np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=tol)


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`[A, B] = AB - BA` (zero iff A and B commute)."""
    return a @ b - b @ a


def anticommutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`{A, B} = AB + BA`."""
    return a @ b + b @ a


def expectation(state: np.ndarray, op: np.ndarray) -> complex:
    """`<psi| op |psi>` for a normalized state vector."""
    psi = np.asarray(state, dtype=complex).reshape(-1)
    return complex(psi.conj() @ (op @ psi))


# ===========================================================================
# 4. Single qubit
# ===========================================================================
class Qubit:
    """A single-qubit state `|psi> = alpha|0> + beta|1>`.

    Parameters
    ----------
    alpha, beta : complex amplitudes (auto-normalized)
    """

    def __init__(self, alpha=1.0, beta=0.0):
        v = np.array([complex(alpha), complex(beta)], dtype=complex)
        n = np.linalg.norm(v)
        if n < _EPS:
            raise ValueError("zero vector is not a valid qubit state")
        self.state = v / n

    # -- constructors ------------------------------------------------------
    @staticmethod
    def zero() -> "Qubit":
        """`|0>`."""
        return Qubit(1, 0)

    @staticmethod
    def one() -> "Qubit":
        """`|1>`."""
        return Qubit(0, 1)

    @staticmethod
    def plus() -> "Qubit":
        """`|+> = (|0> + |1>)/sqrt2`."""
        return Qubit(1 / sqrt(2), 1 / sqrt(2))

    @staticmethod
    def minus() -> "Qubit":
        """`|-> = (|0> - |1>)/sqrt2`."""
        return Qubit(1 / sqrt(2), -1 / sqrt(2))

    @staticmethod
    def from_bloch(theta: float, phi: float) -> "Qubit":
        """State at polar `theta` and azimuth `phi` on the Bloch sphere."""
        return Qubit(cos(theta / 2), cmath.exp(1j * phi) * sin(theta / 2))

    # -- properties --------------------------------------------------------
    @property
    def alpha(self) -> complex:
        return complex(self.state[0])

    @property
    def beta(self) -> complex:
        return complex(self.state[1])

    def probabilities(self) -> tuple:
        """`(P(0), P(1))` -- the measurement outcome distribution."""
        return (float(abs(self.alpha) ** 2), float(abs(self.beta) ** 2))

    def bloch_vector(self) -> np.ndarray:
        """Bloch coordinates `(x, y, z) = (<X>, <Y>, <Z>)`.

        The state is pure iff `|r| = 1` (on the sphere), mixed if `|r| < 1`.
        """
        return np.array([float(np.real(expectation(self.state, P)))
                         for P in (X, Y, Z)])

    def to_bloch_angles(self) -> tuple:
        """`(theta, phi)` in radians recovered from the Bloch vector."""
        x, y, z = self.bloch_vector()
        r = sqrt(x * x + y * y + z * z) or 1.0
        theta = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
        phi = float(np.arctan2(y, x))
        return theta, phi

    def purity(self) -> float:
        """`Tr(rho^2)` = 1 for a pure state."""
        return float(np.sum(np.abs(self.state) ** 4) * 2 - 1 + 1)

    # -- operations --------------------------------------------------------
    def apply(self, u: np.ndarray) -> "Qubit":
        """Apply a 2x2 unitary, returning a new `Qubit` (immutable style)."""
        u = np.asarray(u, dtype=complex)
        if u.shape != (2, 2):
            raise ValueError("single-qubit gate must be 2x2")
        out = Qubit()
        out.state = u @ self.state
        return out

    def measure(self, rng: np.random.Generator | None = None,
                collapse: bool = True):
        """Measure in the computational basis (Born rule).

        Returns the outcome 0 or 1. If `collapse`, this qubit collapses to the
        measured basis state (unless the qubit is shared -- use
        `QuantumState.measure` for entangled registers).
        """
        rng = rng or np.random.default_rng()
        p0 = float(abs(self.alpha) ** 2)
        outcome = 0 if rng.random() < p0 else 1
        if collapse:
            self.state = np.array([1.0, 0.0]) if outcome == 0 else np.array([0.0, 1.0])
        return outcome

    def density_matrix(self) -> np.ndarray:
        """`rho = |psi><psi|`."""
        return np.outer(self.state, self.state.conj())

    def __repr__(self):
        return f"Qubit({self.alpha:.4g}, {self.beta:.4g})"


# ===========================================================================
# 5. Multi-qubit register
# ===========================================================================
class QuantumState:
    """An `n_qubit` register: a state vector in a `2^n`-dimensional space.

    Basis ordering: qubit 0 is the most-significant bit, so the index of a
    basis state `|b_0 b_1 ... b_{n-1}>` is `sum b_i 2^{n-1-i}`.
    """

    def __init__(self, state, n_qubits: int | None = None):
        v = np.asarray(state, dtype=complex).reshape(-1)
        if n_qubits is None:
            n_qubits = int(round(np.log2(v.size)))
            if (1 << n_qubits) != v.size:
                raise ValueError("state length must be a power of two")
        self.n = n_qubits
        self.dim = 1 << n_qubits
        if v.size != self.dim:
            raise ValueError(f"expected {self.dim} amplitudes, got {v.size}")
        norm = np.linalg.norm(v)
        if norm < _EPS:
            raise ValueError("zero vector is not a valid quantum state")
        self.state = v / norm

    # -- constructors ------------------------------------------------------
    @staticmethod
    def zeros(n_qubits: int) -> "QuantumState":
        """`|00...0>`."""
        v = np.zeros(1 << n_qubits, dtype=complex)
        v[0] = 1.0
        return QuantumState(v, n_qubits)

    @staticmethod
    def from_bits(bits) -> "QuantumState":
        """Computational basis state from a bit string/list, e.g. '101'."""
        if isinstance(bits, str):
            bits = [int(b) for b in bits]
        n = len(bits)
        index = 0
        for b in bits:
            index = (index << 1) | int(b)
        v = np.zeros(1 << n, dtype=complex)
        v[index] = 1.0
        return QuantumState(v, n)

    @staticmethod
    def from_qubits(*qubits: Qubit) -> "QuantumState":
        """Product state from individual `Qubit` objects."""
        v = tensor(*[q.state for q in qubits])
        return QuantumState(v, len(qubits))

    # -- helpers -----------------------------------------------------------
    def index_of(self, *bits) -> int:
        """Basis index for a bit pattern (qubit 0 first)."""
        idx = 0
        for b in bits:
            idx = (idx << 1) | int(b)
        return idx

    def bitstring(self, index: int) -> str:
        """Bit string (qubit 0 first) for a basis index."""
        return format(index, f"0{self.n}b")

    def probabilities(self) -> np.ndarray:
        """Measurement distribution over all `2^n` basis states."""
        return np.abs(self.state) ** 2

    def sample(self, n_shots: int = 1, rng: np.random.Generator | None = None) -> list:
        """Sample `n_shots` measurement outcomes; returns bit strings."""
        rng = rng or np.random.default_rng()
        probs = self.probabilities()
        probs = probs / probs.sum()
        idxs = rng.choice(self.dim, size=n_shots, p=probs)
        return [self.bitstring(int(i)) for i in idxs]

    def measure(self, qubit: int | None = None,
                rng: np.random.Generator | None = None):
        """Measure one qubit (or all) with proper collapse.

        Returns the outcome (int) for a single qubit, or a bit string for all.
        Collapsing one qubit of an entangled register correctly projects the
        remaining qubits onto the matching subspace.
        """
        rng = rng or np.random.default_rng()
        if qubit is None:
            probs = self.probabilities()
            outcome = int(rng.choice(self.dim, p=probs / probs.sum()))
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[outcome] = 1.0
            return self.bitstring(outcome)

        mask = 1 << (self.n - 1 - qubit)
        p1 = float(np.sum(np.abs(self.state) ** 2 * ((np.arange(self.dim) & mask) > 0)))
        outcome = 1 if rng.random() < p1 else 0

        keep = (np.arange(self.dim) & mask) > 0
        if outcome == 0:
            keep = ~keep
        self.state = np.where(keep, self.state, 0.0)
        norm = np.linalg.norm(self.state)
        if norm > _EPS:
            self.state = self.state / norm
        return outcome

    # -- operations --------------------------------------------------------
    def apply(self, op: np.ndarray) -> "QuantumState":
        """Apply a `2^n x 2^n` unitary operator."""
        op = np.asarray(op, dtype=complex)
        if op.shape != (self.dim, self.dim):
            raise ValueError(f"operator must be {self.dim}x{self.dim}")
        return QuantumState(op @ self.state, self.n)

    def apply_gate(self, gate_op: np.ndarray, target: int,
                   return_new: bool = True):
        """Apply a single-qubit gate to `target` (others untouched)."""
        op = _embed(np.asarray(gate_op, dtype=complex), target, self.n)
        if return_new:
            return self.apply(op)
        self.state = op @ self.state
        return self

    def apply_controlled(self, u: np.ndarray, control: int, target: int,
                         return_new: bool = True):
        """Apply a controlled-U between two qubits."""
        op = controlled(np.asarray(u, dtype=complex), self.n, control, target)
        if return_new:
            return self.apply(op)
        self.state = op @ self.state
        return self

    def cnot(self, control: int, target: int) -> "QuantumState":
        """Controlled-NOT using this register's qubit indices."""
        return self.apply_controlled(X, control, target)

    # -- structure ---------------------------------------------------------
    def reduced_density_matrix(self, qubit: int) -> np.ndarray:
        """Single-qubit reduced density matrix (tracing out the rest)."""
        return partial_trace(self, [qubit])

    def is_product(self, tol: float = 1e-9) -> bool:
        """True if the state factorizes as a tensor product (unentangled).

        Uses a genuine bipartition (first qubit vs. the rest); a cut of 0
        would trace out nothing and always yield zero entropy.
        """
        if self.n == 1:
            return True
        return entanglement_entropy(self, 1) < tol

    def nonzero_amplitudes(self, tol: float = 1e-9) -> list:
        """`[(bitstring, amplitude), ...]` for amplitudes above `tol`."""
        return [(self.bitstring(i), complex(a))
                for i, a in enumerate(self.state) if abs(a) > tol]

    def __repr__(self):
        terms = ", ".join(f"{amp:.3g}|{bs}>" for bs, amp in self.nonzero_amplitudes())
        return f"QuantumState({self.n} qubits: {terms})"


# ===========================================================================
# 6. Entanglement measures
# ===========================================================================
def partial_trace(state: QuantumState, keep: list) -> np.ndarray:
    """Reduced density matrix over `keep` qubit indices (trace out the rest)."""
    n = state.n
    keep = sorted(keep)
    traced = [q for q in range(n) if q not in keep]

    psi = state.state.reshape([2] * n)
    # Move traced qubits to the end, then reshape to (keep_dim, trace_dim).
    perm = keep + traced
    psi = np.transpose(psi, perm)
    k = 1 << len(keep)
    t = 1 << len(traced)
    mat = psi.reshape(k, t)
    return mat @ mat.conj().T


def schmidt_coefficients(state: QuantumState, cut: int = 1) -> np.ndarray:
    """Schmidt coefficients for a bipartition at `cut` (first `cut` qubits)."""
    rho = partial_trace(state, list(range(cut)))
    eigvals = np.linalg.eigvalsh(rho)
    return np.sqrt(np.clip(eigvals, 0, None))[::-1]


def entanglement_entropy(state: QuantumState, cut: int = 1,
                         base: float = 2.0) -> float:
    """Von Neumann entropy `S = -sum p log p` of the reduced state.

    0 for a product state, `log2(d)` for a maximally mixed reduced state.
    """
    coeffs = schmidt_coefficients(state, cut)
    p = coeffs ** 2
    p = p[p > _EPS]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def concurrence(state) -> float:
    """Concurrence of a 2-qubit state (0 = separable, 1 = maximally entangled).

    `C = |<psi| (Y (x) Y) |psi*>|` for a pure state.
    """
    if isinstance(state, QuantumState):
        psi = state.state
    else:
        psi = np.asarray(state, dtype=complex).reshape(-1)
    if psi.size != 4:
        raise ValueError("concurrence is defined here for 2-qubit states")
    y_y = np.kron(Y, Y)
    return float(abs(psi.conj() @ (y_y @ psi.conj())))


# ===========================================================================
# 7. Famous states
# ===========================================================================
def bell_state(index: int = 0) -> QuantumState:
    """One of the four Bell (maximally entangled) states.

    index 0: (|00>+|11>)/sqrt2
    index 1: (|00>-|11>)/sqrt2
    index 2: (|01>+|10>)/sqrt2
    index 3: (|01>-|10>)/sqrt2
    """
    if index == 0:
        v = np.array([1, 0, 0, 1], dtype=complex)
    elif index == 1:
        v = np.array([1, 0, 0, -1], dtype=complex)
    elif index == 2:
        v = np.array([0, 1, 1, 0], dtype=complex)
    elif index == 3:
        v = np.array([0, 1, -1, 0], dtype=complex)
    else:
        raise ValueError("Bell state index must be 0..3")
    return QuantumState(v / sqrt(2), 2)


def ghz_state(n_qubits: int = 3) -> QuantumState:
    """GHZ state `(|00...0> + |11...1>)/sqrt2` on `n_qubits`."""
    v = np.zeros(1 << n_qubits, dtype=complex)
    v[0] = 1 / sqrt(2)
    v[-1] = 1 / sqrt(2)
    return QuantumState(v, n_qubits)


def w_state(n_qubits: int = 3) -> QuantumState:
    """W state: equal superposition of all single-excitation basis states."""
    v = np.zeros(1 << n_qubits, dtype=complex)
    for q in range(n_qubits):
        v[1 << (n_qubits - 1 - q)] = 1 / sqrt(n_qubits)
    return QuantumState(v, n_qubits)


# ===========================================================================
# 8. Circuits
# ===========================================================================
class QuantumCircuit:
    """A simple gate-sequence circuit over `n_qubits`.

    Example
    -------
    >>> c = QuantumCircuit(2)
    >>> c.h(0).cnot(0, 1)
    >>> c.state           # Bell state
    >>> c.measure_all()   # sampling
    """

    def __init__(self, n_qubits: int = 1):
        self.n = n_qubits
        self.state = QuantumState.zeros(n_qubits)
        self.ops: list = []

    # -- gate application (chainable) -------------------------------------
    def _record(self, name, *args):
        self.ops.append((name, args))
        return self

    def apply(self, op: np.ndarray, label: str = "U") -> "QuantumCircuit":
        self.state = self.state.apply(op)
        return self._record(label, "matrix")

    def single(self, gate_op, target: int, label: str = "U") -> "QuantumCircuit":
        self.state = self.state.apply_gate(gate_op, target)
        return self._record(label, target)

    def h(self, t: int) -> "QuantumCircuit":
        return self.single(H, t, "H")

    def x(self, t: int) -> "QuantumCircuit":
        return self.single(X, t, "X")

    def y(self, t: int) -> "QuantumCircuit":
        return self.single(Y, t, "Y")

    def z(self, t: int) -> "QuantumCircuit":
        return self.single(Z, t, "Z")

    def s(self, t: int) -> "QuantumCircuit":
        return self.single(S, t, "S")

    def t(self, t: int) -> "QuantumCircuit":
        return self.single(T, t, "T")

    def rx(self, theta: float, t: int) -> "QuantumCircuit":
        return self.single(RX(theta), t, "RX")

    def ry(self, theta: float, t: int) -> "QuantumCircuit":
        return self.single(RY(theta), t, "RY")

    def rz(self, theta: float, t: int) -> "QuantumCircuit":
        return self.single(RZ(theta), t, "RZ")

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        self.state = self.state.cnot(control, target)
        return self._record("CNOT", control, target)

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        self.state = self.state.apply_controlled(Z, control, target)
        return self._record("CZ", control, target)

    def swap(self, a: int, b: int) -> "QuantumCircuit":
        self.state = self.state.cnot(a, b)
        self.state = self.state.cnot(b, a)
        self.state = self.state.cnot(a, b)
        return self._record("SWAP", a, b)

    def toffoli(self, c1: int, c2: int, target: int) -> "QuantumCircuit":
        op = controlled(controlled(X, 3, 0, 2), self.n, c1, c2) \
            if self.n == 3 else None
        if self.n != 3:
            raise NotImplementedError("toffoli helper supports n=3 registers")
        self.state = self.state.apply(TOFFOLI)
        return self._record("TOFFOLI", c1, c2, target)

    # -- measurement -------------------------------------------------------
    def measure_all(self, n_shots: int = 1024,
                    rng: np.random.Generator | None = None) -> dict:
        """Sample the circuit in the computational basis.

        Returns `{bitstring: count}`. Does not collapse `self.state`.
        """
        rng = rng or np.random.default_rng()
        probs = self.state.probabilities()
        probs = probs / probs.sum()
        idxs = rng.choice(self.state.dim, size=n_shots, p=probs)
        counts: dict = {}
        for i in idxs:
            bs = self.state.bitstring(int(i))
            counts[bs] = counts.get(bs, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def expectation(self, op: np.ndarray) -> complex:
        """`<psi| op |psi>` for the current circuit state."""
        return expectation(self.state, op)

    def __repr__(self):
        return f"QuantumCircuit(n={self.n}, ops={[o[0] for o in self.ops]})"


# ===========================================================================
# 9. Algorithms / constructions
# ===========================================================================
def teleportation_circuit(alpha=0.6, beta=None) -> dict:
    """Quantum teleportation of an arbitrary single-qubit state.

    Steps:
      1. Prepare `|psi> = alpha|0> + beta|1>` on qubit 0.
      2. Create a Bell pair on qubits 1,2.
      3. Bell-basis measurement on qubits 0,1 -> 2 classical bits.
      4. Apply the corresponding Pauli correction to qubit 2.

    Returns a dict with the state to send, the two measurement outcomes, the
    correction applied, and the final (recovered) qubit state.
    """
    if beta is None:
        beta = cmath.sqrt(1 - abs(alpha) ** 2) if abs(alpha) <= 1 else 0.0
        beta = complex(beta)
    psi = Qubit(alpha, beta)

    # Register: q0 = payload, q1,q2 = Bell pair
    state = QuantumState.from_qubits(psi, Qubit.zero(), Qubit.zero())
    state = state.apply_controlled(X, 1, 2)          # Bell pair on 1,2
    state = state.apply_gate(H, 1)                    # actually H on q1 first
    # Rebuild properly: H(q1) then CNOT(q1,q2)
    state = QuantumState.from_qubits(psi, Qubit.zero(), Qubit.zero())
    state = state.apply_gate(H, 1)
    state = state.apply_controlled(X, 1, 2)

    # Bell measurement on q0,q1
    state = state.cnot(0, 1)
    state = state.apply_gate(H, 0)
    m0 = state.measure(0)
    m1 = state.measure(1)

    # Corrections on q2: X if m1, Z if m0
    if m1:
        state = state.apply_gate(X, 2)
    if m0:
        state = state.apply_gate(Z, 2)

    # Extract q2's reduced state
    rho = state.reduced_density_matrix(2)
    return {
        "sent": psi.state,
        "m0": m0,
        "m1": m1,
        "correction": ("X" if m1 else "") + ("Z" if m0 else "") or "I",
        "received_density": rho,
        "fidelity": float(np.real(psi.state.conj() @ rho @ psi.state)),
    }


def phase_oracle(n_qubits: int, marked: int) -> np.ndarray:
    """Oracle that flips the phase of basis state `marked` (0 -> negative)."""
    dim = 1 << n_qubits
    diag = np.ones(dim, dtype=complex)
    diag[marked] = -1.0
    return np.diag(diag)


def deutsch_jozsa(n_qubits: int, constant: bool = True,
                  balanced_pattern: int = 1) -> dict:
    """Deutsch-Jozsa algorithm (constant vs. balanced oracle).

    Deterministic after ONE quantum query: all-zero outcome => constant,
    otherwise balanced.

    Returns `{'outcome': bitstring, 'verdict': 'constant'|'balanced'}`.
    """
    total = n_qubits + 1
    # Input register in |+>^n, ancilla in |->
    state = QuantumState.zeros(total)
    for q in range(n_qubits):
        state = state.apply_gate(H, q)
    state = state.apply_gate(X, n_qubits)     # ancilla -> |1>
    state = state.apply_gate(H, n_qubits)     # ancilla -> |->

    dim_in = 1 << n_qubits
    if constant:
        phases = np.ones(dim_in, dtype=complex)          # f(x)=0 -> no phase
    else:
        # balanced: f(x) = parity of a fixed mask
        phases = np.array([(-1) ** (bin(i & balanced_pattern).count("1"))
                           for i in range(dim_in)], dtype=complex)

    # Oracle acts as |x>|-> -> (-1)^{f(x)} |x>|->: apply diag on input, I on ancilla
    oracle = np.kron(np.diag(phases), I)
    state = state.apply(oracle)

    for q in range(n_qubits):
        state = state.apply_gate(H, q)

    probs = state.probabilities()
    probs_flat = probs.reshape(dim_in, 2).sum(axis=1)
    outcome_index = int(np.argmax(probs_flat))
    verdict = "constant" if outcome_index == 0 else "balanced"
    return {"outcome": format(outcome_index, f"0{n_qubits}b"),
            "verdict": verdict,
            "expected": "constant" if constant else "balanced",
            "probabilities": probs_flat}


# ===========================================================================
# 10. Reversible classical logic -- the bridge to nn.gates
# ===========================================================================
# Quantum gates are UNITARY, hence reversible: they cannot discard information.
# Classical AND/OR discard bits (2 inputs -> 1 output), so we embed them into
# reversible gates that keep the inputs and write the result to an ancilla:
#
#     AND(a, b) :  (a, b, c) -> (a, b, c XOR (a AND b))
#     OR (a, b) :  (a, b, c) -> (a, b, c XOR (a OR b))
#     NAND(a,b) :  (a, b, c) -> (a, b, c XOR NOT(a AND b))   (universal)
#     NOT(a)    :  (a, c)    -> (a, c XOR NOT a)
#     XOR(a, b) :  (a, b, c) -> (a, b, c XOR (a XOR b))      ( = CNOT chain )
#
# All of these are SINGLE unitary operators, so each is exactly realizable by a
# quantum circuit. This mirrors docs/07_gate_logic.md section 6: a logic gate is
# a neuron; here a logic gate is a unitary.


def _classical_truth(fn, n_in: int):
    """Return a list of output bits for a classical function over all rows."""
    return [int(fn(*row)) for row in itertools.product([0, 1], repeat=n_in)]


def classical_gate_unitary(fn, n_inputs: int) -> np.ndarray:
    """Build a reversible unitary for `f: B^n -> B` with one ancilla output.

    The operator acts on `n_inputs + 1` qubits as

        |x>|y>  ->  |x>|y XOR f(x)>

    which is always a permutation matrix (hence unitary).
    """
    n = n_inputs + 1
    dim = 1 << n
    op = np.zeros((dim, dim), dtype=complex)
    for x in range(1 << n_inputs):
        f_val = int(fn(*[(x >> (n_inputs - 1 - i)) & 1 for i in range(n_inputs)]))
        for y in (0, 1):
            src = (x << 1) | y
            dst = (x << 1) | (y ^ f_val)
            op[dst, src] = 1.0
    return op


def _two_input(fn):
    return classical_gate_unitary(fn, 2)


# Reversible (unitary) realizations of the classical gates.
reversible_and = _two_input(lambda a, b: a & b)
reversible_or = _two_input(lambda a, b: a | b)
reversible_xor = _two_input(lambda a, b: a ^ b)
reversible_nand = _two_input(lambda a, b: 1 - (a & b))
reversible_nor = _two_input(lambda a, b: 1 - (a | b))
reversible_not = classical_gate_unitary(lambda a: 1 - a, 1)

REVERSIBLE_GATES = {
    "AND": reversible_and,
    "OR": reversible_or,
    "XOR": reversible_xor,
    "NAND": reversible_nand,
    "NOR": reversible_nor,
    "NOT": reversible_not,
}


def classical_to_quantum(gate_name: str, a: int, b: int, ancilla: int = 0):
    """Apply a classical gate reversibly on a quantum register.

    Register layout (2 input qubits + 1 ancilla):
        q0 = a, q1 = b, q2 = ancilla holding `y XOR gate(a, b)`.

    Returns the resulting `QuantumState` (basis state, since inputs are
    computational-basis bits).
    """
    key = gate_name.upper()
    if key not in REVERSIBLE_GATES:
        raise KeyError(f"Unknown reversible gate '{gate_name}'. "
                       f"Available: {sorted(REVERSIBLE_GATES)}")
    op = REVERSIBLE_GATES[key]
    n = 3 if key != "NOT" else 2
    bits = [a, b, ancilla] if n == 3 else [a, ancilla]
    state = QuantumState.from_bits(bits[:n])
    out = state.apply(op)
    return out


def gate_from_unitary(op: np.ndarray, n_qubits: int) -> list:
    """Extract the classical truth table realized by a permutation unitary.

    Returns a list of `(input_bits, output_bits)` for every basis state whose
    image is a single basis state (i.e. a classical, reversible map).
    """
    table = []
    for i in range(1 << n_qubits):
        col = op[:, i]
        nz = np.nonzero(np.abs(col) > 1e-9)[0]
        if nz.size == 1:
            in_bits = tuple((i >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits))
            j = int(nz[0])
            out_bits = tuple((j >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits))
            table.append((in_bits, out_bits))
    return table
