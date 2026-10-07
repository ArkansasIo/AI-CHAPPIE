"""
nn.gates -- the mathematics of gate logic, with derivatives.

This module implements, in pure NumPy, everything in docs/07_gate_logic.md:

Boolean algebra
---------------
* hard gates            : `AND`, `OR`, `NOT`, `NAND`, `NOR`, `XOR`, `XNOR`, ...
* the 16 two-input functions and their truth tables
* canonical forms       : `minterms`, `maxterms`, `sop`, `pos`, `anf` (Zhegalkin)
* identities            : De Morgan, absorption, distributivity (verifiable)
* completeness helpers  : represent any gate using only NAND (or NOR)

Differentiable (smooth) logic -- the calculus bridge
----------------------------------------------------
* `soft_and`, `soft_or`, `soft_not`, `soft_xor`, `soft_nand`, ...
  each with a temperature `tau`; as `tau -> 0` they converge to the hard gate.
* analytic gradients of every soft gate (for backprop through a circuit)
* `straight_through` : hard forward value with a smooth backward gradient (STE)

Learning gates
--------------
* `GATE_TABLE`         : names -> truth-table bit patterns
* `gate_dataset`       : build (X, Y) training data for any gate
* `learn_gate`         : train a small MLP to realize a gate (XOR needs a
                         hidden layer; AND/OR/NAND learn with one neuron)

Everything here is NumPy-only and parallels the rest of the library.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    # hard gates
    "AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR",
    "IMPLIES", "NIMPLIES", "TRUE", "FALSE", "BUFFER",
    "hard_gate", "GATE_TABLE", "ALL_BINARY_GATES", "truth_table",
    # canonical forms
    "minterms", "maxterms", "sop", "pos", "anf", "evaluate_anf",
    # identities / completeness
    "de_morgan_check", "is_complete_nand", "to_nand", "to_nor",
    # smooth logic
    "sigmoid", "soft_gate", "soft_and", "soft_or", "soft_not",
    "soft_nand", "soft_nor", "soft_xor", "soft_xnor",
    "soft_gate_grad", "soft_gate_value_and_grad",
    "straight_through",
    # learning
    "gate_dataset", "learn_gate",
]


# ===========================================================================
# 1. Hard Boolean gates
# ===========================================================================
def _as_bool(x):
    """Coerce to a 0/1 integer array."""
    return (np.asarray(x) != 0).astype(int)


def AND(a, b):
    """Conjunction: `a ∧ b` (also equals `a * b` for bits)."""
    return _as_bool(a) & _as_bool(b)


def OR(a, b):
    """Disjunction: `a ∨ b`."""
    return _as_bool(a) | _as_bool(b)


def NOT(a):
    """Negation: `¬a`  (arithmetically `1 - a`)."""
    return 1 - _as_bool(a)


def NAND(a, b):
    """`¬(a ∧ b)` -- the universal gate."""
    return NOT(AND(a, b))


def NOR(a, b):
    """`¬(a ∨ b)` -- also universal."""
    return NOT(OR(a, b))


def XOR(a, b):
    """Exclusive or: `a ⊕ b` = addition mod 2 = `(a + b) mod 2`."""
    return (_as_bool(a) ^ _as_bool(b))


def XNOR(a, b):
    """`¬(a ⊕ b)` -- equality."""
    return NOT(XOR(a, b))


def IMPLIES(a, b):
    """`a → b`  =  `¬a ∨ b`."""
    return OR(NOT(a), b)


def NIMPLIES(a, b):
    """`a ↛ b` = `a ∧ ¬b`."""
    return AND(a, NOT(b))


def TRUE(a, b):
    """Constant 1 (tautology)."""
    return np.ones_like(_as_bool(a))


def FALSE(a, b):
    """Constant 0 (contradiction)."""
    return np.zeros_like(_as_bool(a))


def BUFFER(a, b=None):
    """Identity (1-input buffer); ignores the second argument."""
    return _as_bool(a)


# ---------------------------------------------------------------------------
# Registry of gates, including the full 16 two-input truth-table functions.
# Each value is the 4-bit output row order [(0,0), (0,1), (1,0), (1,1)].
# ---------------------------------------------------------------------------
GATE_TABLE = {
    "FALSE":   (0, 0, 0, 0),   # 0b0000
    "AND":     (0, 0, 0, 1),
    "A_AND_NOTB": (0, 0, 1, 0),
    "A":       (0, 0, 1, 1),
    "NOTA_AND_B": (0, 1, 0, 0),
    "B":       (0, 1, 0, 1),
    "XOR":     (0, 1, 1, 0),
    "OR":      (0, 1, 1, 1),
    "NOR":     (1, 0, 0, 0),
    "XNOR":    (1, 0, 0, 1),
    "NOTB":    (1, 0, 1, 0),
    "A_OR_NOTB": (1, 0, 1, 1),
    "NOTA":    (1, 1, 0, 0),
    "NOTA_OR_B": (1, 1, 0, 1),
    "NAND":    (1, 1, 1, 0),
    "TRUE":    (1, 1, 1, 1),
}

# All 16 possible 2-input Boolean functions, keyed by their index 0..15.
ALL_BINARY_GATES = {i: tuple((i >> (3 - r)) & 1 for r in range(4)) for i in range(16)}

_HARD_IMPL = {
    "FALSE": FALSE, "AND": AND, "OR": OR, "NAND": NAND, "NOR": NOR,
    "XOR": XOR, "XNOR": XNOR, "NOTA": lambda a, b: NOT(a), "NOTB": lambda a, b: NOT(b),
    "A": lambda a, b: _as_bool(a), "B": lambda a, b: _as_bool(b), "TRUE": TRUE,
    "A_AND_NOTB": NIMPLIES, "A_OR_NOTB": lambda a, b: OR(a, NOT(b)),
    "NOTA_AND_B": lambda a, b: AND(NOT(a), b), "NOTA_OR_B": lambda a, b: OR(NOT(a), b),
}


def hard_gate(name: str, a, b=None):
    """Evaluate a named hard gate on inputs `a`, `b`."""
    if name not in _HARD_IMPL:
        raise KeyError(f"Unknown gate '{name}'. Available: {sorted(_HARD_IMPL)}")
    return _HARD_IMPL[name](a, b)


def truth_table(gate, n_inputs: int = 2) -> np.ndarray:
    """Return the full truth table of `gate` over `n_inputs` variables.

    Parameters
    ----------
    gate : str (name in GATE_TABLE) or callable(a, b, ...) -> 0/1
    n_inputs : number of Boolean variables

    Returns
    -------
    array of shape `(2**n, n + 1)`: the first `n` columns are inputs and the
    last column is the output.
    """
    rows = list(itertools.product([0, 1], repeat=n_inputs))
    out = []
    for r in rows:
        if isinstance(gate, str):
            value = int(hard_gate(gate, *r))
        else:
            value = int(gate(*r))
        out.append((*r, value))
    return np.array(out, dtype=int)


# ===========================================================================
# 2. Canonical forms
# ===========================================================================
def minterms(outputs) -> list:
    """Indices of the rows where `outputs == 1` (the on-set)."""
    return [i for i, v in enumerate(np.asarray(outputs).ravel()) if v == 1]


def maxterms(outputs) -> list:
    """Indices of the rows where `outputs == 0` (the off-set)."""
    return [i for i, v in enumerate(np.asarray(outputs).ravel()) if v == 0]


def _term_for_row(row_index: int, n: int, polarity: int) -> str:
    """Build a product (SOP) or sum (POS) term for a row index."""
    bits = [(row_index >> (n - 1 - i)) & 1 for i in range(n)]
    names = [chr(ord("a") + i) for i in range(n)]
    if polarity == 1:      # minterm: 1 -> var, 0 -> NOT var ; joined by AND
        parts = [names[i] if bits[i] else f"¬{names[i]}" for i in range(n)]
        return "(" + "∧".join(parts) + ")"
    else:                  # maxterm: 0 -> var, 1 -> NOT var ; joined by OR
        parts = [names[i] if bits[i] == 0 else f"¬{names[i]}" for i in range(n)]
        return "(" + "∨".join(parts) + ")"


def sop(outputs, n_inputs: int = 2) -> str:
    """Sum-of-products (minterm) canonical form as a string."""
    idx = minterms(outputs)
    if not idx:
        return "0  (FALSE)"
    return " ∨ ".join(_term_for_row(i, n_inputs, 1) for i in idx)


def pos(outputs, n_inputs: int = 2) -> str:
    """Product-of-sums (maxterm) canonical form as a string."""
    idx = maxterms(outputs)
    if not idx:
        return "1  (TRUE)"
    return " ∧ ".join(_term_for_row(i, n_inputs, 0) for i in idx)


def anf(outputs, n_inputs: int = 2):
    """Algebraic Normal Form (Zhegalkin polynomial) over GF(2).

    Returns a list of `(coefficient, monomial)` pairs where a monomial is a
    tuple of variable indices being AND-ed. Uses the Möbius transform over
    GF(2), so it is exact for any function.

    Example
    -------
    >>> anf(GATE_TABLE['XOR'])
    [(0, ()), (1, (0,)), (1, (1,))]      # a ⊕ b
    >>> anf(GATE_TABLE['AND'])
    [(0, ()), (0, (0,)), (0, (1,)), (1, (0, 1))]   # ab
    """
    f = list(np.asarray(outputs).ravel().astype(int))
    n = 1 << n_inputs
    # Möbius transform (in-place) over the hypercube.
    coeff = f[:]
    step = 1
    while step < n:
        for i in range(n):
            if i & step:
                coeff[i] ^= coeff[i ^ step]
        step <<= 1
    terms = []
    for i, c in enumerate(coeff):
        if c:
            monomial = tuple(j for j in range(n_inputs) if (i >> j) & 1)
            terms.append((1, monomial))
    if not terms:
        terms = [(0, ())]   # constant zero
    return terms


def evaluate_anf(terms, inputs) -> int:
    """Evaluate an ANF (from `anf`) on a single Boolean input tuple."""
    acc = 0
    for coef, monomial in terms:
        if coef == 0:
            continue
        val = 1
        for j in monomial:
            val &= int(inputs[j])
        acc ^= (coef & val)
    return acc


# ===========================================================================
# 3. Identities and functional completeness
# ===========================================================================
def de_morgan_check(n_inputs: int = 2) -> bool:
    """Verify `¬(a∧b) = ¬a∨¬b` and `¬(a∨b) = ¬a∧¬b` for all input rows."""
    for r in itertools.product([0, 1], repeat=n_inputs):
        a, b = r[0], r[1] if n_inputs > 1 else r[0]
        if NOT(AND(a, b)) != OR(NOT(a), NOT(b)):
            return False
        if NOT(OR(a, b)) != AND(NOT(a), NOT(b)):
            return False
    return True


def to_nand(gate_name: str, a, b=None):
    """Realize a gate using ONLY NAND gates, returning the result.

    Uses:
        NOT x  = x NAND x
        x∧y    = (x NAND y) NAND (x NAND y)
        x∨y    = (x NAND x) NAND (y NAND y)
        x⊕y    = (x NAND (x NAND y)) NAND (y NAND (x NAND y))
    """
    a = _as_bool(a)
    if b is None:
        b = a
    b = _as_bool(b)

    not_a = NAND(a, a)
    not_b = NAND(b, b)
    and_ab = NAND(NAND(a, b), NAND(a, b))
    or_ab = NAND(not_a, not_b)
    xor_ab = NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b)))

    table = {
        "NOT": not_a, "AND": and_ab, "OR": or_ab, "NAND": NAND(a, b),
        "NOR": NAND(or_ab, or_ab), "XOR": xor_ab, "XNOR": NAND(xor_ab, xor_ab),
    }
    if gate_name not in table:
        raise KeyError(f"to_nand supports {sorted(table)}")
    return table[gate_name]


def to_nor(gate_name: str, a, b=None):
    """Realize a gate using ONLY NOR gates (dual of `to_nand`)."""
    a = _as_bool(a)
    if b is None:
        b = a
    b = _as_bool(b)

    not_a = NOR(a, a)
    not_b = NOR(b, b)
    or_ab = NOR(NOR(a, b), NOR(a, b))
    and_ab = NOR(not_a, not_b)
    xor_ab = NOR(NOR(a, NOR(a, b)), NOR(b, NOR(a, b)))

    table = {
        "NOT": not_a, "AND": and_ab, "OR": or_ab, "NOR": NOR(a, b),
        "NAND": NOR(and_ab, and_ab), "XOR": xor_ab, "XNOR": NOR(xor_ab, xor_ab),
    }
    if gate_name not in table:
        raise KeyError(f"to_nor supports {sorted(table)}")
    return table[gate_name]


def is_complete_nand(verbose: bool = False) -> bool:
    """Empirically verify that NAND alone reproduces AND, OR and NOT.

    (The theoretical proof is in docs/07_gate_logic.md §3.1.)
    """
    ok = True
    for r in itertools.product([0, 1], repeat=2):
        a, b = r
        checks = {
            "NOT": (to_nand("NOT", a), NOT(a)),
            "AND": (to_nand("AND", a, b), AND(a, b)),
            "OR": (to_nand("OR", a, b), OR(a, b)),
            "XOR": (to_nand("XOR", a, b), XOR(a, b)),
        }
        for name, (got, want) in checks.items():
            if int(got) != int(want):
                ok = False
                if verbose:
                    print(f"  mismatch {name} on {(a, b)}: nand={got} want={want}")
    return ok


# ===========================================================================
# 4. Smooth / differentiable logic (the calculus bridge)
# ===========================================================================
def sigmoid(z):
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


_SOFT_BIAS = {
    # (low_bias, high_bias) such that sigmoid((a + b - bias)/tau) gives the gate
    "AND": 1.5,
    "OR": 0.5,
    "NAND": 0.5,      # NOTE handled with a final NOT below where needed
    "NOR": 1.5,
}

_SOFT_IMPL = {
    "AND": lambda a, b, t: sigmoid((a + b - 1.5) / t),
    "OR": lambda a, b, t: sigmoid((a + b - 0.5) / t),
    "NOT": lambda a, b, t: 1.0 - np.asarray(a, dtype=float),
    "NAND": lambda a, b, t: 1.0 - sigmoid((a + b - 1.5) / t),
    "NOR": lambda a, b, t: 1.0 - sigmoid((a + b - 0.5) / t),
    # Algebraic relaxations using a⊕b = a + b - 2ab at {0,1}.
    "XOR": lambda a, b, t: sigmoid(((np.asarray(a, float) + np.asarray(b, float)
                                     - 2 * np.asarray(a, float) * np.asarray(b, float))
                                    - 0.5) / t),
    "XNOR": lambda a, b, t: sigmoid((1.0 - (np.asarray(a, float) + np.asarray(b, float)
                                            - 2 * np.asarray(a, float) * np.asarray(b, float))
                                     - 0.5) / t),
}


def soft_and(a, b, tau: float = 0.1):
    """Smooth AND: `sigmoid((a + b - 1.5)/tau)`; approaches `a∧b` as `tau→0`."""
    return _SOFT_IMPL["AND"](a, b, tau)


def soft_or(a, b, tau: float = 0.1):
    """Smooth OR: `sigmoid((a + b - 0.5)/tau)`."""
    return _SOFT_IMPL["OR"](a, b, tau)


def soft_not(a, tau: float = 0.1):
    """Smooth NOT (exact): `1 - a`."""
    return _SOFT_IMPL["NOT"](a, b=None, t=tau)


def soft_nand(a, b, tau: float = 0.1):
    """Smooth NAND: `1 - sigmoid((a + b - 1.5)/tau)`."""
    return _SOFT_IMPL["NAND"](a, b, tau)


def soft_nor(a, b, tau: float = 0.1):
    """Smooth NOR: `1 - sigmoid((a + b - 0.5)/tau)`."""
    return _SOFT_IMPL["NOR"](a, b, tau)


def soft_xor(a, b, tau: float = 0.1):
    """Smooth XOR via `a ⊕ b = a + b - 2ab` (the ANF parity form)."""
    return _SOFT_IMPL["XOR"](a, b, tau)


def soft_xnor(a, b, tau: float = 0.1):
    """Smooth XNOR (equality)."""
    return _SOFT_IMPL["XNOR"](a, b, tau)


def soft_gate(name: str, a, b=None, tau: float = 0.1):
    """Evaluate a named gate in smooth form."""
    key = name.upper()
    if key not in _SOFT_IMPL:
        raise KeyError(f"No smooth form for '{name}'. "
                       f"Available: {sorted(_SOFT_IMPL)}")
    return _SOFT_IMPL[key](a, b, tau)


def soft_gate_grad(name: str, a, b, tau: float = 0.1):
    """Analytic partial derivatives `(d/da, d/db)` of a smooth gate.

    These are the local chain-rule factors you multiply by the upstream
    gradient when backpropagating through a logic circuit (see
    docs/07_gate_logic.md §4.3). They are derived by differentiating the
    sigmoid compositions; `d sigmoid(z)/dz = sigmoid(z)(1-sigmoid(z))`.
    """
    key = name.upper()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b if b is not None else a, dtype=float)
    t = tau

    def dsig(z):
        s = sigmoid(z)
        return s * (1.0 - s) / t

    if key == "NOT":
        return -np.ones_like(a), np.zeros_like(b)
    if key == "AND":
        g = dsig((a + b - 1.5) / t)
        return g, g
    if key == "OR":
        g = dsig((a + b - 0.5) / t)
        return g, g
    if key == "NAND":
        g = dsig((a + b - 1.5) / t)
        return -g, -g
    if key == "NOR":
        g = dsig((a + b - 0.5) / t)
        return -g, -g
    if key in ("XOR", "XNOR"):
        # z = (a + b - 2ab - 0.5)/t   (XNOR mirrors the sign below)
        dz_da = (1.0 - 2.0 * b) / t
        dz_db = (1.0 - 2.0 * a) / t
        z = (a + b - 2.0 * a * b - 0.5) / t
        g = sigmoid(z) * (1.0 - sigmoid(z))     # d sigmoid/dz already has 1/t folded in dz
        sign = 1.0 if key == "XOR" else -1.0
        return sign * g * dz_da, sign * g * dz_db
    raise KeyError(f"No gradient for '{name}'.")


def soft_gate_value_and_grad(name: str, a, b=None, tau: float = 0.1):
    """Return `(value, (da, db))` for a smooth gate."""
    return soft_gate(name, a, b, tau), soft_gate_grad(name, a, b, tau)


def straight_through(z, tau: float = 1.0):
    """Straight-through estimator (STE) for a hard gate.

    Forward : returns the HARD value `1[z > 0]`.
    Backward: callers should use `sigmoid(z/tau)(1-sigmoid(z/tau))/tau` as the
              surrogate gradient (see `soft_gate_grad`).

    Returns
    -------
    (hard_value, surrogate_grad)
        `hard_value` is your forward value; `surrogate_grad` is what you plug
        into the backward pass.
    """
    z = np.asarray(z, dtype=float)
    hard = (z > 0).astype(float)
    s = sigmoid(z / tau)
    surrogate = s * (1.0 - s) / tau
    return hard, surrogate


# ===========================================================================
# 5. Learning gates with a neural network
# ===========================================================================
def gate_dataset(name: str, n_inputs: int = 2):
    """Build `(X, Y)` training data (one-hot-ish bits) for a named gate.

    Returns
    -------
    X : (2**n, n) float inputs in {0,1}
    Y : (2**n,) int labels in the gate's output domain
    """
    tt = truth_table(name, n_inputs)
    X = tt[:, :n_inputs].astype(float)
    Y = tt[:, n_inputs].astype(int)
    return X, Y


def learn_gate(name: str, hidden: int = 0, epochs: int = 2000, lr: float = 0.1,
               seed: int = 0, verbose: bool = False, one_output: bool = False,
               restarts: int = 5):
    """Train an MLP to realize a Boolean gate; return `(net, accuracy)`.

    Parameters
    ----------
    name      : gate name (see GATE_TABLE)
    hidden    : hidden width; `0` means a single linear neuron (works for
                AND/OR/NAND/NOR, fails for XOR/XNOR)
    one_output: if True, use a 1-logit sigmoid output (`sigmoid_bce` loss);
                else (default) a 2-logit softmax output (`softmax_ce` loss),
                which is far more robust to the all-zero local minimum.
    restarts  : number of random restarts to try; the best model is returned.
                This makes gate learning reliable regardless of seed.

    Returns
    -------
    (net, accuracy) where `net` is an `nn.model.Sequential`.
    """
    from .layers import Dense, Activation
    from .losses import SoftmaxCrossEntropy, SigmoidBinaryCrossEntropy
    from .model import Sequential
    from .optimizers import Adam
    from .utils import set_seed

    X, Y = gate_dataset(name)
    n_in = X.shape[1]

    def build(seed_offset: int):
        set_seed(seed + seed_offset)
        if hidden and hidden > 0:
            if one_output:
                net = Sequential(loss=SigmoidBinaryCrossEntropy())
                net.add(Dense(n_in, hidden, init="xavier_normal"))
                net.add(Activation("tanh"))
                net.add(Dense(hidden, 1, init="xavier_normal"))
                net.add(Activation("sigmoid"))
                y_train = Y.reshape(-1, 1).astype(float)
            else:
                net = Sequential(loss=SoftmaxCrossEntropy())
                net.add(Dense(n_in, hidden, init="xavier_normal"))
                net.add(Activation("tanh"))
                net.add(Dense(hidden, 2, init="xavier_normal"))
                y_train = Y
        else:
            # Single neuron (no hidden layer) -- only separable gates work.
            if one_output:
                net = Sequential(loss=SigmoidBinaryCrossEntropy())
                net.add(Dense(n_in, 1, init="xavier_normal"))
                net.add(Activation("sigmoid"))
                y_train = Y.reshape(-1, 1).astype(float)
            else:
                net = Sequential(loss=SoftmaxCrossEntropy())
                net.add(Dense(n_in, 2, init="xavier_normal"))
                y_train = Y
        return net, y_train

    def evaluate(net, y_train):
        scores = net.predict(X)
        if one_output:
            preds = (scores.reshape(-1) >= 0.5).astype(int)
        else:
            preds = np.argmax(scores, axis=1)
        return preds, float(np.mean(preds == Y))

    best_net, best_acc, best_preds = None, -1.0, None
    for attempt in range(max(1, restarts)):
        net, y_train = build(attempt)
        opt = Adam(lr=lr)
        for _ in range(epochs):
            net.train_step(X, y_train, opt)
        preds, acc = evaluate(net, y_train)
        if acc > best_acc:
            best_net, best_acc, best_preds = net, acc, preds
        if best_acc >= 1.0:      # perfect -- stop early
            break

    if verbose:
        print(f"  learn_gate({name}, hidden={hidden}) -> acc {best_acc:.2f} "
              f"preds {best_preds} target {Y}")
    return best_net, best_acc
