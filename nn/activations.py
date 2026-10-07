"""
nn.activations -- nonlinearities and their analytic derivatives.

Each activation is exposed as a pair `(f, f')` where `f'` is computed *from the
pre-activation `z`* (forward form) unless noted. Some functions cache values
internally for speed; the `_from_output` variants take the activation output.

Derivatives are documented in docs/03_calculus.md, section 5.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "linear", "linear_prime",
    "relu", "relu_prime",
    "leaky_relu", "leaky_relu_prime",
    "elu", "elu_prime",
    "sigmoid", "sigmoid_prime",
    "tanh", "tanh_prime",
    "softplus", "softplus_prime",
    "swish", "swish_prime",
    "gelu", "gelu_prime",
    "ACTIVATIONS",
    "get_activation",
    "get_activation_grad",
    "get_activation_from_output",
]


# ---------------------------------------------------------------------------
# Linear (identity)
# ---------------------------------------------------------------------------
def linear(z):
    """Identity activation."""
    return np.asarray(z, dtype=float)


def linear_prime(z):
    """Derivative of identity = 1 everywhere."""
    return np.ones_like(np.asarray(z, dtype=float))


# ---------------------------------------------------------------------------
# ReLU family
# ---------------------------------------------------------------------------
def relu(z):
    """Rectified linear unit: `max(0, z)`."""
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_prime(z):
    """Subgradient: 1 where z>0, else 0."""
    return (np.asarray(z, dtype=float) > 0).astype(float)


def leaky_relu(z, alpha: float = 0.01):
    """Leaky ReLU: `z` if `z>0` else `alpha*z` (avoids dead units)."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, z, alpha * z)


def leaky_relu_prime(z, alpha: float = 0.01):
    """1 where z>0, else alpha."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, 1.0, alpha)


def elu(z, alpha: float = 1.0):
    """Exponential linear unit."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, z, alpha * (np.exp(np.minimum(z, 0.0)) - 1.0))


def elu_prime(z, alpha: float = 1.0):
    """1 where z>0, else `alpha * e^z`."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, 1.0, alpha * np.exp(np.minimum(z, 0.0)))


# ---------------------------------------------------------------------------
# Sigmoid / tanh
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Logistic sigmoid, stable for large |z|."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_prime(z):
    """`sigma'(z) = sigma(z)(1 - sigma(z))`."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    """Hyperbolic tangent."""
    return np.tanh(np.asarray(z, dtype=float))


def tanh_prime(z):
    """`1 - tanh^2(z)`."""
    t = np.tanh(np.asarray(z, dtype=float))
    return 1.0 - t * t


# ---------------------------------------------------------------------------
# Smooth / modern
# ---------------------------------------------------------------------------
def softplus(z):
    """`ln(1 + e^z)`, numerically stable."""
    z = np.asarray(z, dtype=float)
    # max(z,0) + log1p(exp(-|z|)) is stable for both signs.
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def softplus_prime(z):
    """Derivative of softplus = sigmoid(z)."""
    return sigmoid(z)


def swish(z, beta: float = 1.0):
    """Swish / SiLU: `z * sigmoid(beta*z)`."""
    z = np.asarray(z, dtype=float)
    return z * sigmoid(beta * z)


def swish_prime(z, beta: float = 1.0):
    """Derivative of swish: `s + beta*z*s*(1-s)` with `s = sigmoid(beta*z)`."""
    z = np.asarray(z, dtype=float)
    s = sigmoid(beta * z)
    return s + beta * z * s * (1.0 - s)


def gelu(z, approximate: bool = True):
    """Gaussian Error Linear Unit.

    `approximate=True` uses the tanh approximation used in BERT/GPT.
    """
    z = np.asarray(z, dtype=float)
    if approximate:
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))
    from math import erf
    return 0.5 * z * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def gelu_prime(z, approximate: bool = True):
    """Derivative of (approximate) GELU."""
    z = np.asarray(z, dtype=float)
    if approximate:
        c = np.sqrt(2.0 / np.pi)
        inner = c * (z + 0.044715 * z ** 3)
        t = np.tanh(inner)
        dinner = c * (1.0 + 3 * 0.044715 * z ** 2)
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * dinner
    from math import erf, pi, exp
    phi = np.vectorize(lambda v: exp(-0.5 * v * v) / np.sqrt(2 * pi))(z)
    cdf = np.vectorize(lambda v: 0.5 * (1 + erf(v / np.sqrt(2))))(z)
    return cdf + z * phi


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ACTIVATIONS = {
    "linear": (linear, linear_prime),
    "identity": (linear, linear_prime),
    "relu": (relu, relu_prime),
    "leaky_relu": (leaky_relu, leaky_relu_prime),
    "elu": (elu, elu_prime),
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
    "softplus": (softplus, softplus_prime),
    "swish": (swish, swish_prime),
    "silu": (swish, swish_prime),
    "gelu": (gelu, gelu_prime),
}


def get_activation(name: str):
    """Return the activation function `f` registered under `name`."""
    if name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}'. "
                       f"Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name][0]


def get_activation_grad(name: str):
    """Return the derivative `f'` registered under `name`."""
    if name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}'. "
                       f"Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name][1]


# For activations whose derivative can be expressed from the OUTPUT, map them so
# layers can save compute. Others fall back to the (input-based) prime function.
_OUTPUT_BASED = {
    "sigmoid": lambda a: a * (1.0 - a),
    "tanh": lambda a: 1.0 - a * a,
    "linear": lambda a: np.ones_like(a),
    "identity": lambda a: np.ones_like(a),
}


def get_activation_from_output(name: str):
    """Return `g(a) = f'(z)` computed from the activation output, or None."""
    return _OUTPUT_BASED.get(name)
