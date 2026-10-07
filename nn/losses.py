"""
nn.losses -- objective functions with analytic gradients.

A loss object exposes:

    forward(y_pred, y_true) -> float          (mean loss over the batch)
    backward()              -> ndarray        (dJ/dy_pred, same shape as y_pred)

We use the *mean* over the batch (dividing by m), so gradients are batch-size
independent. Derivations are in docs/03_calculus.md (sections 6-8).

Fused losses (`SoftmaxCrossEntropy`, `SigmoidBinaryCrossEntropy`) accept *raw
logits* and return the analytic gradient `(p - y)` directly -- this is both
faster and numerically safer than chaining a softmax and a log through
automatic differentiation.
"""

from __future__ import annotations

import numpy as np

from .linalg import EPS, log_softmax, softmax

__all__ = [
    "Loss",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "SoftmaxCrossEntropy",
    "SigmoidBinaryCrossEntropy",
    "LOSSES",
    "get_loss",
]


class Loss:
    """Base class for all losses."""

    name = "loss"

    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self._grad = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        if self._grad is None:
            raise RuntimeError("backward() called before forward().")
        return self._grad

    def __call__(self, y_pred, y_true) -> float:
        return self.forward(y_pred, y_true)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Regression losses
# ---------------------------------------------------------------------------
class MeanSquaredError(Loss):
    """`J = (1/(2m)) sum ||yhat - y||^2`; gradient `(yhat - y)/m`."""

    name = "mse"

    def forward(self, y_pred, y_true) -> float:
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        self.y_pred, self.y_true = y_pred, y_true
        m = y_pred.shape[0]
        diff = y_pred - y_true
        self._grad = diff / m
        return float(np.sum(diff * diff) / (2 * m))


class MeanAbsoluteError(Loss):
    """`J = (1/m) sum |yhat - y|`; subgradient `sign(yhat - y)/m`."""

    name = "mae"

    def forward(self, y_pred, y_true) -> float:
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        self.y_pred, self.y_true = y_pred, y_true
        m = y_pred.shape[0]
        diff = y_pred - y_true
        self._grad = np.sign(diff) / m
        return float(np.sum(np.abs(diff)) / m)


# ---------------------------------------------------------------------------
# Probability losses
# ---------------------------------------------------------------------------
class BinaryCrossEntropy(Loss):
    """BCE on already-sigmoided probabilities.

    `J = -(1/m) sum [y ln p + (1-y) ln(1-p)]`, gradient `(p - y)/(m p (1-p))`.
    Prefer `SigmoidBinaryCrossEntropy` for numerical stability.
    """

    name = "bce"

    def forward(self, y_pred, y_true) -> float:
        p = np.clip(np.asarray(y_pred, dtype=float), EPS, 1.0 - EPS)
        y = np.asarray(y_true, dtype=float)
        self.y_pred, self.y_true = p, y
        m = p.shape[0]
        self._grad = (p - y) / (m * p * (1.0 - p))
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / m)


class CategoricalCrossEntropy(Loss):
    """Cross-entropy on already-softmaxed probabilities.

    `J = -(1/m) sum_i y_i ln p_i`,  gradient `-y/(m p)`.
    Prefer `SoftmaxCrossEntropy` when you have raw logits.
    """

    name = "ce"

    def forward(self, y_pred, y_true) -> float:
        p = np.clip(np.asarray(y_pred, dtype=float), EPS, 1.0)
        y = np.asarray(y_true, dtype=float)
        if y.ndim == 1:                      # accept integer labels too
            y = np.eye(p.shape[-1])[y.astype(int)]
        self.y_pred, self.y_true = p, y
        m = p.shape[0]
        self._grad = -y / (m * p)
        return float(-np.sum(y * np.log(p)) / m)


# ---------------------------------------------------------------------------
# Fused / numerically-stable losses (operate on LOGITS)
# ---------------------------------------------------------------------------
class SoftmaxCrossEntropy(Loss):
    """Softmax + cross-entropy fused; gradient is the famous `(p - y)/m`.

    Accepts raw logits `z` of shape `(m, K)` and either one-hot targets of shape
    `(m, K)` or integer labels of shape `(m,)`.
    """

    name = "softmax_ce"

    def forward(self, logits, y_true) -> float:
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y_true)
        if y.ndim == 1:
            y = np.eye(z.shape[-1])[y.astype(int)]
        else:
            y = y.astype(float)
        self.y_true = y
        self.probs = softmax(z, axis=-1)
        m = z.shape[0]
        # Stable NLL using log_softmax.
        logp = log_softmax(z, axis=-1)
        loss = -np.sum(y * logp) / m
        self._grad = (self.probs - y) / m
        return float(loss)


class SigmoidBinaryCrossEntropy(Loss):
    """Sigmoid + BCE fused; gradient `(p - y)/m` computed from logits stably."""

    name = "sigmoid_bce"

    def forward(self, logits, y_true) -> float:
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y_true, dtype=float)
        self.y_true = y
        # softplus(z) - y z is the stable form of the BCE loss.
        loss = np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))
        m = z.shape[0]
        p = 1.0 / (1.0 + np.exp(-z))
        self.probs = p
        self._grad = (p - y) / m
        return float(np.sum(loss) / m)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
LOSSES = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "bce": BinaryCrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
    "ce": CategoricalCrossEntropy,
    "categorical_cross_entropy": CategoricalCrossEntropy,
    "softmax_ce": SoftmaxCrossEntropy,
    "softmax_cross_entropy": SoftmaxCrossEntropy,
    "sigmoid_bce": SigmoidBinaryCrossEntropy,
    "sigmoid_cross_entropy": SigmoidBinaryCrossEntropy,
}


def get_loss(name: str) -> Loss:
    """Instantiate a loss by name."""
    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(LOSSES)}")
    return LOSSES[name]()
