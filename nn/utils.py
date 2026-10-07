"""
nn.utils -- shared helpers: initialization, batching, metrics, one-hot, seeding.

Weight initialization is critical (docs/02_algebra.md, section 5):

* `xavier_uniform` / `xavier_normal` keep variance constant for tanh/sigmoid.
* `he_normal` / `he_uniform` keep variance constant through ReLU.
* variance-scaling init uses `fan_in` from the shape.
"""

from __future__ import annotations

import numpy as np

from .linalg import one_hot, EPS

__all__ = [
    "set_seed",
    "one_hot",
    "xavier_uniform",
    "xavier_normal",
    "he_normal",
    "he_uniform",
    "orthogonal",
    "zeros",
    "minibatches",
    "shuffle",
    "train_test_split",
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "mse_score",
    "mae_score",
    "r2_score",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> np.random.Generator:
    """Seed the global NumPy RNG and return a fresh `Generator`."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def shuffle(x: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None):
    """Return a jointly shuffled copy of `(x, y)` along axis 0."""
    rng = rng or np.random.default_rng()
    idx = rng.permutation(x.shape[0])
    return x[idx], y[idx]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def xavier_uniform(shape, gain: float = 1.0, rng: np.random.Generator | None = None):
    """Glorot uniform: `U(-a, a)`, `a = gain*sqrt(6/(fan_in+fan_out))`."""
    rng = rng or np.random.default_rng()
    fan_in, fan_out = shape[1], shape[0]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def xavier_normal(shape, gain: float = 1.0, rng: np.random.Generator | None = None):
    """Glorot normal: `N(0, gain^2 * 2/(fan_in+fan_out))`."""
    rng = rng or np.random.default_rng()
    fan_in, fan_out = shape[1], shape[0]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=shape)


def he_normal(shape, gain: float = 1.0, rng: np.random.Generator | None = None):
    """He normal: `N(0, 2/fan_in)` (scaled by gain). Use with ReLU."""
    rng = rng or np.random.default_rng()
    fan_in = shape[1]
    std = gain * np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=shape)


def he_uniform(shape, gain: float = 1.0, rng: np.random.Generator | None = None):
    """He uniform: `U(-a, a)`, `a = gain*sqrt(6/fan_in)`."""
    rng = rng or np.random.default_rng()
    fan_in = shape[1]
    limit = gain * np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=shape)


def orthogonal(shape, gain: float = 1.0, rng: np.random.Generator | None = None):
    """Orthogonal init: singular values are 1 (preserves signal norm).

    Good for deep stacks / recurrent cores.
    """
    rng = rng or np.random.default_rng()
    rows, cols = shape
    a = rng.normal(0.0, 1.0, size=(rows, cols))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if rows <= cols else v
    q = q[:rows, :cols]
    return gain * q


def zeros(shape, **_) -> np.ndarray:
    """Zeros (standard for biases)."""
    return np.zeros(shape)


INITIALIZERS = {
    "xavier_uniform": xavier_uniform,
    "glorot_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "glorot_normal": xavier_normal,
    "he_normal": he_normal,
    "kaiming_normal": he_normal,
    "he_uniform": he_uniform,
    "kaiming_uniform": he_uniform,
    "orthogonal": orthogonal,
    "zeros": zeros,
}


def get_initializer(name):
    """Return an initializer callable by name (or pass a callable through)."""
    if callable(name):
        return name
    if name not in INITIALIZERS:
        raise KeyError(f"Unknown initializer '{name}'. "
                       f"Available: {sorted(INITIALIZERS)}")
    return INITIALIZERS[name]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def minibatches(x: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle_data: bool = True, rng: np.random.Generator | None = None):
    """Yield `(x_batch, y_batch)` tuples, optionally shuffled each epoch."""
    m = x.shape[0]
    rng = rng or np.random.default_rng()
    idx = rng.permutation(m) if shuffle_data else np.arange(m)
    for start in range(0, m, batch_size):
        sel = idx[start:start + batch_size]
        yield x[sel], y[sel]


def train_test_split(x, y, test_size: float = 0.2, shuffle_data: bool = True,
                     rng: np.random.Generator | None = None):
    """Split arrays into `(x_train, x_test, y_train, y_test)`."""
    m = x.shape[0]
    rng = rng or np.random.default_rng()
    idx = rng.permutation(m) if shuffle_data else np.arange(m)
    cut = int(round(m * (1.0 - test_size)))
    tr, te = idx[:cut], idx[cut:]
    return x[tr], x[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct predictions.

    Accepts either probability/logit matrices (argmax is taken) or labels.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true).reshape(-1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        preds = np.argmax(y_pred, axis=1)
    else:
        preds = (y_pred.reshape(-1) >= 0.5).astype(int) if y_pred.dtype.kind == "f" \
            else y_pred.reshape(-1).astype(int)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    return float(np.mean(preds == y_true))


def confusion_matrix(y_pred, y_true, num_classes: int | None = None) -> np.ndarray:
    """Confusion matrix `C[i, j]` = #(true = i, predicted = j)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = y_pred.reshape(-1)
    k = num_classes or (int(max(y_true.max(), y_pred.max())) + 1)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_pred, y_true, positive: int = 1) -> dict:
    """Precision, recall and F1 for the given positive class."""
    cm = confusion_matrix(y_pred, y_true)
    tp = cm[positive, positive]
    fp = cm[:, positive].sum() - tp
    fn = cm[positive, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def mse_score(y_pred, y_true) -> float:
    """Mean squared error."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(d * d))


def mae_score(y_pred, y_true) -> float:
    """Mean absolute error."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(d)))


def r2_score(y_pred, y_true) -> float:
    """Coefficient of determination R^2."""
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + EPS
    return float(1.0 - ss_res / ss_tot)
