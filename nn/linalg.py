"""
nn.linalg -- linear algebra and calculus primitives.

This module contains the *mathematical bedrock* of the library:

Linear algebra
--------------
* shape helpers, broadcasting untility
* `matmul`, `dot`, `outer`, `transpose`
* `norm_l1`, `norm_l2`, `frobenius`, `cosine_similarity`
* `svd_rank`, `condition_number`, `gram_matrix`, `pca` (principal components)
* `softmax`, `logsumexp`, `one_hot`, `standardize`

Calculus
--------
* `numerical_gradient`      -- central-difference derivative (1-D scalar f)
* `numerical_gradient_vec`  -- central-difference gradient of f: R^n -> R
* `numerical_jacobian`      -- full Jacobian of f: R^n -> R^m
* `grad_check`              -- compare analytic vs numeric gradient (relative err)

Everything is pure NumPy; see docs/02_algebra.md and docs/03_calculus.md for the
derivations that these functions implement.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EPS",
    "check_shapes",
    "concat",
    "softmax",
    "log_softmax",
    "logsumexp",
    "sigmoid_stable",
    "one_hot",
    "standardize",
    "matmul",
    "dot",
    "outer",
    "transpose",
    "norm_l1",
    "norm_l2",
    "frobenius",
    "cosine_similarity",
    "gram_matrix",
    "covariance",
    "pca",
    "svd_rank",
    "condition_number",
    "numerical_gradient",
    "numerical_gradient_vec",
    "numerical_jacobian",
    "grad_check",
    "clip_grad_norm",
]

# A tiny constant used to avoid division by zero / log(0).
EPS = 1e-12


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------
def check_shapes(*arrays: np.ndarray) -> bool:
    """Raise `ValueError` if any array has mismatched leading (batch) axes.

    All arrays must share the same first dimension.
    """
    shapes = [a.shape for a in arrays]
    if not shapes:
        return True
    first = shapes[0][0] if len(shapes[0]) > 0 else None
    for s in shapes[1:]:
        if len(s) == 0 or s[0] != first:
            raise ValueError(f"Incompatible batch dimensions: {shapes}")
    return True


def concat(arrays, axis: int = 0) -> np.ndarray:
    """Concatenate a list of arrays along `axis`."""
    return np.concatenate([np.atleast_1d(a) for a in arrays], axis=axis)


def _unbroadcast(grad: np.ndarray, target_shape) -> np.ndarray:
    """Sum `grad` back down to `target_shape` (inverse of NumPy broadcasting)."""
    grad = np.asarray(grad)
    target_shape = tuple(target_shape)
    # Sum leading axes that broadcasting inserted.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # Sum axes that were size 1 in the target but expanded.
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


# ---------------------------------------------------------------------------
# Basic linear-algebra operations
# ---------------------------------------------------------------------------
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product `a @ b` (thin wrapper for readability)."""
    return np.matmul(a, b)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector dot product `a . b`."""
    return np.dot(a, b)


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product `a b^T`."""
    return np.outer(a, b)


def transpose(a: np.ndarray) -> np.ndarray:
    """Matrix transpose."""
    return np.transpose(a)


# ---------------------------------------------------------------------------
# Norms & geometry
# ---------------------------------------------------------------------------
def norm_l1(a: np.ndarray) -> float:
    """L1 (Manhattan) norm: sum of absolute values."""
    return float(np.sum(np.abs(a)))


def norm_l2(a: np.ndarray) -> float:
    """L2 (Euclidean) norm: sqrt of sum of squares."""
    return float(np.sqrt(np.sum(a * a)))


def frobenius(a: np.ndarray) -> float:
    """Frobenius norm of a matrix (same formula as L2 for flattened arrays)."""
    return float(np.sqrt(np.sum(a * a)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors."""
    denom = (norm_l2(a) * norm_l2(b)) or EPS
    return float(np.dot(a, b) / denom)


def gram_matrix(x: np.ndarray) -> np.ndarray:
    """Gram matrix `X^T X` -- the (unnormalized) feature second-moment matrix."""
    return x.T @ x


def covariance(x: np.ndarray) -> np.ndarray:
    """Sample covariance matrix (columns are features, rows are examples)."""
    m = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / max(m - 1, 1)


# ---------------------------------------------------------------------------
# Spectral tools
# ---------------------------------------------------------------------------
def svd_rank(a: np.ndarray, tol: float = 1e-8) -> int:
    """Numerical rank via singular values."""
    s = np.linalg.svd(a, compute_uv=False)
    return int(np.sum(s > tol * (s[0] if s.size else 1.0)))


def condition_number(a: np.ndarray) -> float:
    """Ratio of largest to smallest singular value (inf if singular)."""
    s = np.linalg.svd(a, compute_uv=False)
    if s.size == 0 or s[-1] < EPS:
        return float("inf")
    return float(s[0] / s[-1])


def pca(x: np.ndarray, n_components: int | None = None):
    """Principal Component Analysis via SVD.

    Returns
    -------
    (components, explained_variance_ratio)
        `components` has shape `(n_components, n_features)`; rows are the
        principal directions (right singular vectors).
    """
    xc = x - x.mean(axis=0, keepdims=True)
    # Economy SVD: Xc = U S V^T ; principal directions are rows of V^T.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max(x.shape[0] - 1, 1)
    ratio = var / (var.sum() or EPS)
    k = n_components or vt.shape[0]
    return vt[:k], ratio[:k]


# ---------------------------------------------------------------------------
# Numerically-stable primitives
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtracts the max along `axis`).

    Maps real logits to the probability simplex.
    """
    z = np.asarray(z, dtype=float)
    z_shift = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z_shift)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Stable `log(sum(exp(z)))`, computed via the max-shift trick."""
    z = np.asarray(z, dtype=float)
    zmax = np.max(z, axis=axis, keepdims=True)
    shifted = z - zmax
    out = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + EPS) + zmax
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-softmax: `z - logsumexp(z)`."""
    z = np.asarray(z, dtype=float)
    return z - logsumexp(z, axis=axis, keepdims=True)


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Sigmoid that avoids overflow for large |z| (branch on sign)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def one_hot(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Convert integer labels to one-hot rows.

    `labels` -> shape `(m,)`; returns `(m, num_classes)`.
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    out = np.zeros((labels.size, num_classes), dtype=float)
    out[np.arange(labels.size), labels] = 1.0
    return out


def standardize(x: np.ndarray, eps: float = 1e-8):
    """Zero-mean, unit-variance per feature.

    Returns `(x_std, mean, std)` so the transform can be reused on test data.
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps), mean, std


# ---------------------------------------------------------------------------
# Calculus: numeric differentiation
# ---------------------------------------------------------------------------
def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function of a scalar/array.

    `f` may accept a scalar or an array element; we perturb element-wise.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def numerical_gradient_vec(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Alias of `numerical_gradient` for `f: R^n -> R`."""
    return numerical_gradient(f, x, eps=eps)


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Full Jacobian of `f: R^n -> R^m`, shape `(m, n)`.

    `f` must take a 1-D array of length `n` and return an array of length `m`.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.atleast_1d(f(x))
    m, n = f0.size, x.size
    jac = np.zeros((m, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        xm = x.copy()
        xm[j] -= eps
        jac[:, j] = (np.atleast_1d(f(xp)) - np.atleast_1d(f(xm))) / (2 * eps)
    return jac


def grad_check(f, grad_fn, param: np.ndarray, eps: float = 1e-6,
               tol: float = 1e-5) -> dict:
    """Compare an analytic gradient to a numerical one.

    Parameters
    ----------
    f        : callable(param) -> float (the scalar objective)
    grad_fn  : callable(param) -> ndarray (the analytic gradient)
    param    : current parameter value
    tol      : relative-error tolerance

    Returns
    -------
    dict with keys `analytic`, `numeric`, `relative_error`, `ok`.
    """
    param = np.asarray(param, dtype=float)
    analytic = np.asarray(grad_fn(param), dtype=float)
    numeric = numerical_gradient(f, param, eps=eps)
    denom = np.linalg.norm(analytic) + np.linalg.norm(numeric) + EPS
    rel = float(np.linalg.norm(analytic - numeric) / denom)
    return {
        "analytic": analytic,
        "numeric": numeric,
        "relative_error": rel,
        "ok": rel < tol,
    }


def clip_grad_norm(grads, max_norm: float = 5.0):
    """Global-norm gradient clipping (returns scaled copies of the grads)."""
    total = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
    if total <= max_norm or total == 0.0:
        return list(grads)
    scale = max_norm / (total + EPS)
    return [g * scale for g in grads]
