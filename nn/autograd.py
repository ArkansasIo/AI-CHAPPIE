"""
nn.autograd -- a small reverse-mode automatic differentiation engine.

This is the "engine behind deep learning" described in docs/05_autodiff.md.
It builds a dynamic computational graph of `Tensor` nodes and computes gradients
with a single reverse topological sweep.

Example
-------
>>> from nn.autograd import Tensor
>>> x = Tensor([[0.0, 1.0]])
>>> W = Tensor(np.random.randn(1, 2), requires_grad=True)
>>> b = Tensor([0.5], requires_grad=True)
>>> y = ((x @ W.T) + b)
>>> loss = (y * y).sum()        # square
>>> loss.backward()
>>> W.grad                       # dloss/dW
array([[0. , ...]])

Supported ops: +, -, *, /, @ (matmul), **, exp, log, tanh, relu, sigmoid,
softmax, sum, mean, max, reshape, transpose, concatenate, and a stable
cross_entropy / log_softmax pair.

Every op defines its local derivative (the chain rule factor); the engine
multiplies it by the upstream gradient and accumulates with `+=` (multivariate
chain rule / shared subexpressions).
"""

from __future__ import annotations

import numpy as np

__all__ = ["Tensor", "no_grad"]

_EPS = 1e-12


def _ensure_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad: np.ndarray, shape) -> np.ndarray:
    """Sum `grad` so it matches `shape` (inverse of broadcasting)."""
    grad = np.asarray(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An ndarray plus a gradient slot and a backward closure."""

    __array_priority__ = 1000  # so numpy defers to our operators

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- housekeeping ------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, g: np.ndarray) -> None:
        """Accumulate a gradient (sums when a node is used multiple times)."""
        g = _unbroadcast(np.asarray(g, dtype=float), self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    # -- core ops ----------------------------------------------------------
    def __add__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "+")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                other._accumulate(out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other):
        return _ensure_tensor(other) + (-self)

    def __mul__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "*")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * other.data)
            if other.requires_grad:
                other._accumulate(out.grad * self.data)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data / other.data,
                     self.requires_grad or other.requires_grad, (self, other), "/")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / other.data)
            if other.requires_grad:
                other._accumulate(-out.grad * self.data / (other.data ** 2))
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, self.requires_grad, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * power * self.data ** (power - 1))
        out._backward = _backward
        return out

    def __matmul__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate(self.data.T @ out.grad)
        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return _ensure_tensor(other) @ self

    # -- elementwise nonlinearities ---------------------------------------
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * out.data)
        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        safe = np.clip(self.data, _EPS, None)
        out = Tensor(np.log(safe), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / safe)
        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (1.0 - t * t))
        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * s * (1.0 - s))
        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        mask = (self.data > 0).astype(float)
        out = Tensor(self.data * mask, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * mask)
        out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> "Tensor":
        mask = np.where(self.data > 0, 1.0, alpha)
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data),
                     self.requires_grad, (self,), "leaky_relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * mask)
        out._backward = _backward
        return out

    # -- reductions / shape ops -------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
            self._accumulate(np.broadcast_to(g, self.data.shape))
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "mean")

        def _backward():
            if not self.requires_grad:
                return
            n = self.data.size if axis is None else self.data.shape[
                axis if isinstance(axis, int) else axis[0]]
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
            self._accumulate(np.broadcast_to(g, self.data.shape) / n)
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, self.requires_grad, (self,), "max")

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
            mask = (self.data == np.broadcast_to(out_data, self.data.shape))
            self._accumulate(mask * np.broadcast_to(g, self.data.shape))
        out._backward = _backward
        return out

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.data.shape))
        out._backward = _backward
        return out

    def transpose(self, *axes) -> "Tensor":
        axes = axes[0] if len(axes) == 1 and axes[0] is not None else (axes or None)
        out = Tensor(self.data.transpose(axes) if axes else self.data.T,
                     self.requires_grad, (self,), "transpose")

        def _backward():
            if not self.requires_grad:
                return
            if axes:
                inverse = np.argsort(axes)
                self._accumulate(out.grad.transpose(inverse))
            else:
                self._accumulate(out.grad.T)
        out._backward = _backward
        return out

    def __getitem__(self, idx) -> "Tensor":
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward():
            if not self.requires_grad:
                return
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self._accumulate(g)
        out._backward = _backward
        return out

    # -- losses ------------------------------------------------------------
    def softmax(self, axis: int = -1) -> "Tensor":
        z = self.data
        z_shift = z - z.max(axis=axis, keepdims=True)
        e = np.exp(z_shift)
        p = e / (e.sum(axis=axis, keepdims=True) + _EPS)
        out = Tensor(p, self.requires_grad, (self,), "softmax")

        def _backward():
            if not self.requires_grad:
                return
            # dL/dz = p * (dL/dp - sum_j(dL/dp_j * p_j))
            g = out.grad
            dot = (g * p).sum(axis=axis, keepdims=True)
            self._accumulate(p * (g - dot))
        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        z = self.data
        zmax = z.max(axis=axis, keepdims=True)
        shifted = z - zmax
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True) + _EPS)
        out_data = shifted - lse
        out = Tensor(out_data, self.requires_grad, (self,), "log_softmax")

        def _backward():
            if not self.requires_grad:
                return
            p = np.exp(out_data)
            g = out.grad
            self._accumulate(g - p * g.sum(axis=axis, keepdims=True))
        out._backward = _backward
        return out

    def cross_entropy(self, targets) -> "Tensor":
        """Cross-entropy from logits with integer or one-hot targets."""
        y = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
        z = self.data
        if y.ndim == 1:
            y = np.eye(z.shape[-1])[y.astype(int)]
        m = z.shape[0]
        logp = self.log_softmax(axis=-1)
        nll = (-(logp * Tensor(y))).sum(axis=-1).mean()
        return nll

    # -- backward ----------------------------------------------------------
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Reverse-mode pass: compute d(root)/d(every reachable node)."""
        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._prev:
                    build(parent)
                topo.append(node)
        build(self)

        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad, dtype=float)

        for node in reversed(topo):
            node._backward()

    # -- misc --------------------------------------------------------------
    def item(self) -> float:
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"


# ---------------------------------------------------------------------------
# Convenience: a context manager placeholder (grad tracking is per-Tensor here)
# ---------------------------------------------------------------------------
class no_grad:
    """No-op context manager kept for API symmetry / future use."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
