"""
nn.optimizers -- gradient-based update rules.

All optimizers share the interface:

    opt = Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    ...
    opt.step(model.parameters(), model.grads())

`parameters()` returns `{name: ndarray}`; `grads()` returns the same keys with
`d`-prefixed names (e.g. `dW`, `db`, `dgamma`). Optimizers match `p` with `dp`
by stripping the leading `d`.

Implemented: SGD, Momentum, Nesterov, RMSProp, Adam, AdamW, AdaGrad.
See docs/04_optimization.md for the derivations and intuition.
"""

from __future__ import annotations

import numpy as np

from .linalg import EPS

__all__ = [
    "Optimizer",
    "SGD",
    "Momentum",
    "Nesterov",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "AdamW",
    "OPTIMIZERS",
    "get_optimizer",
]


def _match_grad(param_name: str, grads: dict):
    """Find the gradient key that corresponds to a parameter name."""
    for cand in (param_name, "d" + param_name, "d" + param_name.lstrip("_")):
        if cand in grads:
            return grads[cand]
    raise KeyError(f"No gradient found for parameter '{param_name}' in {list(grads)}")


class Optimizer:
    """Base optimizer: holds hyperparameters and per-parameter state."""

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0,
                 clip_norm: float | None = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.t = 0
        self.state = {}

    # -- hooks -------------------------------------------------------------
    def _decay(self, p, name):
        """L2 weight decay: add lambda * p to the gradient (not for biases)."""
        if self.weight_decay and name in ("W", "gamma"):
            return self.weight_decay * p
        return 0.0

    def _clip(self, grads_list):
        if self.clip_norm is None:
            return grads_list
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads_list))
        if total > self.clip_norm and total > 0:
            scale = self.clip_norm / total
            return [g * scale for g in grads_list]
        return grads_list

    # -- interface ---------------------------------------------------------
    def step(self, params: dict, grads: dict) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"


# ---------------------------------------------------------------------------
# Vanilla SGD
# ---------------------------------------------------------------------------
class SGD(Optimizer):
    """Stochastic gradient descent: `theta <- theta - lr * g`."""

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        updated = []
        keys = []
        for name, p in params.items():
            g = _match_grad(name, grads)
            g = g + self._decay(p, name)
            updated.append(g)
            keys.append((name, p))
        updated = self._clip(updated)
        for (name, p), g in zip(keys, updated):
            p -= self.lr * g


# ---------------------------------------------------------------------------
# Momentum family
# ---------------------------------------------------------------------------
class Momentum(Optimizer):
    """Heavy-ball momentum: `v <- beta v + g; theta <- theta - lr v`."""

    def __init__(self, lr=1e-3, momentum: float = 0.9, **kw):
        super().__init__(lr, **kw)
        self.momentum = momentum

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for name, p in params.items():
            g = _match_grad(name, grads) + self._decay(p, name)
            v = self.state.get(name)
            if v is None:
                v = np.zeros_like(p)
            v = self.momentum * v + g
            self.state[name] = v
            p -= self.lr * v


class Nesterov(Momentum):
    """Nesterov accelerated gradient (look-ahead correction)."""

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for name, p in params.items():
            g = _match_grad(name, grads) + self._decay(p, name)
            v = self.state.get(name)
            if v is None:
                v = np.zeros_like(p)
            v_prev = v
            v = self.momentum * v - self.lr * g
            self.state[name] = v
            # Nesterov update: step using the corrected velocity.
            p += -self.momentum * v_prev + (1 + self.momentum) * v


# ---------------------------------------------------------------------------
# AdaGrad / RMSProp
# ---------------------------------------------------------------------------
class AdaGrad(Optimizer):
    """Accumulate squared gradients; `theta -= lr * g / (sqrt(G) + eps)`."""

    def __init__(self, lr=1e-2, eps: float = 1e-8, **kw):
        super().__init__(lr, **kw)
        self.eps = eps

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for name, p in params.items():
            g = _match_grad(name, grads) + self._decay(p, name)
            G = self.state.get(name)
            if G is None:
                G = np.zeros_like(p)
            G += g * g
            self.state[name] = G
            p -= self.lr * g / (np.sqrt(G) + self.eps)


class RMSProp(Optimizer):
    """Running RMS of gradients; `s <- rho s + (1-rho) g^2`."""

    def __init__(self, lr=1e-3, rho: float = 0.9, eps: float = 1e-8, **kw):
        super().__init__(lr, **kw)
        self.rho = rho
        self.eps = eps

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for name, p in params.items():
            g = _match_grad(name, grads) + self._decay(p, name)
            s = self.state.get(name)
            if s is None:
                s = np.zeros_like(p)
            s = self.rho * s + (1.0 - self.rho) * g * g
            self.state[name] = s
            p -= self.lr * g / (np.sqrt(s) + self.eps)


# ---------------------------------------------------------------------------
# Adam / AdamW
# ---------------------------------------------------------------------------
class Adam(Optimizer):
    """Adam: bias-corrected first and second moment estimates.

    m <- b1 m + (1-b1) g
    v <- b2 v + (1-b2) g^2
    m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, **kw):
        super().__init__(lr, **kw)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for name, p in params.items():
            g = _match_grad(name, grads) + self._decay(p, name)
            st = self.state.get(name)
            if st is None:
                st = {"m": np.zeros_like(p), "v": np.zeros_like(p)}
            m, v = st["m"], st["v"]
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * g * g
            st["m"], st["v"] = m, v
            self.state[name] = st
            m_hat = m / (1.0 - b1 ** self.t)
            v_hat = v / (1.0 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (AdamW).

    Instead of adding lambda*p to the gradient, the decay is applied directly
    to the parameter after the adaptive step.
    """

    def step(self, params: dict, grads: dict) -> None:
        wd = self.weight_decay
        # Temporarily disable the gradient-coupled decay; AdamW does it directly.
        self.weight_decay = 0.0
        try:
            super().step(params, grads)
        finally:
            self.weight_decay = wd
        if wd:
            for name, p in params.items():
                if name in ("W", "gamma"):
                    p -= self.lr * wd * p


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "adamw": AdamW,
}


def get_optimizer(name: str, **kwargs) -> Optimizer:
    """Instantiate an optimizer by name."""
    if isinstance(name, Optimizer):
        return name
    if name not in OPTIMIZERS:
        raise KeyError(f"Unknown optimizer '{name}'. "
                       f"Available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](**kwargs)
