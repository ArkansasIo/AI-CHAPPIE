"""
nn.layers -- parameterized layers with explicit forward/backward.

Every layer implements:

    forward(x)      -> y            (caches whatever backward needs)
    backward(dout)  -> dx           (computes dW/db into self.grads)
    parameters()    -> dict[str, ndarray]
    grads()         -> dict[str, ndarray]      (same keys as parameters)
    zero_grad()     -> None                    (clears grads)
    train()/eval()  -> None                    (mode switch for Dropout/BN)

The math for the Dense layer is in docs/03_calculus.md section 4:

    Z = X W^T + b
    dW = dZ^T X          dX = dZ W          db = sum_i dZ[i]

Layers
------
Dense        : affine transform (fully connected)
Activation   : elementwise nonlinearity
Dropout      : inverted dropout regularization
BatchNorm    : normalize pre/post activations with running statistics
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation, get_activation_grad, get_activation_from_output
from .utils import get_initializer

__all__ = ["Layer", "Dense", "Activation", "Dropout", "BatchNorm"]


class Layer:
    """Abstract base class for all layers."""

    name = "layer"

    def __init__(self):
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}

    def grads(self) -> dict:
        return {}

    def zero_grad(self) -> None:
        for g in self.grads().values():
            g[...] = 0.0

    def train(self, mode: bool = True) -> "Layer":
        self.training = mode
        return self

    def eval(self) -> "Layer":
        return self.train(False)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Dense / fully-connected
# ---------------------------------------------------------------------------
class Dense(Layer):
    """Affine layer: `y = x W^T + b`.

    Parameters
    ----------
    in_features, out_features : int
    use_bias   : bool, default True
    init       : str or callable, default 'he_normal'
        one of utils.INITIALIZERS
    bias_init  : str, default 'zeros'
    name       : optional label
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True,
                 init: str = "he_normal", bias_init: str = "zeros",
                 rng: np.random.Generator | None = None, name: str | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.name = name or "dense"

        init_fn = get_initializer(init)
        self.W = init_fn((out_features, in_features), rng=rng)
        self.b = (get_initializer(bias_init)((out_features,))
                  if use_bias else None)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if use_bias else None

        # caches
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._x = x
        z = x @ self.W.T
        if self.use_bias:
            z = z + self.b
        return z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._x
        dout = np.asarray(dout, dtype=float)
        # dW = dZ^T X   (shape: (out, in))
        self.dW[...] = dout.T @ x
        if self.use_bias:
            # db = sum over batch of dZ
            self.db[...] = dout.sum(axis=0)
        # dX = dZ W      (shape: (batch, in))
        return dout @ self.W

    def parameters(self) -> dict:
        p = {"W": self.W}
        if self.use_bias:
            p["b"] = self.b
        return p

    def grads(self) -> dict:
        g = {"W": self.dW}
        if self.use_bias:
            g["b"] = self.db
        return g

    def __repr__(self) -> str:
        return (f"Dense(in={self.in_features}, out={self.out_features}, "
                f"bias={self.use_bias}, init='{self.name}')")


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
class Activation(Layer):
    """Elementwise activation wrapper around functions in `activations.py`."""

    def __init__(self, kind: str = "relu"):
        super().__init__()
        self.kind = kind
        self.name = kind
        self._f = get_activation(kind)
        self._fprime = get_activation_grad(kind)
        self._from_output = get_activation_from_output(kind)
        self._out = None
        self._z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=float)
        self._z = z
        self._out = self._f(z)
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._from_output is not None:
            return np.asarray(dout, dtype=float) * self._from_output(self._out)
        return np.asarray(dout, dtype=float) * self._fprime(self._z)

    def __repr__(self) -> str:
        return f"Activation('{self.kind}')"


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------
class Dropout(Layer):
    """Inverted dropout.

    During training each unit is kept with probability `1 - p` and scaled by
    `1/(1 - p)` so the expected activation is unchanged. At eval time the layer
    is the identity. Backward multiplies the gradient by the same mask.
    """

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.name = "dropout"
        self.rng = rng or np.random.default_rng()
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (self.rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return np.asarray(dout, dtype=float)
        return np.asarray(dout, dtype=float) * self._mask

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------
class BatchNorm(Layer):
    """Batch normalization over the feature axis.

    Forward (training):
        mu  = mean(X, axis=0)
        var = var(X, axis=0)
        xhat = (X - mu) / sqrt(var + eps)
        Y = gamma * xhat + beta
    Running averages of `mu`, `var` are kept for inference.

    Backward uses the standard three-term formula:
        dX = (1/(m*sigma)) * ( m*dYh - sum(dYh) - xhat * sum(dYh * xhat) )
    where dYh = dY * gamma.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.name = "batchnorm"

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self._xhat = None
        self._std = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self.training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self._std = np.sqrt(var + self.eps)
            self._xhat = (x - mu) / self._std
            # Update running statistics with exponential moving average.
            self.running_mean = (self.momentum * self.running_mean
                                 + (1 - self.momentum) * mu)
            self.running_var = (self.momentum * self.running_var
                                + (1 - self.momentum) * var)
        else:
            self._std = np.sqrt(self.running_var + self.eps)
            self._xhat = (x - self.running_mean) / self._std

        return self.gamma * self._xhat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=float)
        m = dout.shape[0]
        # Gradients for the scale/shift parameters.
        self.dgamma[...] = np.sum(dout * self._xhat, axis=0)
        self.dbeta[...] = np.sum(dout, axis=0)

        dxhat = dout * self.gamma
        # Standard batchnorm backward.
        dx = (1.0 / m) / self._std * (
            m * dxhat
            - dxhat.sum(axis=0)
            - self._xhat * (dxhat * self._xhat).sum(axis=0)
        )
        return dx

    def parameters(self) -> dict:
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self) -> dict:
        return {"dgamma": self.dgamma, "dbeta": self.dbeta}

    def __repr__(self) -> str:
        return f"BatchNorm(num_features={self.num_features})"
