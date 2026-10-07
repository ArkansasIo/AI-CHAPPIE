"""
nn.model -- the Sequential container: wiring layers into a trainable network.

`Sequential` owns an ordered list of layers, plus (optionally) a loss. It exposes
the full training contract seen in docs/01_theory.md:

    forward(X)             -> Yhat          (runs every layer in order)
    backward(dout)         -> dX            (runs every layer in reverse)
    parameters()/grads()   -> {name: array} (flattened over layers)
    zero_grad()
    train()/eval()
    fit(X, Y, ...)         -> history       (the training loop)
    predict(X) / evaluate(X, Y)
    save(path) / load(path)

Parameter keys are namespaced by layer index to keep the optimizer's name<->grad
matching unambiguous: e.g. `0.W`, `0.b`, `2.gamma`.
"""

from __future__ import annotations

import pickle

import numpy as np

from .layers import Layer, Dense, Activation, Dropout, BatchNorm
from .losses import Loss, SoftmaxCrossEntropy, get_loss
from .optimizers import get_optimizer
from .utils import minibatches

__all__ = ["Sequential", "MLP"]


class Sequential:
    """An ordered stack of layers trained end-to-end."""

    def __init__(self, layers=None, loss: Loss | str | None = None,
                 name: str = "Sequential"):
        self.layers: list[Layer] = list(layers) if layers else []
        self.name = name
        if isinstance(loss, str):
            loss = get_loss(loss)
        self.loss: Loss | None = loss
        self.history: dict[str, list] = {"loss": [], "val_loss": [],
                                         "accuracy": [], "val_accuracy": []}

    # -- construction ------------------------------------------------------
    def add(self, layer: Layer) -> "Sequential":
        """Append a layer and return self (for chaining)."""
        self.layers.append(layer)
        return self

    # -- mode --------------------------------------------------------------
    def train(self, mode: bool = True) -> "Sequential":
        for layer in self.layers:
            layer.train(mode)
        return self

    def eval(self) -> "Sequential":
        return self.train(False)

    # -- forward / backward ------------------------------------------------
    def forward(self, x: np.ndarray, return_cache: bool = False):
        """Push `x` through every layer. Optionally return per-layer outputs."""
        activations = [x]
        out = x
        for layer in self.layers:
            out = layer.forward(out)
            activations.append(out)
        if return_cache:
            return out, activations
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagate `dout` through the layers in reverse."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def __call__(self, x):
        return self.forward(x)

    # -- parameters --------------------------------------------------------
    def parameters(self) -> dict:
        params = {}
        for i, layer in enumerate(self.layers):
            for key, value in layer.parameters().items():
                params[f"{i}.{key}"] = value
        return params

    def grads(self) -> dict:
        grads = {}
        for i, layer in enumerate(self.layers):
            for key, value in layer.grads().items():
                grads[f"{i}.{key}"] = value
        return grads

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters().values()))

    # -- one training step -------------------------------------------------
    def train_step(self, x: np.ndarray, y: np.ndarray, optimizer, loss=None):
        """Forward, loss, backward, update. Returns the scalar batch loss."""
        if loss is None:
            loss = self.loss
        if loss is None:
            raise ValueError("No loss provided (pass one or set model.loss).")

        y_pred = self.forward(x)          # forward pass
        loss_value = loss.forward(y_pred, y)   # scalar objective
        dout = loss.backward()            # dJ/dy_pred
        self.backward(dout)               # backprop
        optimizer.step(self.parameters(), self.grads())
        self.zero_grad()
        return float(loss_value)

    # -- full training loop ------------------------------------------------
    def fit(self, x, y, epochs: int = 100, batch_size: int = 32,
            optimizer: str | object = "adam", lr: float = 1e-3,
            loss: Loss | str | None = None, validation_data=None,
            verbose: bool = True, log_every: int = 10,
            shuffle_data: bool = True, seed: int | None = 42,
            **optimizer_kwargs) -> dict:
        """Train the network.

        Parameters
        ----------
        x, y        : training arrays (rows = examples)
        epochs      : passes over the data
        batch_size  : mini-batch size
        optimizer   : name (str) or an instantiated Optimizer
        lr          : learning rate
        loss        : override the model loss
        validation_data : optional `(x_val, y_val)`
        verbose     : print progress

        Returns
        -------
        history dict with `loss`, `val_loss`, `accuracy`, `val_accuracy`.
        """
        if loss is not None:
            self.loss = get_loss(loss) if isinstance(loss, str) else loss
        if self.loss is None:
            raise ValueError("No loss set; pass loss= to fit().")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if isinstance(optimizer, str):
            optimizer = get_optimizer(optimizer, lr=lr, **optimizer_kwargs)
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        self.history = {"loss": [], "val_loss": [], "accuracy": [],
                        "val_accuracy": []}

        for epoch in range(1, epochs + 1):
            self.train(True)
            epoch_losses = []
            for xb, yb in minibatches(x, y, batch_size,
                                      shuffle_data=shuffle_data, rng=rng):
                epoch_losses.append(self.train_step(xb, yb, optimizer))
            train_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(train_loss)

            train_acc = self._accuracy_on(x, y)
            self.history["accuracy"].append(train_acc)

            val_line = ""
            if validation_data is not None:
                xv, yv = validation_data
                val_loss = self._loss_on(xv, yv)
                val_acc = self._accuracy_on(xv, yv)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                val_line = f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"

            if verbose and (epoch % log_every == 0 or epoch == 1
                            or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | loss {train_loss:.4f} "
                      f"| acc {train_acc:.4f}{val_line}")

        self.eval()
        return self.history

    # -- inference ---------------------------------------------------------
    def predict(self, x):
        """Forward pass in eval mode (no dropout / uses running BN stats)."""
        was_training = any(l.training for l in self.layers)
        self.eval()
        out = self.forward(np.asarray(x, dtype=float))
        if was_training:
            self.train(True)
        return out

    def predict_classes(self, x):
        """Argmax over the last axis (classification labels)."""
        scores = self.predict(x)
        if scores.ndim > 1 and scores.shape[1] > 1:
            return np.argmax(scores, axis=1)
        return (scores.reshape(-1) >= 0.5).astype(int)

    def evaluate(self, x, y) -> dict:
        """Return a metrics dict for the given data."""
        from .utils import accuracy as acc_fn, mse_score
        y = np.asarray(y)
        scores = self.predict(x)
        out = {"loss": self._loss_on(x, y)}
        if self.loss is not None and self.loss.name in (
                "ce", "categorical_cross_entropy", "softmax_ce",
                "softmax_cross_entropy", "bce", "binary_cross_entropy",
                "sigmoid_bce", "sigmoid_cross_entropy"):
            out["accuracy"] = acc_fn(scores, y)
        else:
            out["mse"] = mse_score(scores, y)
        return out

    # -- private helpers ---------------------------------------------------
    def _loss_on(self, x, y) -> float:
        if self.loss is None:
            return float("nan")
        self.eval()
        return float(self.loss.forward(self.forward(np.asarray(x, dtype=float)), y))

    def _accuracy_on(self, x, y) -> float:
        from .utils import accuracy as acc_fn
        if self.loss is None or self.loss.name in ("mse", "mae",
                                                   "mean_squared_error",
                                                   "mean_absolute_error"):
            # For regression report a simple -MSE proxy? Better: return R^2-ish.
            from .utils import r2_score
            return r2_score(self.predict(x), y)
        return acc_fn(self.predict(x), y)

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        """Pickle the model (architecture + weights)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Sequential":
        """Load a model saved with `save`."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    def summary(self) -> str:
        """Human-readable architecture + parameter count."""
        lines = [f"{self.name} ({self.num_parameters()} parameters)"]
        lines.append("-" * 52)
        for i, layer in enumerate(self.layers):
            n = sum(p.size for p in layer.parameters().values())
            lines.append(f"  [{i}] {layer!r:<40} {n:>7}")
        lines.append("-" * 52)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Sequential({len(self.layers)} layers, loss={self.loss})"


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def MLP(sizes, activations="relu", out_activation="linear",
        init="he_normal", dropout: float = 0.0, batchnorm: bool = False,
        loss=None, seed: int = 42) -> Sequential:
    """Build a multi-layer perceptron from a list of layer widths.

    Example
    -------
    >>> net = MLP([2, 16, 16, 1], activations='tanh', out_activation='sigmoid',
    ...           loss='sigmoid_bce')
    """
    rng = np.random.default_rng(seed)
    net = Sequential(loss=loss)
    n_pairs = len(sizes) - 1
    if isinstance(activations, str):
        activations = [activations] * n_pairs
    for i in range(n_pairs):
        act = out_activation if i == n_pairs - 1 else activations[i]
        net.add(Dense(sizes[i], sizes[i + 1], init=init, rng=rng))
        if batchnorm and i < n_pairs - 1:
            net.add(BatchNorm(sizes[i + 1]))
        if act and act != "linear":
            net.add(Activation(act))
        elif i < n_pairs - 1:
            pass  # keep linear hidden if explicitly requested
        if dropout > 0 and i < n_pairs - 1:
            net.add(Dropout(dropout, rng=rng))
    return net
