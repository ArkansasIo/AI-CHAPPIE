"""
Example 3 -- Train an MLP on MNIST (or a synthetic fallback).

If a real MNIST `.npz` file is available (path via CLI or the env var
`MNIST_NPZ`), it is used. Otherwise a small synthetic image dataset is generated
so the script always runs end-to-end.

Expected NPZ keys: `x_train, y_train, x_test, y_test`
    x_*: uint8 arrays (N, 28, 28) or (N, 784)
    y_*: int labels in [0, 9]

Usage:
    python examples/03_mnist_mlp.py [path/to/mnist.npz]
"""

import os
import sys

import numpy as np

from nn.model import Sequential
from nn.layers import Dense, Activation
from nn.losses import SoftmaxCrossEntropy
from nn.optimizers import Adam
from nn.utils import accuracy, confusion_matrix, set_seed, train_test_split


def load_mnist(path: str | None):
    """Return (x_train, y_train, x_test, y_test) or None if unavailable."""
    if path is None:
        path = os.environ.get("MNIST_NPZ")
    if not path or not os.path.exists(path):
        return None
    data = np.load(path)
    x_tr = data["x_train"].reshape(len(data["x_train"]), -1).astype(float) / 255.0
    x_te = data["x_test"].reshape(len(data["x_test"]), -1).astype(float) / 255.0
    return x_tr, data["y_train"], x_te, data["y_test"]


def make_synthetic(n_train: int = 2000, n_test: int = 500, dim: int = 784,
                   n_classes: int = 10, seed: int = 0):
    """Linearly-separable-ish blobs standing in for images."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_classes, dim))

    def gen(n):
        y = rng.integers(0, n_classes, size=n)
        x = centers[y] * 0.5 + rng.normal(0, 0.7, size=(n, dim))
        return x, y

    x_tr, y_tr = gen(n_train)
    x_te, y_te = gen(n_test)
    return x_tr, y_tr, x_te, y_te


def build_model(in_dim: int, hidden=(128, 64), n_classes: int = 10):
    net = Sequential(loss=SoftmaxCrossEntropy())
    prev = in_dim
    for h in hidden:
        net.add(Dense(prev, h, init="he_normal"))
        net.add(Activation("relu"))
        prev = h
    net.add(Dense(prev, n_classes, init="xavier_normal"))   # logits (softmax in loss)
    return net


def main():
    set_seed(42)

    data = load_mnist(sys.argv[1] if len(sys.argv) > 1 else None)
    if data is None:
        print("No MNIST file found -- using a synthetic dataset instead.\n"
              "  (set MNIST_NPZ=/path/to/mnist.npz to train on real digits)\n")
        x_tr, y_tr, x_te, y_te = make_synthetic()
    else:
        x_tr, y_tr, x_te, y_te = data
        print(f"Loaded MNIST: {x_tr.shape[0]} train / {x_te.shape[0]} test\n")

    # Hold out a validation split for monitoring.
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.1,
                                                rng=np.random.default_rng(0))

    net = build_model(x_tr.shape[1])
    print(net.summary())
    print()

    net.fit(x_tr, y_tr, epochs=10, batch_size=64, optimizer=Adam(lr=1e-3),
            validation_data=(x_val, y_val), verbose=True, log_every=1)

    # ---- final evaluation ----
    test_acc = accuracy(net.predict(x_te), y_te)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(net.predict(x_te), y_te))


if __name__ == "__main__":
    main()
