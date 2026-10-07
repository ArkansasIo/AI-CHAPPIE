"""
Example 2 -- Learn XOR using ONLY the reverse-mode autodiff engine.

No layers, no optimizers: we build the network by hand with `Tensor` ops, call
`.backward()`, and do the gradient-descent step ourselves. This shows exactly what
PyTorch-style frameworks automate.

    python examples/02_xor_autograd.py
"""

import numpy as np

from nn.autograd import Tensor
from nn.utils import set_seed


def main(epochs: int = 4000, lr: float = 0.5, seed: int = 0):
    set_seed(seed)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([0, 1, 1, 0], dtype=int)

    # Two output logits (binary classification via softmax over 2 classes).
    # Parameters (requires_grad=True so backward() fills .grad).
    W1 = Tensor(np.random.randn(8, 2) * 0.5, requires_grad=True)
    b1 = Tensor(np.zeros(8), requires_grad=True)
    W2 = Tensor(np.random.randn(2, 8) * 0.5, requires_grad=True)
    b2 = Tensor(np.zeros(2), requires_grad=True)
    params = [W1, b1, W2, b2]

    X_t = Tensor(X)

    for epoch in range(epochs + 1):
        # ---- forward ----
        z1 = X_t @ W1.T + b1         # (4, 8)
        a1 = z1.tanh()               # hidden activation
        z2 = a1 @ W2.T + b2          # (4, 2) logits
        loss = z2.cross_entropy(Y)   # softmax + NLL

        # ---- backward ----
        for p in params:
            p.zero_grad()
        loss.backward()

        # ---- manual SGD update ----
        for p in params:
            p.data -= lr * p.grad

        if epoch % 500 == 0:
            print(f"  epoch {epoch:4d} | loss {loss.item():.5f}")

    # ---- report ----
    z2 = ((Tensor(X) @ W1.T + b1).tanh() @ W2.T + b2).data
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    print("\nLearned probabilities (P[class=1]) and predictions:")
    for xi, yi, pi in zip(X.astype(int), Y, p[:, 1]):
        print(f"  {xi} -> target {yi} | p1={pi:.4f} | pred {int(pi >= 0.5)}")


if __name__ == "__main__":
    print("XOR trained purely with nn.autograd.Tensor\n")
    main()
