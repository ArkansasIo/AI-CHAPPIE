"""
Example 1 -- Learn XOR with the hand-written layer + optimizer stack.

XOR is the canonical non-linearly-separable problem: a single linear model
*cannot* solve it, so it demonstrates why hidden layers + nonlinearities matter.

Run from the project root:

    python examples/01_xor_manual.py
"""

import numpy as np

from nn.model import Sequential, MLP
from nn.layers import Dense, Activation
from nn.losses import SigmoidBinaryCrossEntropy
from nn.optimizers import Adam
from nn.utils import set_seed

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)   # XOR truth table


def demo_mlp_builder():
    print("=" * 60)
    print("XOR via MLP builder (Dense -> tanh -> Dense -> sigmoid)")
    print("=" * 60)
    set_seed(0)
    net = MLP([2, 8, 8, 1], activations="tanh", out_activation="sigmoid",
              init="xavier_normal", loss="sigmoid_bce", seed=3)
    print(net.summary())

    net.fit(X, Y, epochs=3000, batch_size=4, optimizer="adam", lr=0.1,
            loss="sigmoid_bce", verbose=True, log_every=500)

    probs = net.predict(X)
    print("\nPredictions:")
    for xi, yi, pi in zip(X, Y.ravel(), probs.ravel()):
        print(f"  {xi.astype(int)} -> target {int(yi)} | p={pi:.4f} "
              f"| class {int(pi >= 0.5)}")
    return net


def demo_manual_stack():
    print("\n" + "=" * 60)
    print("XOR via explicit layer stack + manual training loop")
    print("=" * 60)
    set_seed(1)

    net = Sequential(loss=SigmoidBinaryCrossEntropy())
    net.add(Dense(2, 4, init="xavier_normal"))
    net.add(Activation("tanh"))
    net.add(Dense(4, 1, init="xavier_normal"))
    net.add(Activation("sigmoid"))
    opt = Adam(lr=0.05)

    for step in range(6000):
        loss = net.train_step(X, Y, opt)
        if step % 800 == 0:
            print(f"  step {step:4d} | loss {loss:.5f}")

    print("\nFinal probabilities:", np.round(net.predict(X).ravel(), 4))
    print("Rounded classes    :", net.predict_classes(X))
    return net


if __name__ == "__main__":
    demo_mlp_builder()
    demo_manual_stack()
    print("\nDone. A linear model gets 50% here; the hidden layer gets 100%.")
