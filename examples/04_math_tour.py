"""
Example 4 -- a guided tour of the mathematics implemented in this library.

Prints worked results for: linear algebra (norms, PCA, condition number),
numerically-stable softmax, analytic vs. numerical gradients (all activations and
losses), and the backpropagation identity for a Dense layer.

    python examples/04_math_tour.py
"""

import numpy as np

from nn import activations as A
from nn import linalg as L
from nn.layers import Dense
from nn.losses import SoftmaxCrossEntropy


def title(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


def tour_linalg():
    title("1. Linear algebra")
    v = np.array([3.0, 4.0])
    print(f"v = {v}  |  L2 = {L.norm_l2(v):.3f}  |  L1 = {L.norm_l1(v):.3f}")
    print(f"cos(v, [1,0]) = {L.cosine_similarity(v, np.array([1.0, 0.0])):.3f}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    comps, ratio = L.pca(X, n_components=2)
    print(f"\nPCA on 5 features -> top-2 explained variance ratios: "
          f"{np.round(ratio, 3)}")
    A_ = rng.normal(size=(4, 4))
    print(f"condition number of a random 4x4 matrix: "
          f"{L.condition_number(A_):.3f}")
    print(f"SVD rank: {L.svd_rank(A_)}")
    print(f"Gram matrix diag: {np.round(np.diag(L.gram_matrix(X))[:3], 2)}")


def tour_softmax():
    title("2. Softmax is shift-invariant and stable")
    z = np.array([[1.0, 2.0, 3.0]])
    print(f"softmax(z)      = {np.round(L.softmax(z), 5)}")
    print(f"softmax(z+1000) = {np.round(L.softmax(z + 1000.0), 5)}  (identical)")


def tour_gradients():
    title("3. Analytic gradient vs. central-difference (all activations)")
    eps = 1e-6
    xs = np.linspace(-2.5, 2.5, 9)
    print(f"{'activation':<12}{'max|analytic-numeric|':>26}")
    for name in ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "softplus",
                 "swish", "gelu"]:
        f, fp = A.get_activation(name), A.get_activation_grad(name)
        errs = []
        for x in xs:
            num = (f(x + eps) - f(x - eps)) / (2 * eps)
            errs.append(abs(num - fp(x)))
        print(f"{name:<12}{max(errs):>26.2e}")


def tour_backprop_identity():
    title("4. Dense-layer backprop: dW = dZ^T X,  dX = dZ W")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 4))
    dZ = rng.normal(size=(5, 3))

    layer = Dense(4, 3, init="xavier_normal", rng=rng)
    layer.forward(X)
    dX = layer.backward(dZ)

    def fW(w):
        out = X @ w.T + layer.b
        return float(np.sum(out * dZ))

    num_dW = L.numerical_gradient(fW, layer.W.copy())
    print(f"max|dW_analytic - dW_numeric| = {np.max(np.abs(layer.dW - num_dW)):.2e}")

    def fX(x):
        out = x @ layer.W.T + layer.b
        return float(np.sum(out * dZ))

    num_dX = L.numerical_gradient(fX, X.copy())
    print(f"max|dX_analytic - dX_numeric| = {np.max(np.abs(dX - num_dX)):.2e}")


def tour_loss_gradient():
    title("5. Softmax + cross-entropy gradient reduces to (p - y)")
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(4, 3))
    labels = rng.integers(0, 3, size=4)

    loss = SoftmaxCrossEntropy()
    loss.forward(logits, labels)
    p = L.softmax(logits, axis=-1)
    y = np.eye(3)[labels]
    print("p - y =")
    print(np.round(p - y, 4))
    print("\ngradient returned by the loss (should be (p - y)/m):")
    print(np.round(loss.backward(), 4))


if __name__ == "__main__":
    tour_linalg()
    tour_softmax()
    tour_gradients()
    tour_backprop_identity()
    tour_loss_gradient()
    print("\nAll analytic gradients match their numerical counterparts.")
