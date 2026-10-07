"""
Gradient / correctness tests for the neural network library.

Run:
    pytest -q
or
    python tests/test_math.py        (falls back to a tiny built-in runner)
"""

import numpy as np

from nn import activations as A
from nn import linalg as L
from nn.layers import BatchNorm, Dense, Dropout
from nn.losses import (CategoricalCrossEntropy, MeanSquaredError,
                       SigmoidBinaryCrossEntropy, SoftmaxCrossEntropy)
from nn.model import MLP, Sequential
from nn.optimizers import Adam, SGD, get_optimizer


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def test_softmax_sums_to_one():
    z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    p = L.softmax(z, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_softmax_shift_invariance():
    z = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(L.softmax(z), L.softmax(z + 100.0))


def test_logsumexp_matches_naive():
    z = np.array([[1.0, 2.0, 3.0]])
    naive = np.log(np.exp(z).sum(axis=-1))
    assert np.allclose(L.logsumexp(z, axis=-1), naive)


def test_norms():
    v = np.array([3.0, 4.0])
    assert abs(L.norm_l2(v) - 5.0) < 1e-12
    assert abs(L.norm_l1(v) - 7.0) < 1e-12


def test_one_hot():
    out = L.one_hot(np.array([0, 2, 1]), num_classes=3)
    assert np.array_equal(out, np.eye(3)[[0, 2, 1]])


# ---------------------------------------------------------------------------
# Activation derivatives vs finite differences
# ---------------------------------------------------------------------------
def _check_activation(name, xs=None):
    f = A.get_activation(name)
    fprime = A.get_activation_grad(name)
    xs = xs if xs is not None else np.linspace(-3, 3, 21)
    eps = 1e-6
    for x in xs:
        num = (f(x + eps) - f(x - eps)) / (2 * eps)
        ana = fprime(x)
        assert abs(num - ana) < 1e-4, (name, x, num, ana)


def test_activation_derivatives():
    # relu and leaky_relu are non-differentiable at 0 -> test away from it
    axis = np.linspace(-3, 3, 21)
    away_from_zero = axis[np.abs(axis) > 0.2]
    _check_activation("relu", away_from_zero)
    _check_activation("leaky_relu", away_from_zero)
    for name in ["sigmoid", "tanh", "softplus", "elu", "swish", "gelu"]:
        _check_activation(name)


# ---------------------------------------------------------------------------
# Layer gradient checking
# ---------------------------------------------------------------------------
def test_dense_gradients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    dout = rng.normal(size=(5, 3))

    layer = Dense(4, 3, init="xavier_normal", rng=rng)
    layer.forward(x)
    dx = layer.backward(dout)

    eps = 1e-6

    # gradient wrt W
    for i in range(3):
        for j in range(4):
            old = layer.W[i, j]
            layer.W[i, j] = old + eps
            fp = np.sum(layer.forward(x) * dout)
            layer.W[i, j] = old - eps
            fm = np.sum(layer.forward(x) * dout)
            layer.W[i, j] = old
            assert abs((fp - fm) / (2 * eps) - layer.dW[i, j]) < 1e-5

    # gradient wrt b
    for i in range(3):
        old = layer.b[i]
        layer.b[i] = old + eps
        fp = np.sum(layer.forward(x) * dout)
        layer.b[i] = old - eps
        fm = np.sum(layer.forward(x) * dout)
        layer.b[i] = old
        assert abs((fp - fm) / (2 * eps) - layer.db[i]) < 1e-5

    # gradient wrt input
    for a in range(5):
        for j in range(4):
            old = x[a, j]
            x[a, j] = old + eps
            fp = np.sum(layer.forward(x) * dout)
            x[a, j] = old - eps
            fm = np.sum(layer.forward(x) * dout)
            x[a, j] = old
            assert abs((fp - fm) / (2 * eps) - dx[a, j]) < 1e-5


def test_batchnorm_gradients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 4))
    dout = rng.normal(size=(6, 4))
    bn = BatchNorm(4)
    bn.train(True)
    bn.forward(x)
    dx = bn.backward(dout)

    def loss_at(xx):
        out = bn.forward(xx)
        return float(np.sum(out * dout))

    num = L.numerical_gradient(loss_at, x.copy(), eps=1e-6)
    assert np.allclose(dx, num, atol=1e-4)


# ---------------------------------------------------------------------------
# Loss gradient checking
# ---------------------------------------------------------------------------
def test_mse_gradients():
    rng = np.random.default_rng(2)
    yp = rng.normal(size=(4, 3))
    yt = rng.normal(size=(4, 3))
    loss = MeanSquaredError()
    loss.forward(yp, yt)
    g = loss.backward()

    def f(p):
        return MeanSquaredError().forward(p, yt)

    num = L.numerical_gradient(f, yp.copy(), eps=1e-6)
    assert np.allclose(g, num, atol=1e-6)


def test_softmax_ce_gradients():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(5, 4))
    labels = rng.integers(0, 4, size=5)
    loss = SoftmaxCrossEntropy()
    loss.forward(logits, labels)
    g = loss.backward()

    def f(z):
        return SoftmaxCrossEntropy().forward(z, labels)

    num = L.numerical_gradient(f, logits.copy(), eps=1e-6)
    assert np.allclose(g, num, atol=1e-5)


def test_sigmoid_bce_gradients():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(6, 1))
    y = rng.integers(0, 2, size=(6, 1)).astype(float)
    loss = SigmoidBinaryCrossEntropy()
    loss.forward(logits, y)
    g = loss.backward()

    def f(z):
        return SigmoidBinaryCrossEntropy().forward(z, y)

    num = L.numerical_gradient(f, logits.copy(), eps=1e-6)
    assert np.allclose(g, num, atol=1e-5)


def test_categorical_ce_gradients():
    rng = np.random.default_rng(5)
    p = L.softmax(rng.normal(size=(4, 3)), axis=-1)
    y = np.eye(3)[rng.integers(0, 3, size=4)]
    loss = CategoricalCrossEntropy()
    loss.forward(p, y)
    g = loss.backward()

    def f(pp):
        return CategoricalCrossEntropy().forward(pp, y)

    num = L.numerical_gradient(f, p.copy(), eps=1e-6)
    assert np.allclose(g, num, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------
def test_xor_learns():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    net = MLP([2, 16, 16, 1], activations="tanh", out_activation="sigmoid",
              init="xavier_normal", loss="sigmoid_bce", seed=7)
    net.fit(X, Y, epochs=3000, batch_size=4, optimizer=Adam(lr=0.05),
            loss="sigmoid_bce", verbose=False)
    probs = net.predict(X).ravel()
    preds = (probs >= 0.5).astype(int)
    assert np.array_equal(preds, Y.ravel().astype(int)), probs


def test_optimizers_run():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    for name in ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw"]:
        net = MLP([2, 6, 1], activations="tanh", out_activation="sigmoid",
                  init="xavier_normal", loss="sigmoid_bce", seed=1)
        opt = get_optimizer(name, lr=0.05)
        for _ in range(50):
            net.train_step(X, Y, opt)
        # Just needs to produce a finite loss.
        assert np.isfinite(net.loss.forward(net.forward(X), Y))


def test_dropout_train_vs_eval():
    rng = np.random.default_rng(0)
    x = np.ones((100, 10))
    d = Dropout(0.5, rng=rng)
    d.train(True)
    out_train = d.forward(x)
    d.train(False)
    out_eval = d.forward(x)
    assert np.array_equal(out_eval, x)
    # Mean preserved in expectation (inverted dropout).
    assert abs(out_train.mean() - 1.0) < 0.15


def test_grad_check_utility():
    def f(p):
        return float(np.sum(p ** 2))

    def g(p):
        return 2 * p

    p = np.random.default_rng(0).normal(size=5)
    res = L.grad_check(f, g, p)
    assert res["ok"], res["relative_error"]


# ---------------------------------------------------------------------------
# Minimal runner (used when pytest is not installed)
# ---------------------------------------------------------------------------
def _run_all():
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
