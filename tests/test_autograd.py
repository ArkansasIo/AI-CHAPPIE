t"""Tests for the reverse-mode autodiff engine (nn.autograd)."""

import numpy as np

from nn.autograd import Tensor
from nn import linalg as L


def _num_grad(f, x, eps=1e-6):
    return L.numerical_gradient(f, np.array(x, dtype=float), eps=eps)


def test_add_mul_grads():
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    c = a * b + a          # dc/da = b + 1 = 3 ; dc/db = a = 3
    c.backward()
    assert abs(a.grad - 3.0) < 1e-9
    assert abs(b.grad - 3.0) < 1e-9


def test_shared_subexpression():
    a = Tensor(2.0, requires_grad=True)
    y = a * a              # dy/da = 2a = 4 (tests gradient accumulation)
    y.backward()
    assert abs(a.grad - 4.0) < 1e-9


def test_matmul_grads():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 2))

    ta = Tensor(A, requires_grad=True)
    tb = Tensor(B, requires_grad=True)
    out = (ta @ tb).sum()
    out.backward()

    def fA(x):
        return float(np.sum(x @ B))
    def fB(x):
        return float(np.sum(A @ x))

    assert np.allclose(ta.grad, _num_grad(fA, A), atol=1e-5)
    assert np.allclose(tb.grad, _num_grad(fB, B), atol=1e-5)


def test_broadcast_bias_grad():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 3))
    b = rng.normal(size=3)              # broadcast across rows

    tX = Tensor(X, requires_grad=True)
    tb = Tensor(b, requires_grad=True)
    (tX + tb).sum().backward()

    # d/db of sum(X + b) = number of rows = 5 for each feature
    assert np.allclose(tb.grad, np.full(3, 5.0))


def test_tanh_and_sigmoid_grads():
    x = 0.3
    t = Tensor(x, requires_grad=True)
    t.tanh().backward()
    assert abs(t.grad - (1 - np.tanh(x) ** 2)) < 1e-9

    s = Tensor(x, requires_grad=True)
    s.sigmoid().backward()
    sig = 1 / (1 + np.exp(-x))
    assert abs(s.grad - sig * (1 - sig)) < 1e-9


def test_relu_grad():
    x = np.array([-1.0, 2.0, -0.5, 3.0])
    t = Tensor(x, requires_grad=True)
    t.relu().sum().backward()
    assert np.array_equal(t.grad, (x > 0).astype(float))


def test_softmax_cross_entropy_matches_manual():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(5, 4))
    labels = rng.integers(0, 4, size=5)

    t = Tensor(logits, requires_grad=True)
    loss = t.cross_entropy(labels)
    loss.backward()

    # manual: softmax then -log
    p = L.softmax(logits, axis=-1)
    y = np.eye(4)[labels]
    nll = -np.sum(y * np.log(p)) / 5
    assert abs(loss.item() - nll) < 1e-8

    def f(z):
        pp = L.softmax(z, axis=-1)
        return float(-np.sum(y * np.log(pp)) / 5)

    assert np.allclose(t.grad, _num_grad(f, logits), atol=1e-5)


def test_chain_of_ops():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 3))
    W = rng.normal(size=(2, 3))

    def build(xx):
        tx = Tensor(xx, requires_grad=True)
        tw = Tensor(W, requires_grad=True)
        h = (tx @ tw.T).tanh()
        out = (h * h).sum()
        out.backward()
        return tx, tw

    tx, tw = build(x)

    def fX(xx):
        h = np.tanh(xx @ W.T)
        return float(np.sum(h * h))
    def fW(ww):
        h = np.tanh(x @ ww.T)
        return float(np.sum(h * h))

    assert np.allclose(tx.grad, _num_grad(fX, x), atol=1e-5)
    assert np.allclose(tw.grad, _num_grad(fW, W), atol=1e-5)


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
