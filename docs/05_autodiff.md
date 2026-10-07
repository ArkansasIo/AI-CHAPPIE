# Chapter 5 — Automatic Differentiation (the engine behind deep learning)

## 1. Computational graphs

Every expression becomes a DAG: leaves are inputs/parameters, internal nodes
are operations, the root is the scalar loss. Example:

```
J = (W x + b − y)²
```

```
      y   x   W   b
       \   \  |  /
        \   \ | /
        sub  matmul
           \  /
           add
            |
          square
            |
            J
```

## 2. The two modes

* **Forward mode**: push derivatives from inputs toward outputs. Cost ∝ number
  of *inputs*. Great when few inputs, many outputs.
* **Reverse mode** (= backprop): push from the output toward inputs. Cost ∝
  number of *outputs*. Since our loss is a scalar (1 output), this is optimal —
  one pass gives the derivative w.r.t. every parameter.

## 3. The `Tensor` abstraction

Each node stores:
* `.data` — the forward value
* `.grad` — accumulated `∂J/∂self` after backward
* `._backward` — a closure that pushes grad to its inputs
* `._prev` — the parents, used for topological ordering

## 4. Building the graph (`__add__`, `__mul__`, `matmul`, `exp`, ...)

Each operation does two things:

1. computes the forward value and records the parents,
2. defines a `_backward` closure that, given `∂J/∂out`, computes the local
   chain contributions to each parent, e.g. for `C = A @ B`:

```
A.grad += C.grad @ B.dataᵀ
B.grad += A.dataᵀ @ C.grad
```

The gradients use `+=` because a node may feed into several downstream paths
(multivariate chain rule: sum over paths).

## 5. The topological backward pass

```
topo = topological_sort(root)      # parents before children
root.grad = 1                      # dJ/dJ = 1
for node in reversed(topo):
    node._backward()               # push grad to parents
```

Topological order is what makes a single reverse sweep correct even when the
graph has shared subexpressions and branches.

## 6. Broadcasting (very important)

If a tensor of shape `(m, n)` is added to one of shape `(m, 1)`, the gradient
of the smaller tensor must be **summed** over the broadcast axis. Otherwise
shapes mismatch. Our `_unbroadcast` does this:

```
while grad.ndim > x.ndim: grad = grad.sum(0 or axis)
for axes where x.shape[a]==1: grad = grad.sum(axis=a, keepdims=True)
```

## 7. Numerical stability tricks used in this engine

* `softmax`: subtract the row max before `exp`.
* `log`: clamp the argument away from 0.
* `logsumexp` for stable log-softmax.
* gradient checking with central differences to validate every op.

## 8. Manual layers vs. autograd

This project ships **both**:

* `nn/layers.py` — hand-written forward/backward (fast, explicit, educational).
* `nn/autograd.py` — a general reverse-mode engine (compose any op and call
  `.backward()`).

They should agree; `tests/test_gradients.py` checks the manual gradients with
finite differences, and `examples/02_xor_autograd.py` trains a net using only
the graph engine.
