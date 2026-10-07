# Chapter 1 — Theory of Neural Networks

## 1. What a neural network is

A neural network is a **parameterized function** that maps an input vector
`x ∈ R^n` to an output vector `ŷ ∈ R^m`:

```
ŷ = f(x; θ)
```

where `θ` are the *learnable parameters* (weights `W` and biases `b`). The
function is a composition of *layers*:

```
ŷ = f_L( f_{L-1}( ... f_1(x) ))
```

Each layer applies an **affine transform** followed by a **nonlinearity**:

```
z      = W a^{l-1} + b        (pre-activation)
a^l    = σ(z)                 (activation)
```

The nonlinearity is essential: without it a stack of affine layers collapses
into a single affine layer (a matrix product is still a matrix).

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| `L` | number of layers |
| `W^l` | weight matrix of layer `l`, shape `(n_out, n_in)` |
| `b^l` | bias vector of layer `l`, shape `(n_out, 1)` |
| `z^l` | pre-activation, `W^l a^{l-1} + b^l` |
| `a^l` | activation, `σ(z^l)`; `a^0 = x` |
| `m` | number of examples in a mini-batch |
| `J` | cost / loss function |
| `η` | learning rate |

## 3. The universal approximation idea

A network with a single sufficiently wide hidden layer can approximate any
continuous function on a compact domain to arbitrary accuracy. Depth usually
buys *efficiency*: many functions need exponentially fewer units when
represented deeply rather than shallowly.

## 4. Learning = optimization

We choose `θ` to minimize an empirical risk:

```
J(θ) = (1/m) Σ_{i=1..m} ℓ( f(x_i; θ), y_i ) + λ Ω(θ)
```

* `ℓ` is the per-example loss (e.g. MSE, cross-entropy).
* `Ω(θ)` is a regularizer (e.g. L2 = `½‖W‖²`).
* `λ` controls regularization strength.

We minimize with **gradient-based** methods: compute `∇_θ J` and step downhill.

```
θ ← θ − η ∇_θ J
```

## 5. Bias–variance, capacity, and generalization

* **Underfitting** (high bias): model too small / too little training.
* **Overfitting** (high variance): model memorizes training noise.

Tools to control capacity:

* weight decay (L2/L1), dropout, early stopping, data augmentation,
* batch normalization (smooths the loss landscape, speeds training),
* more data.

## 6. Why gradients, not random search

The gradient points in the direction of steepest ascent. Moving opposite to it
is the locally optimal first-order step. For a parameter vector of millions of
values, gradient descent costs one forward + one backward pass — far cheaper
than finite differences (which need one forward pass *per parameter*).

## 7. The forward / backward contract

Every layer must provide two functions:

1. `forward(x)` → output (caches what it needs)
2. `backward(dout)` → `(dx, dW, db)` — the local **Jacobian-vector products**

Backpropagation is just the chain rule applied in reverse over the
computational graph (see Chapter 5).

## 8. A note on the training loop

```
for epoch in range(E):
    for (X, Y) in minibatches(data):
        Yhat = model.forward(X)        # forward pass
        loss = criterion(Yhat, Y)      # loss
        dout = criterion.backward()    # dJ/dYhat
        model.backward(dout)           # backprop
        optimizer.step()               # θ ← θ − η g
        model.zero_grad()
```

Everything else (convergence, regularization, scheduling) is a variation of
this loop.
