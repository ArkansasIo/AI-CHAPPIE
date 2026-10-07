# Chapter 2 — Linear Algebra of Neural Networks

## 1. Vectors, matrices, shapes

Everything is a tensor. We adopt the convention that a **batch** is stored
column-major relative to the math, but for NumPy we store it row-major:

* `X` has shape `(m, n)` — `m` examples, `n` features.
* `W` has shape `(n_out, n_in)`.
* `Z = X Wᵀ + b` has shape `(m, n_out)`.

The bias `b` broadcasts across the batch. Equivalently, the per-example view is
`z_i = W x_i + b`.

## 2. Matrix multiplication as a sum of outer products

```
Z[i, :] = Σ_k X[i, k] · W[:, k]      (row = sum over features of scalings)
```

Each example is a linear combination of the **columns of Wᵀ**, i.e. each
neuron computes a dot product ("projection") of its weight vector with the
input.

## 3. The dot product, norms, and geometry

* Dot product: `⟨u, v⟩ = uᵀv = Σ u_i v_i`.
* `L2` norm: `‖u‖₂ = √(uᵀu)`.
* Cosine similarity: `⟨u,v⟩ / (‖u‖‖v‖)` — the angle between vectors.

A neuron's pre-activation is `‖w‖‖x‖cosθ + b`: it fires hardest when the input
points along its weight direction.

## 4. Linear maps and Jacobians

For `y = Wx`, the Jacobian is `∂y/∂x = W`. For `z = XWᵀ`, the differential is

```
dz = dX Wᵀ  (for dX)
dz = dW X   (for dW)   … transposed appropriately
```

These two identities are the **entire** backward formula for a dense layer:

```
dW = dZᵀ X
dX = dZ  W
db = Σ_i dZ[i, :]
```

## 5. Rank, eigenvectors, and why initialization matters

A product of `k` weight matrices has rank ≤ `min` of them; if `k` is large,
the product tends toward degenerate singular-value spectra — **vanishing or
exploding gradients**. This motivates:

* Xavier/Glorot init: `Var(W) = 2 / (n_in + n_out)`
* He init: `Var(W) = 2 / n_in` (for ReLU)

so that the forward signal variance is preserved layer to layer.

## 6. SVD view of a layer

Any `W = U Σ Vᵀ`. A dense layer rotates (`Vᵀ`), scales (`Σ`), rotates back
(`U`), then adds `b`. Training reshapes `Σ` — it controls the "gain" along
principal directions of the layer's input space.

## 7. The covariance / Gram matrix

`G = XᵀX` (shape `n×n`) summarizes feature second moments. Its eigenvectors
are principal component directions (PCA). Whitening (decorrelating inputs)
often makes optimization much better-conditioned — the reason we standardize
inputs.

## 8. Softmax, probabilities, and the simplex

`softmax(z)_i = exp(z_i) / Σ_j exp(z_j)`. It maps `R^K` onto the probability
simplex (nonnegative, sums to 1) and is shift-invariant: `softmax(z+c)=softmax(z)`.
Subtracting `max(z)` is a numerically stable, mathematically identical trick.

## 9. Key identities used throughout

```
(W x)ᵀ = xᵀ Wᵀ
(A B)ᵀ = Bᵀ Aᵀ
∂/∂A  aᵀ A b   = a bᵀ            (scalar chain element)
∂/∂A  ½‖A‖²_F  = A
(A ⊙ B) row-sum →  Σ over broadcast axis
```

These identities are exactly what let each layer implement `backward` in
closed form.
