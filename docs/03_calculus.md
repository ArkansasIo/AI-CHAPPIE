# Chapter 3 — Calculus & Backpropagation Derivations

## 1. The chain rule, single variable

```
y = f(g(x))
dy/dx = f'(g(x)) · g'(x)
```

## 2. The chain rule, vector form

If `z = f(a)` and `a = g(x)`, then with Jacobians:

```
∂z/∂x = (∂z/∂a)(∂a/∂x)
```

We rarely form these matrices explicitly. Instead we compute
**vector–Jacobian products**: given `∂J/∂z` (the upstream gradient), we get
`∂J/∂a = (∂J/∂z) · (∂z/∂a)`.

## 3. Reverse-mode automatic differentiation (why backprop is cheap)

For a scalar output `J` and `N` parameters, one reverse pass yields *all* `N`
partials in `O(cost of forward pass + backward pass)`, independent of `N`.
Forward-mode needed `N` passes. This is the single most important efficiency
fact behind deep learning.

## 4. Dense layer

Forward: `Z = X Wᵀ + b`.

Let `D = ∂J/∂Z`. Then:

```
∂J/∂W = Dᵀ X
∂J/∂b = Dᵀ 1  =  Σ_i D[i, :]
∂J/∂X = D W
```

**Derivation of `∂J/∂W`:**  `Z_ij = Σ_k X_ik W_jk + b_j`.

```
∂Z_ij/∂W_lk = δ_{jl} X_ik
∂J/∂W_lk = Σ_i Σ_j (∂J/∂Z_ij) ∂Z_ij/∂W_lk
         = Σ_i (∂J/∂Z_il) X_ik = (Dᵀ X)_lk     ∎
```

## 5. Activation derivatives

| Name | σ(z) | σ'(z) |
|------|------|-------|
| ReLU | `max(0, z)` | `1[z>0]` |
| Leaky ReLU | `z` if `z>0` else `αz` | `1` if `z>0` else `α` |
| Sigmoid | `1/(1+e^{-z})` | `σ(1−σ)` |
| Tanh | `tanh z` | `1 − tanh²z` |
| Softplus | `ln(1+e^z)` | `σ(z)` |
| ELU | `z` if `z>0` else `α(e^z−1)` | `1` if `z>0` else `α e^z` |
| GELU (approx) | `0.5z(1+tanh(√(2/π)(z+0.044715z³)))` | see code |

**Sigmoid derivation:**
```
σ'(z) = d/dz (1+e^{-z})^{-1}
      = (1+e^{-z})^{-2} e^{-z}
      = σ(z) · (1 − σ(z))        ∎
```

**ReLU derivation:** piecewise; subgradient `1[z>0]` at 0.

## 6. Softmax + cross-entropy — the famous simplification

Let logits `z ∈ R^K`, `p = softmax(z)`, target one-hot `y`.

```
J = −Σ_k y_k ln p_k
∂J/∂z = p − y
```

**Derivation:**  `∂p_i/∂z_j = p_i(δ_ij − p_j)` (softmax Jacobian).

```
∂J/∂z_j = −Σ_i y_i (1/p_i) ∂p_i/∂z_j
        = −Σ_i y_i (δ_ij − p_j)
        = −y_j + p_j Σ_i y_i
        = p_j − y_j                (since Σ y_i = 1)   ∎
```                (since Σ y_i = 1)   ∎
```

That clean `p − y` is why softmax is paired with cross-entropy.

## 7. Binary cross-entropy with sigmoid

`p = σ(z)`, `J = −[y ln p + (1−y) ln(1−p)]`.

```
∂J/∂z = p − y
```

## 8. Mean squared error

`J = (1/(2m)) Σ_i ‖ŷ_i − y_i‖²`, so `∂J/∂ŷ = (ŷ − y)/m`.

## 9. Backpropagation algorithm (general)

```
1. Forward: compute and cache a^0..a^L
2. Loss:    J = ℓ(a^L, y)
3. δ^L    = ∂J/∂z^L  (uses the fused loss+activation rule)
4. For l = L..1:
       dW^l = δ^l (a^{l-1})ᵀ
       db^l = δ^l  (summed over batch)
       δ^{l-1} = (W^l)ᵀ δ^l ⊙ σ'(z^{l-1})
5. Update: θ ← θ − η ∇J
```

**Why `δ^{l-1}` looks like that:** `z^{l} = W^l a^{l-1} + b`, so `∂z^l/∂a^{l-1} = W^l`,
and `a^{l-1}=σ(z^{l-1})`, so `∂a^{l-1}/∂z^{l-1}=σ'(z^{l-1})` — elementwise. Chain them.

## 10. Gradient checking

Numerically verify analytic gradients with central differences:

```
g_num = (J(θ+ε) − J(θ−ε)) / (2ε)
relative error = ‖g_num − g_ana‖ / (‖g_num‖ + ‖g_ana‖ + eps)
```

A relative error `< 1e-6` (double precision) indicates a correct gradient.
The project ships a `grad_check` utility that does exactly this.
