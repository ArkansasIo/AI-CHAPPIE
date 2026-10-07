# Chapter 4 — Optimization: How Training Actually Converges

## 1. Batch, stochastic, mini-batch

* **Batch GD**: one step per full dataset. Accurate gradient, slow.
* **SGD**: one step per example. Noisy, but the noise helps escape saddles.
* **Mini-batch** (what we use): a compromise; vectorizes on hardware and the
  gradient noise decreases as batch size grows (`∝ 1/b`).

## 2. Convergence intuition

For a smooth, convex `J` with `L`-Lipschitz gradient, gradient descent with
`η < 2/L` converges at rate `O(1/k)`. For non-convex losses (all real networks)
we settle for convergence to a **stationary point**, and in practice the
interesting minimum is a good one because most local minima of large networks
are close in value.

## 3. Learning rate — the single most important hyperparameter

* too large → divergence / oscillation, `NaN`
* too small → glacial progress
* **schedules**: step decay, exponential, cosine, warmup+decay.

## 4. Momentum methods

**Momentum / heavy ball**
```
v ← β v + g
θ ← θ − η v
```
Accumulates velocity along consistent directions, cancels oscillation.

**Nesterov** looks ahead: evaluate the gradient at `θ − ηβv`.

## 5. Adaptive methods

**RMSProp** divides by a running RMS of the gradient:
```
s ← ρ s + (1−ρ) g²
θ ← θ − η g / (√s + ε)
```

**Adam** = momentum + RMSProp + bias correction:
```
m ← β₁ m + (1−β₁) g
v ← β₂ v + (1−β₂) g²
m̂ = m / (1−β₁ᵗ),   v̂ = v / (1−β₂ᵗ)
θ ← θ − η m̂ / (√v̂ + ε)
```
Bias correction matters early in training when `m`,`v` are initialized to 0.

**AdamW** decouples weight decay from the adaptive step:
`θ ← θ − η(m̂/(√v̂+ε) + λ θ)`.

## 6. Loss landscape pathologies

* **saddle points** dominate in high dimension — momentum and noise help.
* **ravines**: steep in one direction, shallow in another → momentum/Adam.
* **ill-conditioning**: fix with normalization, whitening, or second-order info.

## 7. Regularization as optimization

* L2 weight decay shrinks weights toward 0 → smoother function.
* Dropout injects multiplicative noise → ensembling effect.
* BatchNorm reparametrizes the landscape → larger usable learning rates.

## 8. Practical recipe

1. Standardize inputs.
2. He/Xavier init.
3. Adam with `η=1e-3` as a default; SGD+momentum for the last drop of accuracy.
4. Mini-batch 32–256.
5. Early stopping on a validation set.
