# Chapter 6 — Architectures & Practical Notes

## 1. Multi-Layer Perceptron (MLP)

Alternating Dense + activation blocks, finished by a task-specific head:

* regression → linear output, MSE loss
* binary classification → sigmoid output, BCE loss
* multi-class → softmax output, cross-entropy

Depth increases the *composition* of features: layer 1 detects edges, layer 2
detects shapes, layer 3 detects objects (in vision). Each layer's features are
built from the previous layer's.

## 2. Training tricks baked into this library

* **Dropout** (`p`): randomly zero units during training, scale by `1/(1−p)` so
  the expected activation is unchanged. Disabled at inference.
* **BatchNorm**: normalize pre-activations per mini-batch; learn `γ`, `β`. At
  inference use running statistics. Speeds convergence dramatically.
* **Weight init**: He for ReLU, Xavier for tanh/sigmoid.
* **Gradient clipping**: rescale if `‖g‖ > threshold` — essential for RNNs.

## 3. Losses (paired with outputs)

| Task | Output | Loss |
|------|--------|------|
| Regression | linear | MSE |
| Binary | sigmoid | Binary cross-entropy |
| Multi-class | softmax | Categorical cross-entropy |

Using MSE with softmax gives vanishing gradients (`p−y` × small `p'`); the CE
pairing avoids it.

## 4. Evaluation metrics

* Accuracy, precision, recall, F1.
* Confusion matrix.
* For regression: MSE, MAE, R².

## 5. Beyond MLPs (the same math, new layers)

* **CNN**: convolution = weight sharing; the same backward identities but on
  sliding windows (`∂W = Σ patchesᵀ · dout`).
* **RNN/LSTM**: parameters shared across time; backprop-through-time is the
  chain rule unrolled. Prone to vanishing gradients → gated cells.
* **Transformers**: attention = softmax(QKᵀ/√d)V; multi-head; layer norm;
  residual streams. Same autograd, different primitives.

## 6. Debugging a network that won't learn

1. Can it overfit a single batch? If not, the bug is in the code (check
   gradients with `grad_check`).
2. Is the loss `NaN`? → too-high LR or `log(0)`. Clip and stabilize.
3. Dead ReLUs? → switch to Leaky ReLU or lower LR.
4. Gradients vanishing? → better init, normalization, residual connections.
5. Everything flat? → wrong loss/activation pairing or labels misaligned.

## 7. Reproducibility

Seed the RNG, log hyperparameters, save weights (`model.save`/`load`), and keep
a held-out validation set.
