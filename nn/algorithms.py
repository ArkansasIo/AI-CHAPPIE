"""
nn.algorithms -- every algorithm in the project, in one place.

This module is the algorithmic spine that ties the systems together. It has four
sections:

1. **Core machine-learning algorithms** (the `nn` stack, from first principles)
   * `gradient_descent`, `stochastic_gradient_descent`, `minibatch_gradient_descent`
   * `momentum_descent`, `adam_descent`
   * `backprop_step` / `train_mlp`   (manual forward/backward, no autograd)
   * `linear_regression` (normal equation), `logistic_regression` (GD)
   * `kmeans`, `pca`, `knn_classify`, `knn_regress`
   * `perceptron`, `pocket_algorithm`
   * `decision_tree_stump`, `adaboost`

2. **Discrete / combinatorial algorithms**
   * search: `linear_search`, `binary_search`, `interpolation_search`
   * sort:   `bubble_sort`, `insertion_sort`, `merge_sort`, `quick_sort`,
             `heap_sort`, `counting_sort`
   * graph:  `bfs`, `dfs`, `dijkstra`, `bellman_ford`, `floyd_warshall`,
             `topological_sort`, `union_find` (Kruskal MST)
   * `dynamic_programming` helpers: `lcs`, `edit_distance`, `knapsack`

3. **Cross-system algorithms** (bridging gates, qubits, coordinates, language)
   * `learn_boolean_gate`      -- gates.py   (train an MLP to realize a gate)
   * `gate_to_neuron`          -- gates.py   (weights for AND/OR/NAND/NOT)
   * `nand_universal_builder`  -- gates.py   (build any gate from NAND only)
   * `deutsch_jozsa_algorithm` -- qubits.py
   * `grover_iteration`        -- qubits.py
   * `quantum_teleport`        -- qubits.py
   * `field_gradient_descent`  -- coordinates.py (descend a Values3D field)
   * `parse_and_conjugate`     -- grammar.py + conjugation.py

4. **Registry + complexity metadata**
   * `ALGORITHMS`        : name -> callable
   * `COMPLEXITY`        : name -> (time, space, notes)
   * `describe(name)`    : human-readable card for an algorithm
   * `list_algorithms()` : the full catalogue

Everything is NumPy + standard library only; every function is deterministic
given a seed.
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field
from math import inf, log2

import numpy as np

__all__ = [
    # --- registry ---
    "ALGORITHMS", "COMPLEXITY", "list_algorithms", "describe", "get_algorithm",
    # --- core ML ---
    "gradient_descent", "stochastic_gradient_descent",
    "minibatch_gradient_descent", "momentum_descent", "adam_descent",
    "backprop_step", "train_mlp", "linear_regression", "logistic_regression",
    "kmeans", "pca_reduce", "knn_classify", "knn_regress",
    "perceptron", "pocket_algorithm", "decision_tree_stump", "adaboost",
    # --- discrete ---
    "linear_search", "binary_search", "interpolation_search",
    "bubble_sort", "insertion_sort", "merge_sort", "quick_sort", "heap_sort",
    "counting_sort",
    "bfs", "dfs", "dijkstra", "bellman_ford", "floyd_warshall",
    "topological_sort", "union_find", "kruskal_mst",
    "lcs", "edit_distance", "knapsack",
    # --- cross-system ---
    "learn_boolean_gate", "gate_to_neuron", "nand_universal_builder",
    "deutsch_jozsa_algorithm", "grover_iteration", "quantum_teleport",
    "field_gradient_descent", "parse_and_conjugate",
]


# ===========================================================================
# 1. Core machine-learning algorithms
# ===========================================================================
def gradient_descent(grad, x0, lr: float = 0.1, steps: int = 100,
                     tol: float = 1e-10):
    """Vanilla batch gradient descent on a differentiable objective.

    Parameters
    ----------
    grad : callable(x) -> ndarray    (the gradient of the loss)
    x0   : ndarray                   (initial parameters)
    lr   : float                     (learning rate / step size)
    steps: int                       (iteration budget)
    tol  : float                     (stop when the step norm is below this)

    Returns
    -------
    dict with `x` (solution), `history` (loss proxy = ||grad||^2), `steps_taken`.

    Complexity: O(steps * cost(grad)) time, O(|x|) space.
    """
    x = np.array(x0, dtype=float)
    history = []
    for i in range(steps):
        g = np.asarray(grad(x), dtype=float)
        history.append(float(np.sum(g * g)))
        if np.linalg.norm(g) < tol:
            return {"x": x, "history": history, "steps_taken": i}
        x = x - lr * g
    return {"x": x, "history": history, "steps_taken": steps}


def stochastic_gradient_descent(loss_grad_i, x0, data, lr: float = 0.01,
                                epochs: int = 10, seed: int = 0):
    """SGD: one parameter update per example, in random order each epoch.

    `loss_grad_i(x, sample)` returns the gradient of the loss on one example.

    Complexity: O(epochs * n) updates, O(|x|) space.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    data = list(data)
    history = []
    for _ in range(epochs):
        idx = rng.permutation(len(data))
        total = 0.0
        for i in idx:
            g = np.asarray(loss_grad_i(x, data[i]), dtype=float)
            x = x - lr * g
            total += float(np.sum(g * g))
        history.append(total / max(len(data), 1))
    return {"x": x, "history": history}


def minibatch_gradient_descent(grad, x0, n_samples: int, batch_size: int = 32,
                               lr: float = 0.05, epochs: int = 20, seed: int = 0):
    """Mini-batch GD: average the gradient over a random batch each step.

    Complexity: O(epochs * n/batch * cost(grad)) time.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    history = []
    for _ in range(epochs):
        for start in range(0, n_samples, batch_size):
            idx = rng.choice(n_samples, size=min(batch_size, n_samples - start),
                             replace=False)
            g = np.asarray(grad(x, idx), dtype=float)
            x = x - lr * g
        history.append(float(np.linalg.norm(grad(x, np.arange(n_samples)))))
    return {"x": x, "history": history}


def momentum_descent(grad, x0, lr: float = 0.05, beta: float = 0.9,
                     steps: int = 100):
    """Heavy-ball momentum: `v <- beta*v + g; x <- x - lr*v`.

    Dampens oscillation along ravines; O(steps * cost(grad)).
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = []
    for _ in range(steps):
        g = np.asarray(grad(x), dtype=float)
        v = beta * v + g
        x = x - lr * v
        history.append(float(np.sum(g * g)))
    return {"x": x, "history": history}


def adam_descent(grad, x0, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, steps: int = 100):
    """Adam: bias-corrected first/second moment estimates.

    Complexity: O(steps * cost(grad)); O(|x|) extra space for m and v.
    """
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    for t in range(1, steps + 1):
        g = np.asarray(grad(x), dtype=float)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(float(np.sum(g * g)))
    return {"x": x, "history": history}


# ---------------------------------------------------------------------------
# Backpropagation from scratch (manual, no autograd)
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def backprop_step(X, Y, W1, b1, W2, b2, lr: float = 0.1):
    """One gradient-descent step of a 2-layer MLP with sigmoid activations.

    Network:  X -> (W1, b1) -> sigmoid -> (W2, b2) -> sigmoid -> loss (MSE)

    Backward pass (chain rule, see docs/03_calculus.md):
        dZ2 = (A2 - Y) * A2 * (1 - A2)
        dW2 = A1^T dZ2 / m         db2 = mean(dZ2)
        dA1 = dZ2 W2^T
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = X^T dZ1 / m          db1 = mean(dZ1)

    Returns `(loss, (W1, b1, W2, b2))`.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m = X.shape[0]

    # forward
    Z1 = X @ W1 + b1
    A1 = _sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = _sigmoid(Z2)

    loss = float(np.mean((A2 - Y) ** 2))

    # backward
    dZ2 = (A2 - Y) * A2 * (1 - A2)
    dW2 = A1.T @ dZ2 / m
    db2 = dZ2.mean(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = X.T @ dZ1 / m
    db1 = dZ1.mean(axis=0)

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return loss, (W1, b1, W2, b2)


def train_mlp(X, Y, hidden: int = 4, epochs: int = 2000, lr: float = 0.5,
              seed: int = 0):
    """Train a 2-layer sigmoid MLP with manual backprop.

    Returns `{'params': (W1,b1,W2,b2), 'history': [...], 'predict': fn}`.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
    n_in, n_out = X.shape[1], Y.shape[1]
    W1 = rng.normal(0, 1, (n_in, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1, (hidden, n_out))
    b2 = np.zeros(n_out)

    history = []
    for _ in range(epochs):
        loss, (W1, b1, W2, b2) = backprop_step(X, Y, W1, b1, W2, b2, lr)
        history.append(loss)

    def predict(x):
        a1 = _sigmoid(np.asarray(x, dtype=float) @ W1 + b1)
        return _sigmoid(a1 @ W2 + b2)

    return {"params": (W1, b1, W2, b2), "history": history, "predict": predict}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def linear_regression(X, y, ridge: float = 0.0):
    """Closed-form least squares (normal equation).

        theta = (X^T X + lambda I)^-1 X^T y

    `ridge` adds L2 regularization (lambda > 0 makes the inverse well-posed).

    Complexity: O(n * d^2 + d^3) time, O(d^2) space.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])       # bias column
    d = Xb.shape[1]
    A = Xb.T @ Xb + ridge * np.eye(d)
    theta = np.linalg.solve(A, Xb.T @ y)
    return {"intercept": float(theta[0]), "coef": theta[1:], "theta": theta,
            "predict": lambda Xn: np.hstack(
                [np.ones((np.asarray(Xn).shape[0], 1)), np.asarray(Xn)]) @ theta}


def logistic_regression(X, y, lr: float = 0.1, epochs: int = 500,
                        l2: float = 0.0, seed: int = 0):
    """Binary logistic regression trained by gradient descent.

    Model: p = sigmoid(X w + b); loss = BCE; gradient = X^T (p - y) / m + l2*w.

    Complexity: O(epochs * n * d).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0
    history = []
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        # clip to avoid log(0)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        loss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        history.append(loss)
        err = p - y
        gw = X.T @ err / n + l2 * w
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return {"w": w, "b": b, "history": history,
            "predict_proba": lambda Xn: _sigmoid(np.asarray(Xn) @ w + b),
            "predict": lambda Xn: (_sigmoid(np.asarray(Xn) @ w + b) >= 0.5).astype(int)}


# ---------------------------------------------------------------------------
# Unsupervised
# ---------------------------------------------------------------------------
def kmeans(X, k: int, iters: int = 100, seed: int = 0, tol: float = 1e-8):
    """k-means clustering (Lloyd's algorithm).

    Steps: init (k-means++), then alternate assign / update until convergence.

    Complexity: O(iters * n * k * d) time, O(n * d) space.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if k > n:
        raise ValueError("k cannot exceed the number of points")

    # k-means++ initialization
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        dist2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = dist2 / (dist2.sum() or 1.0)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assignment step
        d2 = np.stack([np.sum((X - c) ** 2, axis=1) for c in C], axis=1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update step
        new_C = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
            for j in range(k)])
        if np.linalg.norm(new_C - C) < tol:
            C = new_C
            break
        C = new_C

    inertia = float(sum(np.sum((X[i] - C[labels[i]]) ** 2) for i in range(n)))
    return {"centers": C, "labels": labels, "inertia": inertia}


def pca_reduce(X, n_components: int = 2):
    """Principal Component Analysis via SVD.

    Complexity: O(n * d^2) time.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    scores = Xc @ comps.T
    var = (S ** 2) / max(len(X) - 1, 1)
    return {"components": comps, "scores": scores, "mean": mean,
            "explained_variance_ratio": (var / (var.sum() or 1.0))[:n_components]}


def knn_classify(X_train, y_train, X_test, k: int = 3):
    """k-nearest-neighbour classification (Euclidean, majority vote).

    Complexity: O(n_test * n_train * d).
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    out = []
    for x in X_test:
        d2 = np.sum((X_train - x) ** 2, axis=1)
        idx = np.argsort(d2)[:k]
        votes = Counter(y_train[idx].tolist())
        out.append(votes.most_common(1)[0][0])
    return np.array(out)


def knn_regress(X_train, y_train, X_test, k: int = 3):
    """k-nearest-neighbour regression (mean of the k nearest targets)."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    out = []
    for x in X_test:
        d2 = np.sum((X_train - x) ** 2, axis=1)
        idx = np.argsort(d2)[:k]
        out.append(float(y_train[idx].mean()))
    return np.array(out)


# ---------------------------------------------------------------------------
# Classic learners
# ---------------------------------------------------------------------------
def perceptron(X, y, epochs: int = 100, lr: float = 1.0, seed: int = 0):
    """Rosenblatt perceptron (online, mistake-driven update).

    Guaranteed to converge iff the data is linearly separable.
    Complexity: O(epochs * n * d).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    y = np.where(y <= 0, -1.0, 1.0)          # use +/-1 labels
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    mistakes = []
    for _ in range(epochs):
        errs = 0
        for i in rng.permutation(n):
            if y[i] * (X[i] @ w + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                errs += 1
        mistakes.append(errs)
        if errs == 0:
            break
    return {"w": w, "b": b, "mistakes": mistakes,
            "predict": lambda Xn: np.where(np.asarray(Xn) @ w + b >= 0, 1, -1)}


def pocket_algorithm(X, y, epochs: int = 100, lr: float = 1.0, seed: int = 0):
    """Pocket perceptron: keep the best weights seen (works when not separable)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    y = np.where(y <= 0, -1.0, 1.0)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    best_w, best_b, best_score = w.copy(), b, -1
    for _ in range(epochs):
        for i in rng.permutation(n):
            if y[i] * (X[i] @ w + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
        score = int(np.sum(np.sign(X @ w + b) == y))
        if score > best_score:
            best_w, best_b, best_score = w.copy(), b, score
    return {"w": best_w, "b": best_b, "accuracy": best_score / n}


def decision_tree_stump(X, y, criterion: str = "gini"):
    """A depth-1 decision tree (one split), the AdaBoost base learner.

    Chooses the feature/threshold minimizing weighted impurity.
    Complexity: O(n * d * n_thresholds).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    classes = np.unique(y)

    def impurity(left, right):
        if len(left) == 0 or len(right) == 0:
            return inf
        total = len(left) + len(right)

        def score(a):
            if criterion == "gini":
                p = np.array([np.mean(a == c) for c in classes])
                return 1 - np.sum(p ** 2)
            # entropy
            p = np.array([np.mean(a == c) for c in classes])
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        return (len(left) / total) * score(left) + (len(right) / total) * score(right)

    best = {"impurity": inf, "feature": 0, "threshold": 0.0, "polarity": 1}
    for f in range(d):
        vals = np.unique(X[:, f])
        thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
        for t in thresholds:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]
            imp = impurity(left, right)
            if imp < best["impurity"]:
                # majority class on each side
                lc = Counter(left.tolist()).most_common(1)[0][0] if len(left) else 0
                rc = Counter(right.tolist()).most_common(1)[0][0] if len(right) else 0
                best = {"impurity": imp, "feature": f, "threshold": float(t),
                        "left_class": lc, "right_class": rc}
    return best


def adaboost(X, y, n_estimators: int = 10):
    """AdaBoost with decision stumps (SAMME algorithm for binary labels).

    Complexity: O(T * n * d * n_thresholds).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(X)
    w = np.ones(n) / n
    classes = np.unique(y)
    models = []
    alphas = []

    for _ in range(n_estimators):
        stump = decision_tree_stump_weighted(X, y, w)
        pred = np.where(X[:, stump["feature"]] <= stump["threshold"],
                        stump["left_class"], stump["right_class"])
        err = float(np.sum(w * (pred != y)) / np.sum(w))
        err = min(max(err, 1e-10), 1 - 1e-10)
        alpha = 0.5 * log2((1 - err) / err)
        w = w * np.exp(-alpha * np.where(pred == y, 1, -1))
        w = w / w.sum()
        models.append(stump)
        alphas.append(alpha)
        if err < 1e-10:
            break

    def predict(Xn):
        Xn = np.asarray(Xn, dtype=float)
        agg = np.zeros(len(Xn))
        for stump, alpha in zip(models, alphas):
            p = np.where(Xn[:, stump["feature"]] <= stump["threshold"],
                         stump["left_class"], stump["right_class"])
            agg += alpha * np.where(p == classes[0], -1.0, 1.0)
        return np.where(agg >= 0, classes[-1], classes[0])

    return {"models": models, "alphas": alphas, "predict": predict,
            "classes": classes}


def decision_tree_stump_weighted(X, y, w):
    """Weighted decision stump (used by AdaBoost)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    classes = np.unique(y)
    best = {"error": inf, "feature": 0, "threshold": 0.0,
            "left_class": classes[0], "right_class": classes[-1]}
    for f in range(d):
        vals = np.unique(X[:, f])
        thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
        for t in thresholds:
            left_mask = X[:, f] <= t
            for lc in classes:
                for rc in classes:
                    pred = np.where(left_mask, lc, rc)
                    err = float(np.sum(w * (pred != y)))
                    if err < best["error"]:
                        best = {"error": err, "feature": f, "threshold": float(t),
                                "left_class": lc, "right_class": rc}
    return best
