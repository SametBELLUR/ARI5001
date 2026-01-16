from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


class MLPBinary:
    """A minimal 1-hidden-layer MLP for binary classification (NumPy only)."""

    def __init__(self, d_in: int, h: int = 32, lr: float = 1e-3, l2: float = 1e-4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(d_in, h))
        self.b1 = np.zeros((1, h))
        self.W2 = rng.normal(0, 0.1, size=(h, 1))
        self.b2 = np.zeros((1, 1))
        self.lr = lr
        self.l2 = l2

    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        P = sigmoid(Z2)
        cache = (X, Z1, A1, Z2, P)
        return P, cache

    def bce(self, y: np.ndarray, p: np.ndarray, pos_weight: float = 1.0) -> float:
        eps = 1e-9
        p = np.clip(p, eps, 1.0 - eps)
        y = y.reshape(-1, 1)
        w = np.where(y == 1, pos_weight, 1.0)
        loss = -(w * y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(loss.mean())

    def step(self, cache, y: np.ndarray, pos_weight: float = 1.0):
        X, Z1, A1, Z2, P = cache
        y = y.reshape(-1, 1)
        n = X.shape[0]

        w = np.where(y == 1, pos_weight, 1.0)
        dZ2 = (w * (P - y)) / n

        dW2 = A1.T @ dZ2 + self.l2 * self.W2
        db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)

        dW1 = X.T @ dZ1 + self.l2 * self.W1
        db1 = dZ1.sum(axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p, _ = self.forward(X)
        return p.reshape(-1)

    def predict(self, X: np.ndarray, thr: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= thr).astype(int)
