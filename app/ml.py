"""Tiny pure-numpy replacements for the three scikit-learn pieces this
project uses (StandardScaler, Ridge, LogisticRegression). Deliberately not
a general-purpose ML library -- just enough, implemented plainly enough
that anyone can read exactly what the model does.

Why this exists instead of `pip install scikit-learn`: this project needs
to run in places where installing new packages isn't possible (locked-down
corporate/managed-device networks), but numpy is almost always already
available. Every environment this shipped to so far already had numpy, so
this module trades a bit of hand-rolled math for zero extra install steps.
"""
from __future__ import annotations

import numpy as np


class StandardScaler:
    """Same contract as sklearn's: fit(X) learns per-column mean/std,
    transform(X) standardizes. mean_ and scale_ are public, matching
    sklearn's attribute names, since forecast.py's contribution
    decomposition reads them directly."""

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RidgeRegression:
    """Closed-form ridge regression: w = (X'X + alpha*I)^-1 X'y, fit on an
    intercept-augmented, standardized design matrix (the caller standardizes
    via StandardScaler first, same as the sklearn Pipeline this replaces)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        A = Xb.T @ Xb + self.alpha * np.eye(d + 1)
        A[0, 0] -= self.alpha  # don't regularize the intercept
        b = Xb.T @ y
        coef = np.linalg.solve(A, b)
        self.intercept_ = float(coef[0])
        self.coef_ = coef[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


class LogisticRegression:
    """Binary logistic regression fit by batch gradient descent with L2
    regularization (sklearn's `C` is the inverse of this class's `alpha`,
    matching the convention forecast.py already used)."""

    def __init__(self, alpha: float = 2.0, lr: float = 0.5, iterations: int = 500):
        self.alpha = alpha
        self.lr = lr
        self.iterations = iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        y = y.astype(float)
        for _ in range(self.iterations):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad_w = X.T @ (p - y) / n + (self.alpha / n) * w
            grad_b = float(np.mean(p - y))
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = np.asarray([w])  # shape (1, d), matches sklearn's shape
        self.intercept_ = np.asarray([b])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef_[0] + self.intercept_[0]
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


class Pipeline:
    """Minimal stand-in for sklearn's Pipeline: just a scaler then an
    estimator, with the same .named_steps access forecast.py uses to reach
    into the fitted scaler/model for the contribution decomposition."""

    def __init__(self, steps: list[tuple[str, object]]):
        self.steps = steps
        self.named_steps = dict(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Pipeline":
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt)
        self.steps[-1][1].fit(Xt, y)
        return self

    def _transform_all_but_last(self, X: np.ndarray) -> np.ndarray:
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.steps[-1][1].predict(self._transform_all_but_last(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.steps[-1][1].predict_proba(self._transform_all_but_last(X))
