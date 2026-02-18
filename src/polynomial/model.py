# src/polynomial/model.py

import numpy as np

from src.common.types import FitResult, TrainConfig


class NumericalInstabilityError(Exception):
    """Raised when training becomes numerically unstable."""
    pass


def check_tolerance(losses, tolerance):
    if len(losses) < 2:
        return False
    prev = losses[-2]
    current = losses[-1]
    if prev == 0:
        return abs(current - prev) < tolerance
    return abs(current - prev) / abs(prev) < tolerance


class PolynomialRegressionGD:
    """
    1D Polynomial Regression trained with gradient descent.

    Model
    -----
    y = theta0 + theta1*x + theta2*x^2 + ... + thetad*x^d

    Notes
    -----
    - Internally standardizes x for training stability.
    - Parameters are learned in the scaled-x space.
    - Prediction uses the same stored scaling (mean/std) automatically.
    """

    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree

        self.theta = None  # shape: (degree+1,)

        # scaling parameters for x
        self._x_mean = None
        self._x_std = None

    def _poly_features(self, x_scaled: np.ndarray) -> np.ndarray:
        """
        Build design matrix with bias term:
        [1, x, x^2, ..., x^degree]
        """
        x_scaled = np.asarray(x_scaled, dtype=float).reshape(-1)
        # columns: power 0..degree
        return np.vstack([x_scaled ** p for p in range(self.degree + 1)]).T

    def fit(
        self,
        x,
        y,
        config: TrainConfig,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> FitResult:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
        if x.size == 0:
            raise ValueError("Empty input: x/y must contain at least one value.")

        n = x.size

        # Standardize x for stability
        x_mean = x.mean()
        x_std = x.std()
        if x_std == 0:
            raise NumericalInstabilityError("Standard deviation of x is zero, cannot normalize.")
        x_scaled = (x - x_mean) / x_std

        X = self._poly_features(x_scaled)  # shape (n, d+1)

        theta = np.zeros(self.degree + 1, dtype=float)
        losses = []
        failed = False

        for _ in range(config.epochs):
            y_hat = X @ theta
            r = y_hat - y

            loss = (1.0 / n) * (r @ r)
            losses.append(loss)

            if np.isnan(loss) or np.isinf(loss) or loss > 1e12:
                failed = True
                break

            # gradient for MSE: (2/n) * X^T r
            grad = (2.0 / n) * (X.T @ r)
            theta -= config.lr * grad

            if check_tolerance(losses, tolerance):
                break

        if failed:
            raise NumericalInstabilityError(
                "Training failed due to numerical instability. Try reducing learning rate and/or degree."
            )

        self.theta = theta
        self._x_mean = x_mean
        self._x_std = x_std

        if verbose:
            print(f"Converged in {len(losses)} epochs")

        return FitResult(
            epochs_run=len(losses),
            final_loss=losses[-1],
            loss_history=losses,
        )

    def predict(self, x):
        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")

        x = np.asarray(x, dtype=float).reshape(-1)

        x_scaled = (x - self._x_mean) / self._x_std
        X = self._poly_features(x_scaled)
        return X @ self.theta
