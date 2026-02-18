import numpy as np
import pandas as pd

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


class LinearRegressionGD:
    """
    Linear regression trained with gradient descent.
    Model: y = a*x + b
    """

    def __init__(self):
        self.a = None  # slope (original scale)
        self.b = None  # intercept (original scale)

        # store normalization params for reference (not required after rescale)
        self._x_mean = None
        self._x_std = None

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
        n = len(x)
        
        x_mean = x.mean()
        x_std = np.std(x)
        if x_std == 0:
            raise NumericalInstabilityError(
                "Standard deviation of x is zero, cannot normalize."
            )

        x_scaled = (x - x_mean) / x_std

        losses = []
        a = 0.0
        b = 0.0
        failed = False

        for epoch in range(config.epochs):
            y_hat = a * x_scaled + b
            r = y_hat - y

            loss = (1 / n) * r.dot(r)
            losses.append(loss)

            if np.isnan(loss) or np.isinf(loss) or loss > 1e12:
                failed = True
                break

            da = (2 / n) * r.dot(x_scaled)
            db = (2 / n) * r.sum()

            a -= config.lr * da
            b -= config.lr * db

            if check_tolerance(losses, tolerance):
                break

        if failed:
            raise NumericalInstabilityError(
                "Training failed due to numerical instability. Try reducing the learning rate."
            )

        if verbose:
            print(f"Converged in {len(losses)} epochs")

        # convert parameters back to original scale
        a_original = a / x_std
        b_original = b - (a * x_mean) / x_std

        # store learned parameters
        self.a = a_original
        self.b = b_original
        self._x_mean = x_mean
        self._x_std = x_std

        return FitResult(
            epochs_run=len(losses),
            final_loss=losses[-1],
            loss_history=losses,
        )

    def predict(self, x):
        """
        Predict using learned parameters.
        """
        if self.a is None or self.b is None:
            raise ValueError("Model has not been fitted yet.")

        x = np.asarray(x, dtype=float)
        return self.a * x + self.b
