from typing import Any
import numpy as np

def _to_1d(a: Any) -> np.ndarray:
    """
    Convert input to a 1D NumPy array of shape (n,).

    Parameters
    ----------
    a : Any
        Array-like input (e.g., list, NumPy array).

    Returns
    -------
    np.ndarray
        1D NumPy array of shape (n,).
    """
    return np.asarray(a).reshape(-1)


def mse(y_true: Any, y_pred: Any) -> float:
    """
    Compute mean squared error between true targets and predictions.

    Parameters
    ----------
    y_true : Any
        True target values (array-like).
    y_pred : Any
        Predicted target values (array-like).

    Returns
    -------
    float
        Mean squared error.
    """
    y_true_1d = _to_1d(y_true)
    y_pred_1d = _to_1d(y_pred)

    if y_true_1d.shape != y_pred_1d.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true_1d.shape}, y_pred={y_pred_1d.shape}"
        )
    if y_true_1d.size == 0:
        raise ValueError("Empty input: y_true/y_pred must contain at least one value.")

    r = y_pred_1d - y_true_1d
    return float(np.mean(r * r))


def mae(y_true: Any, y_pred: Any) -> float:
    """
    Compute mean absolute error between true targets and predictions.

    Parameters
    ----------
    y_true : Any
        True target values (array-like).
    y_pred : Any
        Predicted target values (array-like).

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true_1d = _to_1d(y_true)
    y_pred_1d = _to_1d(y_pred)

    if y_true_1d.shape != y_pred_1d.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true_1d.shape}, y_pred={y_pred_1d.shape}"
        )
    if y_true_1d.size == 0:
        raise ValueError("Empty input: y_true/y_pred must contain at least one value.")

    return float(np.mean(np.abs(y_pred_1d - y_true_1d)))