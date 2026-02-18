# src/polynomial/evaluate.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.common.metrics import mse, mae
from src.common.types import FitResult


def plot_loss_curve(loss_history, path: str) -> None:
    loss_history = np.asarray(loss_history, dtype=float)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def print_metrics(y_true, y_pred, label: str) -> None:
    print(f"{label}_mse: {mse(y_true, y_pred)}")
    print(f"{label}_mae: {mae(y_true, y_pred)}")


def plot_fit_curve_1d(x, y, model_predict_fn, path: str, num_points: int = 300) -> None:
    """
    Plot scatter of (x, y) plus a smooth fitted curve produced by model_predict_fn.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    x_min = float(x.min())
    x_max = float(x.max())
    x_grid = np.linspace(x_min, x_max, num_points)
    y_grid = model_predict_fn(x_grid)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x_grid, y_grid)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data + Fitted Polynomial Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_plots(
    fit_result: FitResult,
    x_train,
    y_train_true,
    y_train_pred,
    model_predict_fn,
    x_test=None,
    y_test_true=None,
    y_test_pred=None,
    plots_dir: str = "plots",
    name: str = "poly",
) -> None:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Print basic training stats
    print(f"epochs_run: {fit_result.epochs_run}")
    print(f"final_loss: {fit_result.final_loss}")
    if fit_result.loss_history:
        print(f"best_loss: {min(fit_result.loss_history)}")

    # Print metrics
    print_metrics(y_train_true, y_train_pred, "train")
    if y_test_true is not None and y_test_pred is not None:
        print_metrics(y_test_true, y_test_pred, "test")

    # Loss curve
    plot_loss_curve(fit_result.loss_history, str(plots_dir / f"{name}_loss.png"))

    # Fit curve (train data)
    plot_fit_curve_1d(
        x_train,
        y_train_true,
        model_predict_fn=model_predict_fn,
        path=str(plots_dir / f"{name}_fit_train.png"),
    )

    # Optional: fit curve on test scatter
    if x_test is not None and y_test_true is not None:
        plot_fit_curve_1d(
            x_test,
            y_test_true,
            model_predict_fn=model_predict_fn,
            path=str(plots_dir / f"{name}_fit_test.png"),
        )
