# src/linear/evaluate.py

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


def plot_pred_vs_true(y_true, y_pred, path: str) -> None:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])  # identity line: perfect predictions

    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def print_metrics(y_true, y_pred, label: str) -> None:
    mse_value = mse(y_true, y_pred)
    mae_value = mae(y_true, y_pred)

    print(f"{label}_mse: {mse_value}")
    print(f"{label}_mae: {mae_value}")

def save_plots(
    fit_result: FitResult,
    y_train_true,
    y_train_pred,
    y_test_true=None,
    y_test_pred=None,
    plots_dir: str = "plots",
    name: str = "linear",
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

    # Save plots
    plot_loss_curve(fit_result.loss_history, str(plots_dir / f"{name}_loss.png"))
    plot_pred_vs_true(
        y_train_true, y_train_pred, str(plots_dir / f"{name}_pred_vs_true_train.png")
    )
    if y_test_true is not None and y_test_pred is not None:
        plot_pred_vs_true(
            y_test_true, y_test_pred, str(plots_dir / f"{name}_pred_vs_true_test.png")
        )