# src/common/reporting.py

import json
from pathlib import Path

from src.common.types import FitResult


def build_fit_summary(result: FitResult) -> dict:
    """
    Build a small summary dictionary from a FitResult.

    Parameters
    ----------
    result : FitResult
        Output returned by model.fit(...).

    Returns
    -------
    dict
        Summary values.
    """
    best_loss = min(result.loss_history) if result.loss_history else None

    return {
        "epochs_run": result.epochs_run,
        "final_loss": result.final_loss,
        "best_loss": best_loss,
    }


def format_summary_text(summary: dict) -> str:
    """
    Convert a summary dictionary into readable text.

    Parameters
    ----------
    summary : dict
        Dictionary returned by build_fit_summary(...).

    Returns
    -------
    str
        Multi-line text.
    """
    lines = []
    lines.append(f"epochs_run: {summary['epochs_run']}")
    lines.append(f"final_loss: {summary['final_loss']}")
    lines.append(f"best_loss: {summary['best_loss']}")
    return "\n".join(lines)


def save_summary_json(summary: dict, path: str) -> None:
    """
    Save summary dictionary to a JSON file.

    Parameters
    ----------
    summary : dict
        Dictionary returned by build_fit_summary(...).
    path : str
        Output file path, e.g. 'artifacts/linear_summary.json'.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
