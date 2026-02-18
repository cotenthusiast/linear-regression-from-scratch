from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FitResult:
    """
    Summary of a training run.

    Attributes
    ----------
    epochs_run : int
        Number of epochs actually completed.
    final_loss : float
        Loss value at the final epoch.
    loss_history : List[float]
        Loss per epoch (length == epochs_run).
    """
    epochs_run: int
    final_loss: float
    loss_history: List[float]

@dataclass(frozen=True)
class TrainConfig:
    """
    Training hyperparameters.

    Attributes
    ----------
    lr : float
        Learning rate.
    epochs : int
        Maximum number of epochs.
    """
    lr: float
    epochs: int