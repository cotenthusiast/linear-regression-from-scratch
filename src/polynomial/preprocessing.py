# src/polynomial/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_california_housing_1d(feature: str = "MedInc") -> pd.DataFrame:
    """
    Load sklearn California Housing dataset and return a 1D regression DataFrame
    with columns: 'x', 'y'.

    Notes
    -----
    - This dataset may download on first use.
    - Pick one numeric feature as x (default: MedInc).
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()  # includes features + target column (MedHouseVal)

    target_col = "MedHouseVal"
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(df.columns)}")

    out = df[[feature, target_col]].rename(columns={feature: "x", target_col: "y"}).copy()
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    return out


def train_test_split(x, y, test_ratio=0.2, shuffle=True, random_state=None):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    split = int(n * (1 - test_ratio))
    train_idx = idx[:split]
    test_idx = idx[split:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
