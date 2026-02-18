# src/linear/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


def load_diabetes_1d(feature: str = "bmi") -> pd.DataFrame:
    """
    Load the sklearn diabetes dataset and return a 1D regression DataFrame
    with columns: 'x', 'y'.

    Parameters
    ----------
    feature : str
        Feature column name to use as x (e.g. 'bmi', 'bp', 's1'...).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['x', 'y'].
    """
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()  # includes features + 'target'
    df = df.rename(columns={"target": "y"})

    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(df.columns)}")

    out = df[[feature, "y"]].rename(columns={feature: "x"}).copy()
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    return out


def train_test_split(x, y, test_ratio=0.2, shuffle=True, random_state=None):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
