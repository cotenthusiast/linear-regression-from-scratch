# scripts/run_linear.py

import numpy as np

from src.common.types import TrainConfig
from src.linear.preprocessing import load_diabetes_1d, train_test_split
from src.linear.model import LinearRegressionGD
from src.linear.evaluate import save_plots


def main():
    # 1) Load dataset (1D feature -> x, target -> y)
    df = load_diabetes_1d(feature="bmi")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # 2) Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_ratio=0.2, shuffle=True, random_state=42
    )

    # 3) Train
    model = LinearRegressionGD()
    config = TrainConfig(lr=0.01, epochs=2000)
    fit_result = model.fit(x_train, y_train, config, tolerance=1e-6, verbose=True)

    # 4) Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 5) Evaluate + save plots
    save_plots(
        fit_result=fit_result,
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        plots_dir="plots",
        name="linear_diabetes_bmi",
    )


if __name__ == "__main__":
    main()
