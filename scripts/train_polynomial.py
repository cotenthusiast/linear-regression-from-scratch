# scripts/run_polynomial.py

from src.common.types import TrainConfig
from src.polynomial.preprocessing import load_california_housing_1d, train_test_split
from src.polynomial.model import PolynomialRegressionGD
from src.polynomial.evaluate import save_plots


def main():
    # 1) Load dataset (1D feature -> x, target -> y)
    df = load_california_housing_1d(feature="MedInc")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # 2) Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_ratio=0.2, shuffle=True, random_state=42
    )

    # 3) Train
    degree = 3
    model = PolynomialRegressionGD(degree=degree)

    # Polynomial can get unstable: start conservative
    config = TrainConfig(lr=0.001, epochs=5000)
    fit_result = model.fit(x_train, y_train, config, tolerance=1e-6, verbose=True)

    # 4) Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 5) Evaluate + save plots
    save_plots(
        fit_result=fit_result,
        x_train=x_train,
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        model_predict_fn=model.predict,
        x_test=x_test,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        plots_dir="plots",
        name=f"poly_california_MedInc_deg{degree}",
    )


if __name__ == "__main__":
    main()
