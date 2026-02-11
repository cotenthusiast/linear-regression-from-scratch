import numpy as np
from data_gen import generate_random_quadratic_dataset

# ---- Helpers (pure math, no training) ----

def predict_poly2(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    # w = [w0, w1, w2]
    return w[0] + w[1] * z + w[2] * (z ** 2)

def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    r = y_hat - y
    return float(np.mean(r ** 2))

def analytic_grads(z: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = y_hat - y
    n = z.size
    dw0 = (2.0 / n) * np.sum(r)
    dw1 = (2.0 / n) * np.sum(r * z)
    dw2 = (2.0 / n) * np.sum(r * (z ** 2))
    return np.array([dw0, dw1, dw2], dtype=float)

def numeric_grads(z: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    grads = np.zeros_like(w, dtype=float)
    for k in range(w.size):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[k] += eps
        w_minus[k] -= eps

        loss_plus = mse_loss(predict_poly2(z, w_plus), y)
        loss_minus = mse_loss(predict_poly2(z, w_minus), y)
        grads[k] = (loss_plus - loss_minus) / (2.0 * eps)
    return grads

def rel_error(a: float, b: float) -> float:
    denom = max(1e-12, abs(a) + abs(b))
    return abs(a - b) / denom


# ---- Main ----

def main():
    # 1) Small dataset
    df = generate_random_quadratic_dataset(seed=42, n=20)

    # 2) Extract x, y (adjust names if your df uses different columns)
    # Common options: df["x"], df["y"] or df["X"], df["Y"]
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # 3) Standardize x -> z
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        raise ValueError("sigma is zero; cannot standardize.")
    z = (x - mu) / sigma

    # 4) Fixed weights (DO NOT TRAIN)
    w = np.array([0.1, -0.2, 0.05], dtype=float)

    # 5) Analytic gradients
    y_hat = predict_poly2(z, w)
    grads_a = analytic_grads(z, y_hat, y)

    # 6) Numerical gradients
    eps = 1e-5
    grads_n = numeric_grads(z, y, w, eps=eps)

    # 7) Print comparison
    names = ["w0", "w1", "w2"]
    print(f"eps = {eps}")
    for i, name in enumerate(names):
        a = grads_a[i]
        n = grads_n[i]
        print(f"{name}: analytic={a:.8f}  numeric={n:.8f}  abs_diff={abs(a-n):.8e}  rel_err={rel_error(a,n):.8e}")

    # Optional: simple pass/fail threshold
    max_rel = max(rel_error(grads_a[i], grads_n[i]) for i in range(3))
    print(f"max_rel_err = {max_rel:.8e}")
    if max_rel < 1e-4:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()