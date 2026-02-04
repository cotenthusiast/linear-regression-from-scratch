import numpy as np
import pandas as pd

# Note: These functions were created by ChatGPT and not me. The main point of this project is for me to implement linear regression from scratch.

import numpy as np
import pandas as pd

def generate_random_quadratic_dataset(seed=42, n=20, a=10.0, b=-1.2, c=0.03, noise_std=5.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-50, 50, size=n)
    noise = rng.normal(0.0, noise_std, size=n)
    y = a + b*x + c*(x**2) + noise
    return pd.DataFrame({"x": x, "y": y})

def generate_random_linear_dataset(
    n_min: int = 12,
    n_max: int = 80,
    x_range: tuple[float, float] = (-50.0, 50.0),
    slope_range: tuple[float, float] = (-5.0, 5.0),
    intercept_range: tuple[float, float] = (-30.0, 30.0),
    noise_std_range: tuple[float, float] = (0.5, 10.0),
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generates a random 2-column dataset (x, y) following:
        y = m*x + c + eps
    where m, c, noise std, n, and x values are randomized.

    Returns a DataFrame with columns: ['x', 'y'].
    """
    rng = np.random.default_rng(seed)

    # Randomize dataset size and underlying linear relationship
    n = int(rng.integers(n_min, n_max + 1))
    m = float(rng.uniform(*slope_range))
    c = float(rng.uniform(*intercept_range))
    noise_std = float(rng.uniform(*noise_std_range))

    # Randomize x values (not evenly spaced), then sort for nicer plotting
    x = rng.uniform(x_range[0], x_range[1], size=n).astype(float)
    x.sort()

    # Generate noisy linear targets
    eps = rng.normal(0.0, noise_std, size=n)
    y = m * x + c + eps

    return pd.DataFrame({"x": x, "y": y})
    