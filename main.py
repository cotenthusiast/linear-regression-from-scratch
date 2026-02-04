from linear_regression import train_linear_regression, NumericalInstabilityError, train_poly2_regression 
from data_gen import generate_random_linear_dataset , generate_random_quadratic_dataset
from plotting import make_figure 
import matplotlib.pyplot as plt 
import numpy as np


lr = 0.01 # Initial learning rate 
min_lr = 1e-6 # Minimum learning rate to avoid infinite loop 
max_retries = 10 # Maximum number of retries to prevent infinite loops 
tries = 0 # Retry counter 
success = False 

# Generate a random dataset 
df = generate_random_quadratic_dataset(seed=42) 

# Train the linear regression model 
while lr >= min_lr and tries < max_retries: 
    try: 
        print(f"Attempt {tries+1}, lr={lr}") 
        w, losses, epochs_ran, x_mean, x_std = train_poly2_regression(df, lr=lr, epochs=1000, tolerance=1e-6, verbose=True)
        print("w:", w)
        success = True 
        break 
    except NumericalInstabilityError as e: 
        tries += 1 
        print(e) 
        print("Retrying with a smaller learning rate...") 
        lr *= 0.5 # Reduce learning rate and retry

if not success:
     raise Exception("Failed to train the model after multiple retries. Try starting with a smaller learning rate.") # Exit if training was unsuccessful 

x = np.array(df["x"], dtype=float)
x_grid = np.linspace(x.min(), x.max(), 400)
xs_grid = (x_grid - x_mean) / x_std

Phi_grid = np.zeros((len(x_grid), 3))
for i in range(len(x_grid)):
    Phi_grid[i, 0] = 1.0
    Phi_grid[i, 1] = xs_grid[i]
    Phi_grid[i, 2] = xs_grid[i] * xs_grid[i]

y_grid = Phi_grid @ w

# 4) pass to plotting
fig, (ax1, ax2) = make_figure(df, losses, x_grid, y_grid)  # adjust signature
fig.savefig("result.png", dpi=200)