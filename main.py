from poly_regression import * 
from data_gen import  *
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

x = np.array(df["x"], dtype = float)
y = np.array(df["y"], dtype = float)

x_grid = np.linspace(x.min(), x.max(), 40)
z_grid = (x_grid - x_mean) / x_std
y_grid = w[0] + w[1] * z_grid + w[2] * z_grid ** 2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.scatter(x, y, c = "red")
ax1.plot(x_grid, y_grid)
ax1.set_title("Data + fitted curve")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Bottom: loss over epochs
ax2.plot(range(1, len(losses) + 1), losses)
ax2.set_title("Loss over epochs")
ax2.set_xlabel("epoch")
ax2.set_ylabel("MSE")

fig.tight_layout()
fig.savefig("result.png", dpi=200)

