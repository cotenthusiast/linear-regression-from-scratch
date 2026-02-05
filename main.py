from linear_regression import * 
from data_gen import * 
from plotting import *
import matplotlib.pyplot as plt 

lr = 0.01 # Initial learning rate 
min_lr = 1e-6 # Minimum learning rate to avoid infinite loop 
max_retries = 10 # Maximum number of retries to prevent infinite loops 
tries = 0 # Retry counter 
success = False 

# Generate a random dataset 
df = generate_random_linear_dataset(seed=42) 

# Train the linear regression model 
while lr >= min_lr and tries < max_retries: 
    try: 
        print(f"Attempt {tries+1}, lr={lr}") 
        a, b, losses, epochs_ran = train_linear_regression(df, lr, epochs=1000, tolerance=1e-6, verbose=True) 
        success = True 
        break 
    except NumericalInstabilityError as e: 
        tries += 1 
        print(e) 
        print("Retrying with a smaller learning rate...") 
        lr *= 0.5 # Reduce learning rate and retry

if not success:
     raise Exception("Failed to train the model after multiple retries. Try starting with a smaller learning rate.") # Exit if training was unsuccessful 

# Create the figure with plots 
fig, (ax1, ax2) = make_figure(df, a, b, losses) 

# Show the plots 
fig.savefig("result.png", dpi=200)