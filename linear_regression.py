import numpy as np
import pandas as pd

class NumericalInstabilityError(Exception): # Custom exception for numerical instability
    pass

def check_tolerance(losses, tolerance):
    '''
    Function that checks if the change in loss is within the specified tolerance
    
    :param losses: Array of loss values
    :param tolerance: Tolerance for stopping criterion
    :return: Boolean indicating if the change in loss is within the tolerance
    '''
    if len(losses) < 2:
        return False
    prev = losses[-2]
    current = losses[-1]
    if prev == 0:
        return abs(current - prev) < tolerance
    return abs(current - prev) / abs(prev) < tolerance

def train_linear_regression(df, lr = 0.01, epochs = 1000, tolerance = 1e-6, verbose = False):
    '''
    Function that trains a linear regression model using gradient descent.
    This function is trained based on a scaled version of x to improve stability.
    
    :param df: dataframe with x and y columns
    :param lr: learning rate for gradient descent
    :param epochs: number of epochs to train for
    :param tolerance: tolerance for stopping training based on loss improvement
    :return: tuple containing slope (a), intercept (b), list of losses (MSE over epochs), and number of epochs ran
    '''
    
    # equation for prediction: y_hat = a*x + b
    # residual vector: r = y_hat - y
    # MSE loss: (1/n) * r.dot(r)
    # updating parameters: a -= lr * (2/n) * r.dot(x)
    #                      b -= lr * (2/n) * r.sum()
    # Standardizing x: x = (x - x_mean) / x_std
    # Reverting a and b to original scale after training: original_a = a / x_std
    #                                         original_b = b - (a * x_mean) / x_std

    n = len(df) 
    x = np.array(df['x'], dtype=float)
    x_mean = x.mean()
    x_std = np.std(x)
    if x_std == 0:
        raise NumericalInstabilityError("Standard deviation of x is zero, cannot normalize.")
    x = (x - x_mean) / x_std  # Standardize x (zero mean, unit variance) for more stable gradient descent.
    y = np.array(df['y'], dtype=float)
    losses = []
    a = 0.0 # The slope being learned
    b = 0.0 # The intercept being learned
    failed = False
    for epoch in range (epochs): 
        y_hat = a * x + b 
        r = y_hat - y 
        losses.append(1/n * r.dot(r))
        if np.isnan(losses[-1]) or np.isinf(losses[-1]) or losses[-1] > 1e12:
            failed = True # numerical instability detected
            break
        da = (2/n) * r.dot(x) 
        db = (2/n) * r.sum() 
        a -= lr * da 
        b -= lr * db 
        if check_tolerance(losses, tolerance):
            break
    if failed: 
        raise Exception("Training failed due to numerical instability. Try reducing the learning rate.")
    if verbose:
        print(f"Converged in {len(losses)} epochs")
    epochs_ran = len(losses)
    a_original = a / x_std # converting a back to original scale
    b_original = b - (a * x_mean) / x_std
    return a_original, b_original, losses, epochs_ran


