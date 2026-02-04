import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(ax, losses):
    '''
    Function that plots the losses over epochs.
    
    :param ax: matplotlib axes object to plot on
    :param losses: losses array to plot
    :return: None
    '''
    ax.plot(range(len(losses)), losses) # Plot losses over epochs
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

def plot_fit(ax, df, a, b):
    '''
    Function that plots the data points and the fitted line.
    
    :param ax: matplotlib axes object to plot on
    :param df: dataframe with x and y columns
    :param a: slope of the fitted line
    :param b: intercept of the fitted line
    :return: None
    '''
    ordered_x = np.sort(df['x']) # Sort x values for line plotting
    ax.scatter(df['x'], df['y'], color='red', label='Data Points')
    ax.plot(ordered_x, a * ordered_x + b, color='blue', label='Fitted Line')
    ax.set_title("Data + Fitted Line")  
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

def make_figure(df, losses, x_fit=None, y_fit=None):
    '''
    Method that sets up the figure 

    :param df: dataframe with x and y columns
    :param losses: np array of losses during training
    :param x_fit


    '''
    x = np.array(df["x"], dtype=float)
    y = np.array(df["y"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: data + fitted curve/line
    ax1.scatter(x, y)
    if x_fit is not None and y_fit is not None:
        ax1.plot(x_fit, y_fit)
    ax1.set_title("Data + Fit")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Right: loss curve
    ax2.plot(range(1, len(losses) + 1), losses)
    ax2.set_title("MSE over epochs")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE")

    return fig, (ax1, ax2)
