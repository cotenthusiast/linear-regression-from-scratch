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

def make_figure(df, a, b, losses):
    '''
    Function that creates a figure with two subplots: one for the data + fitted line,
    and one for the losses over epochs.
    
    :param df: dataframe with x and y columns
    :param a: slope of the fitted line
    :param b: intercept of the fitted line
    :param losses: losses array to plot
    :return: tuple containing the figure and axes objects
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_fit(ax1, df, a, b)
    plot_losses(ax2, losses)
    fig.tight_layout()
    return fig, (ax1, ax2)