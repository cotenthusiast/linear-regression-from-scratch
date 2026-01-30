# Linear Regression From Scratch (Gradient Descent)

This repository contains a from-scratch implementation of **1D linear regression** trained with **gradient descent**.  
The goal is to understand the mechanics behind prediction, residuals, loss functions, gradients, and iterative optimization (rather than treating ML as a black box).

## Project structure

- `main.py`  
  Runs the full pipeline: generate data → train → plot.

- `linear_regression.py`  
  Core implementation: gradient descent training loop (MSE loss), feature scaling for stability, and conversion back to original units.

- `plotting.py`  
  Plot utilities: loss curve and fitted line.

- `data_gen.py`  
  Synthetic data generator used only to test the training loop on many randomized linear datasets.

## Data generation note

`data_gen.py` is a small synthetic-data helper used only to stress-test the training loop across different randomly generated linear datasets. It was produced with ChatGPT assistance and is not the focus of this project.

The main learning objective here is the implementation and verification of the gradient-descent training loop in `linear_regression.py`.

## What the model learns

We fit a straight line:

$$
\hat{y} = a x + b
$$

- $a$ is the slope
- $b$ is the intercept

## Training objective (MSE)

Define residuals:

$$
r_i = \hat{y}_i - y_i
$$

Mean Squared Error (MSE):

$$
L = \frac{1}{n}\sum_{i=1}^{n} r_i^2
$$

Vector form (what the code computes):

$$
L = \frac{1}{n}(r \cdot r)
$$

## Gradients and parameter updates

Gradients for MSE:

$$
\frac{\partial L}{\partial a} = \frac{2}{n}\sum_{i=1}^{n} r_i x_i
$$

$$
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n} r_i
$$

Gradient descent update rule:

$$
a \leftarrow a - lr \cdot \frac{\partial L}{\partial a}
$$

$$
b \leftarrow b - lr \cdot \frac{\partial L}{\partial b}
$$

## Feature scaling (stability)

To make gradient descent more stable across different random datasets, the training loop standardizes $x$:

$$
x' = \frac{x - \mu}{\sigma}
$$

where:
- $\mu$ is the mean of $x$
- $\sigma$ is the standard deviation of $x$

The model is trained internally on $x'$, then converted back to original $x$ units so the final model is still:

$$
\hat{y} = a_{\text{orig}} x + b_{\text{orig}}
$$

Conversion back:

$$
a_{\text{orig}} = \frac{a'}{\sigma}
$$

$$
b_{\text{orig}} = b' - a'\frac{\mu}{\sigma}
$$

## How to run

### 1) Install dependencies

Create and activate a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
```

### 2) Run
```bash
python main.py
```
You should see:
- A loss vs epoch plot (loss decreases then plateaus)
- a scatter plot of data + the fitted regression line 

## Expected behavior
- The loss should generally decrease over epochs.
- The fitted line should match the overall trend of the data.
- Because the data generator adds noise, the line will not pass through every point.

## Notes / learning status
- This project is intentionally small and focused
- 1 feature (x)
- 1 target (y)
- MSE loss
- gradient descent optimization

It’s designed as a baseline before moving to multi-feature regression (matrix form), classification, and other loss functions.
