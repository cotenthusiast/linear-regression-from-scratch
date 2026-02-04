# Polynomial Regression From Scratch (Quadratic)

This repository implements **1D quadratic (degree-2) polynomial regression** from scratch using **gradient descent** with **MSE loss**.  
The point is to understand prediction → residuals → loss → gradients → parameter updates end-to-end.

## Project structure

- `main.py`  
  Runs the full pipeline: generate data → train → build fitted curve points → plot → save output.

- `linear_regression.py`  
  Core training loop for **quadratic regression** via gradient descent.

- `plotting.py`  
  Generates a 2-panel figure: **data + fitted curve** and **loss vs epoch**.

- `data_gen.py`  
  Synthetic data generation used for testing (can be linear or quadratic depending on the function you call).

## What the model learns

A quadratic curve (in 1D):

\[
\hat{y} = w_0 + w_1 z + w_2 z^2
\]

Where the input is standardized for stability:

\[
z = \frac{x - \mu}{\sigma}
\]

- \(w_0\): bias (intercept in standardized space)  
- \(w_1\): linear term coefficient  
- \(w_2\): quadratic term coefficient  
- \(\mu\): mean of training \(x\)  
- \(\sigma\): std of training \(x\)

**Important:** The training returns \(w\) in standardized-input space, so any prediction must scale inputs the same way.

## Training objective (MSE)

Residuals:

\[
r_i = \hat{y}_i - y_i
\]

Mean Squared Error:

\[
L = \frac{1}{n}\sum_{i=1}^{n} r_i^2
\]

Vector form:

\[
L = \frac{1}{n}(r \cdot r)
\]

## Gradients and updates

With \(\hat{y}_i = w_0 + w_1 z_i + w_2 z_i^2\):

\[
\frac{\partial L}{\partial w_0} = \frac{2}{n}\sum_{i=1}^{n} r_i
\]
\[
\frac{\partial L}{\partial w_1} = \frac{2}{n}\sum_{i=1}^{n} r_i z_i
\]
\[
\frac{\partial L}{\partial w_2} = \frac{2}{n}\sum_{i=1}^{n} r_i z_i^2
\]

Gradient descent:

\[
w_k \leftarrow w_k - lr \cdot \frac{\partial L}{\partial w_k}
\]

## Plotting the fitted curve

To draw a smooth curve:

1) Create dense x-values:
- `x_grid = linspace(x_min, x_max, 400)`

2) Scale them using training stats:
- `z_grid = (x_grid - mu) / sigma`

3) Predict:
- `y_grid = w0 + w1*z_grid + w2*(z_grid**2)`

4) Plot:
- scatter `(x, y)`
- line `(x_grid, y_grid)`

## How to run

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2) Run
```bash
python3 main.py
```

### Output:

- a saved plot image (e.g. result.png) showing:

  - data + fitted quadratic curve

  - loss over epochs

## Expected behavior

- Loss generally decreases then plateaus.

- With a quadratic dataset, the fitted curve visibly bends.

- With a linear dataset, the quadratic term typically goes near 0 and the fit looks like a line.

## Notes / learning scope

- 1 input feature (x)

- polynomial expansion to degree 2

- MSE loss

- gradient descent optimization

- explicit feature scaling for training stability

## Attribution

The matplotlib plotting boilerplate (figure layout + saving) had minor ChatGPT assistance. Everything else was implemented and verified by me (including the training loop and gradient derivations).