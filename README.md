* * * * *

Polynomial Regression From Scratch (Quadratic)
==============================================

This repository implements **1D quadratic (degree-2) polynomial regression** from scratch using **gradient descent** with **MSE loss**.

The goal is to understand the full pipeline:\
prediction → residuals → loss → gradients → parameter updates.

* * * * *

Project structure
-----------------

-   `main.py`\
    Runs the full pipeline: generate data → train → build fitted curve points → plot → save output.

-   `linear_regression.py`\
    Core training loop for **quadratic regression** via gradient descent.

-   `plotting.py`\
    Generates a 2-panel figure:

    -   data + fitted curve

    -   loss vs epoch

-   `data_gen.py`\
    Synthetic data generation used for testing\
    (linear or quadratic depending on the function called).

* * * * *

What the model learns
---------------------

A quadratic curve in 1D:

`y_hat = w0 + w1*z + w2*z^2`

The input is standardized for numerical stability:

`z = (x - mu) / sigma`

Parameter meanings:

-   `w0` : bias (intercept in standardized space)

-   `w1` : linear coefficient

-   `w2` : quadratic coefficient

-   `mu` : mean of training x

-   `sigma` : standard deviation of training x

**Important:**\
Training returns weights in standardized-input space.\
Any prediction must apply the same scaling.

* * * * *

Training objective (MSE)
------------------------

Residual for each data point:

`r_i = y_hat_i - y_i`

Mean Squared Error:

`L = (1 / n) * sum(r_i^2)`

Vector form:

`L = (1 / n) * (r dot r)`

* * * * *

Gradients and updates
---------------------

Prediction:

`y_hat_i = w0 + w1*z_i + w2*z_i^2`

Gradients:

`dL/dw0 = (2 / n) * sum(r_i)
dL/dw1 = (2 / n) * sum(r_i * z_i)
dL/dw2 = (2 / n) * sum(r_i * z_i^2)`

Gradient descent update rule:

`w_k = w_k - lr * (dL/dw_k)`

* * * * *

Plotting the fitted curve
-------------------------

To draw a smooth fitted curve:

1.  Create dense x values:

`x_grid = linspace(x_min, x_max, 400)`

1.  Scale using training statistics:

`z_grid = (x_grid - mu) / sigma`

1.  Predict:

`y_grid = w0 + w1*z_grid + w2*(z_grid**2)`

1.  Plot:

`scatter(x, y)
line(x_grid, y_grid)`

* * * * *

How to run
----------

### 1) Install dependencies

`python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt`

### 2) Run

`python3 main.py`

* * * * *

Output
------

A saved plot image (e.g. `result.png`) showing:

-   data + fitted quadratic curve

-   loss over epochs

* * * * *

Expected behavior
-----------------

-   Loss decreases, then plateaus.

-   Quadratic dataset → visibly curved fit.

-   Linear dataset → quadratic term approaches zero and the fit looks linear.

* * * * *

Notes / learning scope
----------------------

-   single input feature (x)

-   polynomial expansion to degree 2

-   MSE loss

-   gradient descent optimization

-   explicit feature scaling for training stability