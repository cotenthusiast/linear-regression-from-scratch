Linear and Polynomial Regression From Scratch (Gradient Descent)
==========================================

This repository contains from-scratch implementations of linear regression and polynomial regression trained with gradient descent.

The purpose is to build a clear, first-principles understanding of:

-   Prediction functions

-   Residuals and loss functions

-   Analytical gradients

-   Iterative optimization with gradient descent

-   Numerical stability (feature scaling)

-   Clean project structure and modular design

This is not intended to compete with libraries like scikit-learn. The goal is mechanical understanding and engineering clarity.


Implemented models
------------------

### 1) Linear Regression (1D)

Model:\
y_hat = a*x + b

Where:

-   a = slope

-   b = intercept

### 2) Polynomial Regression (1D)

Model:\
y_hat = theta0 + theta1*x + theta2*x^2 + ... + thetad*x^d

Polynomial regression is implemented as linear regression on transformed features.


Training objective (MSE)
------------------------

Residual:\
r_i = y_hat_i - y_i

Mean Squared Error:\
L = (1/n) * sum(r_i^2)

Vector form used in code:\
L = (1/n) * (r dot r)


Gradient descent updates
------------------------

For linear regression:\
dL/da = (2/n) * sum(r_i * x_i)\
dL/db = (2/n) * sum(r_i)

Update rule:\
parameter = parameter - lr * gradient

Polynomial regression uses the matrix equivalent:\
gradient = (2/n) * X^T * r


Feature scaling (stability)
---------------------------

To improve gradient descent stability, the input feature is standardized:\
x_scaled = (x - mean(x)) / std(x)

The model is trained internally on scaled features. Predictions are still returned in the original target units.


Project structure
-----------------

src/\
common/\
metrics.py\
reporting.py\
types.py

linear/\
model.py\
preprocessing.py\
evaluate.py

polynomial/\
model.py\
preprocessing.py\
evaluate.py

scripts/\
run_linear.py\
run_polynomial.py

plots/

Design principles:

-   src/ contains reusable library code

-   scripts/ contains runnable pipelines

-   Models use OOP (fit, predict)

-   Evaluation and plotting are separated from training logic


Datasets
--------

Two real datasets are used:

-   Linear regression: sklearn Diabetes dataset (single feature)

-   Polynomial regression: sklearn California Housing dataset (single feature)

Using real data helps verify behavior beyond synthetic examples.


How to run
----------

1.  Install dependencies\
    pip install -r requirements.txt

2.  Run linear regression\
    python scripts/run_linear.py

3.  Run polynomial regression\
    python scripts/run_polynomial.py

Outputs:

-   Loss curves

-   Predicted vs true plots

-   Fitted polynomial curves

Plots are saved in the plots/ directory.



Refactoring note (AI usage)
---------------------------

The original implementation of gradient descent and the core training logic was written manually.

Later, significant refactoring was performed to:

-   Improve structure and modularity

-   Introduce OOP design

-   Separate preprocessing, training, and evaluation

-   Clean up duplicated logic

-   Standardize interfaces across models

AI tools were used heavily during the refactoring phase. This usage was mainly for:

-   Code organization suggestions

-   Boilerplate restructuring

-   Interface design ideas

-   Cleanup and consistency improvements

The math and training mechanics were understood and implemented before the refactor.
