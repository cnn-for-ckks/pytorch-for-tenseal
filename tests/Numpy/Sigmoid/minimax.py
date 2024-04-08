from numpy.polynomial import Polynomial, Chebyshev

import numpy as np

if __name__ == "__main__":
    # Generate sample data points
    x = np.linspace(-1, 1, 5)
    y = 1 / (1 + np.exp(-x))
    y_differential = np.exp(-x) / (1 + np.exp(-x)) ** 2

    # Define the degree of the polynomial approximation
    degree = 3

    # Perform least squares approximation
    coeffs = Chebyshev.fit(x, y, degree).convert(kind=Polynomial).coef

    # Construct the polynomial approximation
    approx_poly = Polynomial(coeffs)

    # Evaluate the approximation at desired points
    approx_values = approx_poly(x)

    # Compute the mean squared error
    mse = np.mean((approx_values - y) ** 2)

    print(f"Mean Squared Error: {mse}")

    # Differentiate the polynomial approximation
    approx_poly_derivative = approx_poly.deriv()

    # Evaluate the derivative at desired points
    approx_derivative_values = approx_poly_derivative(x)

    # Compute the mean squared error of the derivative
    mse_derivative = np.mean((approx_derivative_values - y_differential) ** 2)

    print(f"Mean Squared Error of the Derivative: {mse_derivative}")
