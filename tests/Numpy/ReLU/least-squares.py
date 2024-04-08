from numpy.polynomial import Polynomial

import numpy as np

if __name__ == "__main__":
    # Generate sample data points
    x = np.linspace(-1, 1, 10)
    y = np.maximum(0, x)
    y_differential = np.heaviside(x, 0)

    # Define the degree of the polynomial approximation
    degree = 5

    # Perform least squares approximation
    coeffs = Polynomial.fit(x, y, degree).convert(kind=Polynomial).coef

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
