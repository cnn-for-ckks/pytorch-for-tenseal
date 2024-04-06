from numpy.polynomial import Polynomial

import numpy as np

if __name__ == "__main__":
    # Generate sample data points
    x = np.linspace(-1, 1, 100)
    y = np.sin(x)

    # Define the degree of the polynomial approximation
    degree = 5

    # Perform least squares approximation
    coeffs = Polynomial.fit(x, y, degree).convert().coef

    # Construct the polynomial approximation
    approx_poly = Polynomial(coeffs)

    # Evaluate the approximation at desired points
    approx_values = approx_poly(x)

    # Compute the mean squared error
    mse = np.mean((approx_values - y) ** 2)

    print(f"Mean Squared Error: {mse}")
