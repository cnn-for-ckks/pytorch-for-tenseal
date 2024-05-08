from typing import Callable, Literal, Union
from numpy.polynomial import Polynomial, Chebyshev
from torchseal.function import TanhFunction
from torchseal.wrapper import CKKSWrapper

import typing
import numpy as np
import torch


class Tanh(torch.nn.Module):
    def __init__(self, start: float, stop: float, num_of_sample: int, degree: int, approximation_type: Union[Literal["minimax"], Literal["least-squares"]]) -> None:
        super(Tanh, self).__init__()

        # Create the polynomial
        x = np.linspace(start, stop, num_of_sample)
        y = (lambda x: np.tanh(x))(x)

        # Perform the polynomial approximation
        self.coeffs: np.ndarray = Polynomial.fit(
            x, y, degree
        ).convert(kind=Polynomial).coef if approximation_type == "least-squares" else Chebyshev.fit(x, y, degree).convert(kind=Polynomial).coef

        # Construct the polynomial approximation
        approx_poly = Polynomial(self.coeffs)

        # Differentiate the polynomial approximation
        self.approx_poly_derivative: Callable[
            [float], float
        ] = approx_poly.deriv()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            TanhFunction.apply(
                enc_x, self.coeffs, self.approx_poly_derivative
            )
        )

        return enc_output
