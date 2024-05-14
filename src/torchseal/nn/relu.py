from typing import Literal, Union
from numpy.polynomial import Polynomial, Chebyshev
from torchseal.function import ReLUFunction
from torchseal.wrapper import CKKSWrapper

import typing
import numpy as np
import torch


class ReLU(torch.nn.Module):
    coeffs: np.ndarray
    deriv_coeffs: np.ndarray

    def __init__(
            self,
            start: float,
            stop: float,
            num_of_sample: int,
            degree: int,
            approximation_type: Union[
                Literal["minimax"], Literal["least-squares"]
            ],
            deriv_start: float,
            deriv_stop: float,
            deriv_num_of_sample: int,
            deriv_degree: int,
            deriv_approximation_type: Union[
                Literal["minimax"], Literal["least-squares"]
            ]
    ) -> None:
        super(ReLU, self).__init__()

        # Create the polynomial
        x = np.linspace(start, stop, num_of_sample)
        y = (lambda x: np.maximum(0, x))(x)

        # Perform the polynomial approximation
        self.coeffs = Polynomial.fit(
            x, y, degree
        ).convert(kind=Polynomial).coef if approximation_type == "least-squares" else Chebyshev.fit(x, y, degree).convert(kind=Polynomial).coef

        # Create the polynomial for the derivative
        deriv_x = np.linspace(deriv_start, deriv_stop, deriv_num_of_sample)
        deriv_y = (lambda x: (x > 0) * 1)(deriv_x)

        # Perform the polynomial approximation for the derivative
        self.deriv_coeffs = Polynomial.fit(
            deriv_x, deriv_y, deriv_degree
        ).convert(kind=Polynomial).coef if deriv_approximation_type == "least-squares" else Chebyshev.fit(deriv_x, deriv_y, deriv_degree).convert(kind=Polynomial).coef

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            ReLUFunction.apply(
                enc_x, self.coeffs, self.deriv_coeffs
            )
        )

        return enc_output
