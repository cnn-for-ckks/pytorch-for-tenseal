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

        self.deriv_coeffs = Polynomial(self.coeffs).deriv().coef

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            ReLUFunction.apply(
                enc_x, self.coeffs, self.deriv_coeffs
            )
        )

        return enc_output
