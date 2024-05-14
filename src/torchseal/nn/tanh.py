from typing import Literal, Union
from numpy.polynomial import Polynomial, Chebyshev
from torchseal.wrapper import CKKSWrapper
from torchseal.function import TanhFunction

import typing
import numpy as np
import torch


class Tanh(torch.nn.Module):
    coeffs: np.ndarray

    def __init__(self, start: float, stop: float, num_of_sample: int, degree: int, approximation_type: Union[Literal["minimax"], Literal["least-squares"]]) -> None:
        super(Tanh, self).__init__()

        # Create the polynomial
        x = np.linspace(start, stop, num_of_sample)
        y = (lambda x: np.tanh(x))(x)

        # Perform the polynomial approximation
        self.coeffs = Polynomial.fit(
            x, y, degree
        ).convert(kind=Polynomial).coef if approximation_type == "least-squares" else Chebyshev.fit(x, y, degree).convert(kind=Polynomial).coef

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            TanhFunction.apply(
                enc_x, self.coeffs
            )
        )

        return enc_output
