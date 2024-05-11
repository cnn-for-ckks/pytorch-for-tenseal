from typing import Literal, Union
from numpy.polynomial import Polynomial, Chebyshev
from torchseal.wrapper import CKKSWrapper
from torchseal.function import SoftmaxFunction

import typing
import numpy as np
import torch


class Softmax(torch.nn.Module):
    def __init__(
            self,
            exp_start: float,
            exp_stop: float,
            exp_num_of_sample: int,
            exp_degree: int,
            exp_approximation_type: Union[
                Literal["minimax"],
                Literal["least-squares"]
            ],
            inverse_start: float,
            inverse_stop: float,
            inverse_num_of_sample: int,
            inverse_degree: int,
            inverse_approximation_type: Union[
                Literal["minimax"],
                Literal["least-squares"]
            ],
            inverse_iterations: int
    ) -> None:
        super(Softmax, self).__init__()

        # Create the polynomial for the exp function
        exp_x = np.linspace(exp_start, exp_stop, exp_num_of_sample)
        exp_y = (lambda x: np.exp(x))(exp_x)

        # Perform the polynomial approximation for the exp function
        self.exp_coeffs = Polynomial.fit(
            exp_x, exp_y, exp_degree
        ).convert(kind=Polynomial).coef if exp_approximation_type == "least-squares" else Chebyshev.fit(exp_x, exp_y, exp_degree).convert(kind=Polynomial).coef

        # Create the polynomial for the inverse function
        inverse_x = np.linspace(
            inverse_start, inverse_stop, inverse_num_of_sample
        )
        inverse_y = (lambda x: 1 / x)(inverse_x)

        # Perform the polynomial approximation for the inverse function
        self.inverse_coeffs = Polynomial.fit(
            inverse_x, inverse_y, inverse_degree
        ).convert(kind=Polynomial).coef if inverse_approximation_type == "least-squares" else Chebyshev.fit(inverse_x, inverse_y, inverse_degree).convert(kind=Polynomial).coef

        # Save the iterations
        self.iterations = inverse_iterations

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        # TODO: Implement the forward pass based on self.training flag

        enc_output = typing.cast(
            CKKSWrapper,
            SoftmaxFunction.apply(
                enc_x, self.exp_coeffs, self.inverse_coeffs, self.iterations
            )
        )

        return enc_output

    def train(self, mode=True) -> "Softmax":
        # TODO: Change the plaintext parameters to encrypted parameters if mode is True
        # TODO: Else, change the encrypted parameters to plaintext parameters

        return super(Softmax, self).train(mode)

    def eval(self) -> "Softmax":
        # TODO: Change the encrypted parameters to plaintext parameters

        return super(Softmax, self).eval()
