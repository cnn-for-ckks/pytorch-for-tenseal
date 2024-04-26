from typing import Callable, Optional
from tenseal import CKKSTensor

import torch
import tenseal as ts
import numpy as np


class CKKSWrapper(torch.Tensor):
    __ckks_data: CKKSTensor

    @property
    def ckks_data(self) -> CKKSTensor:
        return self.__ckks_data

    @ckks_data.setter
    def ckks_data(self, ckks_data: CKKSTensor) -> None:
        self.__ckks_data = ckks_data

    # # Logging method
    # @classmethod
    # def __torch_function__(cls, func, types, *args, **kwargs):
    #     # Print the function and the types
    #     print(f"Function: {func}")

    #     return super(CKKSWrapper, cls).__torch_function__(func, types, *args, **kwargs)

    # Overridden methods
    def __new__(cls, _data: torch.Tensor, _ckks_data: CKKSTensor, *args, **kwargs) -> "CKKSWrapper":
        # Create the tensor
        data = super(CKKSWrapper, cls).__new__(cls, *args, **kwargs)

        # Create the instance
        instance = torch.Tensor._make_subclass(cls, data)

        return instance

    # Overridden methods
    def __init__(self, data: torch.Tensor, ckks_data: CKKSTensor, *args, **kwargs) -> None:
        # Call the super constructor
        super(torch.Tensor, self).__init__(*args, **kwargs)

        # Set the data
        self.data = data
        self.ckks_data = ckks_data

    # Overridden methods
    def clone(self, *args, **kwargs) -> "CKKSWrapper":
        # Clone the data
        data = super(torch.Tensor, self).clone(*args, **kwargs)

        # Clone the ckks_data
        ckks_data: CKKSTensor = self.ckks_data.copy()  # type: ignore

        # Create the new instance
        instance = CKKSWrapper(data, ckks_data)

        return instance

    # Overridden methods
    def view_as(self, other: "CKKSWrapper") -> "CKKSWrapper":
        self.data = super(torch.Tensor, self).view_as(other)

        return self

    # CKKS Operation (with shape change)
    def do_encryption(self, context: ts.Context) -> "CKKSWrapper":
        # Define the new CKKS tensor
        new_ckks_tensor = ts.ckks_tensor(context, self.data.tolist())

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Blur the data
        tensor = torch.zeros(new_ckks_tensor.shape)

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation (with shape change)
    def do_linear(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> "CKKSWrapper":
        # Apply the linear transformation to the encrypted input
        new_ckks_tensor = self.ckks_data.mm(
            ts.plain_tensor(weight.t().tolist())
        ).add(
            ts.plain_tensor(bias.tolist())
        ) if bias is not None else self.ckks_data.mm(
            ts.plain_tensor(weight.t().tolist())
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Change the shape of the data
        tensor = torch.zeros(new_ckks_tensor.shape)

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation (with shape change)
    def do_sum(self, axis: int) -> "CKKSWrapper":
        # Apply the sum function to the encrypted input
        new_ckks_tensor: CKKSTensor = self.ckks_data.sum(
            axis=axis
        )  # type: ignore

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Change the shape of the data
        tensor = torch.zeros(new_ckks_tensor.shape)

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation
    def do_multiplication(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the multiplication function to the encrypted input
        new_ckks_tensor: CKKSTensor = self.ckks_data.mul(other.ckks_data)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Change the shape of the data
        tensor = torch.zeros(new_ckks_tensor.shape)

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation
    def do_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        new_ckks_tensor: CKKSTensor = self.ckks_data.square()  # type: ignore

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_activation_function(self, polyval_coeffs: np.ndarray) -> "CKKSWrapper":
        # Apply the activation function to the encrypted input
        new_ckks_tensor: CKKSTensor = self.ckks_data.polyval(
            polyval_coeffs.tolist()
        )  # type: ignore

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_multiplicative_inverse(self, polyval_coeffs: np.ndarray, iterations: int) -> "CKKSWrapper":
        # Source: https://en.wikipedia.org/wiki/Division_algorithm#Newton%E2%80%93Raphson_division

        # Start with an initial guess (encoded as a scalar)
        # For multiplicative inverse, good initial guess can be crucial - let's assume approx. inverse is known
        # This should be based on some estimation method
        new_ckks_tensor: CKKSTensor = self.ckks_data.polyval(
            polyval_coeffs.tolist()
        )  # type: ignore

        # Newton-Raphson iteration to refine the inverse
        for _ in range(iterations):
            prod = self.ckks_data.mul(new_ckks_tensor)  # d * x_n

            neg_prod: CKKSTensor = prod.neg()  # type: ignore
            correction = neg_prod.add(
                torch.ones(
                    self.ckks_data.shape
                ).mul(2).tolist()
            )  # 2 - d * x_n

            new_ckks_tensor = new_ckks_tensor.mul(
                correction
            )  # x_n * (2 - d * x_n)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # Data Operation
    def do_decryption(self) -> "CKKSWrapper":
        # Define the new tensor
        new_tensor = torch.tensor(
            self.ckks_data.decrypt().tolist(), requires_grad=True
        )

        # Update the data
        self.data = new_tensor.data

        return self

    # Data Operation
    def do_activation_function_backward(self, polyval_derivative: Callable[[float], float]) -> "CKKSWrapper":
        # Apply the activation function backward to the data
        new_tensor = torch.tensor(
            np.vectorize(polyval_derivative)(self.data)
        )

        # Update the data
        self.data = new_tensor.data

        return self

    # Data Operation
    def do_clamp(self, min: float, max: float) -> "CKKSWrapper":
        # Apply the clamp function to the data
        new_tensor = torch.clamp(self.data, min=min, max=max)

        # Update the data
        self.data = new_tensor.data

        return self
