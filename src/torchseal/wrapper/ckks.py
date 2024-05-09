from typing import Callable, Optional
from tenseal import CKKSTensor

import typing
import numpy as np
import torch
import tenseal as ts


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
        ckks_data = typing.cast(CKKSTensor, self.ckks_data.copy())

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
        new_ckks_tensor = typing.cast(
            CKKSTensor,
            self.ckks_data.sum(axis=axis)
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Change the shape of the data
        tensor = torch.zeros(new_ckks_tensor.shape)

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation
    def do_scalar_multiplication(self, scalar: float) -> "CKKSWrapper":
        # Apply the scalar multiplication function to the encrypted input
        new_ckks_tensor = self.ckks_data.mul(scalar)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_element_multiplication(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the multiplication function to the encrypted input
        new_ckks_tensor = self.ckks_data.mul(other.ckks_data)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        new_ckks_tensor = typing.cast(
            CKKSTensor,
            self.ckks_data.square()
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_softmax(self, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int) -> "CKKSWrapper":
        # Apply the exp function to the encrypted input
        act_x = typing.cast(
            CKKSTensor,
            self.ckks_data.polyval(
                exp_coeffs.tolist()
            )
        )

        # Copy the encrypted activation
        act_x_copy = typing.cast(
            CKKSTensor,
            act_x.copy()
        )

        # Apply the sum function to the encrypted input
        sum_x = typing.cast(
            CKKSTensor,
            act_x.sum(axis=1)
        )

        print("sum_x", sum_x.decrypt().tolist())

        # Apply the multiplicative inverse function to the encrypted input
        inverse_sum_x = typing.cast(
            CKKSTensor,
            sum_x.polyval(
                inverse_coeffs.tolist()
            )
        )

        # Newton-Raphson iteration to refine the inverse
        for _ in range(inverse_iterations):
            prod = sum_x.mul(inverse_sum_x)  # d * x_n

            correction = typing.cast(
                CKKSTensor, prod.neg()
            ).add(
                torch.ones(
                    sum_x.shape
                ).mul(2).tolist()
            )  # 2 - d * x_n

            inverse_sum_x = inverse_sum_x.mul(
                correction
            )  # x_n * (2 - d * x_n)

        # Reshape the inverse sum
        reshaped_inverse_sum_x = typing.cast(
            CKKSTensor,
            inverse_sum_x.reshape(
                [inverse_sum_x.shape[0], 1]
            )
        )

        # Change the inverse sum to a matrix
        binary_matrix = torch.ones(1, act_x_copy.shape[1])
        matrix_inverse_sum = reshaped_inverse_sum_x.mm(binary_matrix.tolist())

        # Apply the division function to the encrypted input
        new_ckks_tensor = act_x_copy.mul(matrix_inverse_sum)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_activation_function(self, polyval_coeffs: np.ndarray) -> "CKKSWrapper":
        # Apply the activation function to the encrypted input
        new_ckks_tensor = typing.cast(
            CKKSTensor,
            self.ckks_data.polyval(
                polyval_coeffs.tolist()
            )
        )

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
