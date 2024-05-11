import typing
import numpy as np
import torch
import tenseal as ts


# TODO: Move all overridden to a __torch_dispatch__ implementation
# TODO: Make sure all non-overridden methods are able to accept both CKKSWrapper and torch.Tensor
class CKKSWrapper(torch.Tensor):
    __ckks_data: ts.CKKSTensor

    @property
    def ckks_data(self) -> ts.CKKSTensor:
        return self.__ckks_data

    @ckks_data.setter
    def ckks_data(self, ckks_data: ts.CKKSTensor) -> None:
        self.__ckks_data = ckks_data

    # Special methods
    @staticmethod
    def __new__(cls, context: ts.Context, data: torch.Tensor) -> "CKKSWrapper":
        # Create the instance
        instance = torch.Tensor._make_subclass(cls, data)

        return instance

    # Special methods
    def __init__(self, context: ts.Context, data: torch.Tensor) -> None:
        # Call the super constructor
        super(CKKSWrapper, self).__init__()

        # Set the ckks_data
        self.ckks_data = ts.ckks_tensor(context, data.tolist())

        # Set the data to zeros
        self.data = torch.zeros(data.shape)

    # Overridden methods
    def clone(self) -> "CKKSWrapper":
        # Get the context
        context = self.ckks_data.context()

        # Clone the data
        data = super(CKKSWrapper, self).clone()

        # Create the new instance
        instance = CKKSWrapper(
            context,
            data,
        )

        return instance

    # Overridden methods
    def view_as(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Resize the ckks_data
        self.ckks_data = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.reshape(list(self.data.size()))
        )

        # Resize the data
        self.data = super(CKKSWrapper, self).view_as(other)

        return self

    # CKKS Operation (with shape change)
    def do_matrix_multiplication(self, other: torch.Tensor) -> "CKKSWrapper":
        # Apply the linear transformation to the encrypted input
        new_ckks_tensor = self.ckks_data.mm(other.tolist())

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
            ts.CKKSTensor,
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
    def do_addition(self, other: torch.Tensor) -> "CKKSWrapper":
        # Apply the addition function to the encrypted input
        new_ckks_tensor = self.ckks_data.add(other)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_scalar_multiplication(self, scalar: float) -> "CKKSWrapper":
        # Apply the scalar multiplication function to the encrypted input
        new_ckks_tensor = self.ckks_data.mul(scalar)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_enc_element_multiplication(self, enc_other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the multiplication function to the encrypted input
        new_ckks_tensor = self.ckks_data.mul(enc_other.ckks_data)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_negation(self) -> "CKKSWrapper":
        # Apply the negation function to the encrypted input
        new_ckks_tensor = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.neg()
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        new_ckks_tensor = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.square()
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_polynomial(self, polyval_coeffs: np.ndarray) -> "CKKSWrapper":
        # Apply the activation function to the encrypted input
        new_ckks_tensor = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.polyval(
                polyval_coeffs.tolist()
            )
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_softmax(self, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int) -> "CKKSWrapper":
        # Apply the exp function to the encrypted input
        act_x = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.polyval(
                exp_coeffs.tolist()
            )
        )

        # Copy the encrypted activation
        act_x_copy = typing.cast(
            ts.CKKSTensor,
            act_x.copy()
        )

        # Apply the sum function to the encrypted input
        sum_x = typing.cast(
            ts.CKKSTensor,
            act_x.sum(axis=1)
        )

        # Apply the multiplicative inverse function to the encrypted input
        inverse_sum_x = typing.cast(
            ts.CKKSTensor,
            sum_x.polyval(
                inverse_coeffs.tolist()
            )
        )

        # Newton-Raphson iteration to refine the inverse
        for _ in range(inverse_iterations):
            prod = sum_x.mul(inverse_sum_x)  # d * x_n

            correction = typing.cast(
                ts.CKKSTensor, prod.neg()
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
            ts.CKKSTensor,
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

    # Data Operation
    def do_decryption(self) -> "CKKSWrapper":
        # Define the new tensor
        new_data_tensor = torch.tensor(
            self.ckks_data.decrypt().tolist(), requires_grad=True
        )

        # Update the data
        self.data = new_data_tensor.data

        return self
