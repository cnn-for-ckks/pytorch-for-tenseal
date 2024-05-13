from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type
from torch.utils._pytree import tree_map, PyTree
from torchseal.state import CKKSState


import typing
import numpy as np
import torch
import tenseal as ts


# Helper functions
def unwrap(ckks_wrapper: Any) -> Any:
    return ckks_wrapper.plaintext_data if isinstance(ckks_wrapper, CKKSWrapper) else ckks_wrapper


# Helper functions
def wrap(tensor: Any) -> Any:
    return CKKSWrapper(tensor) if isinstance(tensor, torch.Tensor) else tensor


# CKKS Wrapper
class CKKSWrapper(torch.Tensor):
    __plaintext_data: torch.Tensor
    __ckks_data: Optional[ts.CKKSTensor] = None

    # Properties
    @property
    def ckks_data(self) -> ts.CKKSTensor:
        assert self.__ckks_data is not None, "CKKS data is not initialized"

        return self.__ckks_data

    # Properties
    @ckks_data.setter
    def ckks_data(self, ckks_data: Optional[ts.CKKSTensor]) -> None:
        self.__ckks_data = ckks_data

    # Properties
    @property
    def plaintext_data(self) -> torch.Tensor:
        return self.__plaintext_data

    # Properties
    @plaintext_data.setter
    def plaintext_data(self, plaintext_data: torch.Tensor) -> None:
        self.__plaintext_data = plaintext_data

    # Predicates
    def is_encrypted(self) -> bool:
        return self.__ckks_data is not None

    # Special methods
    @staticmethod
    def __new__(cls, plaintext_data: torch.Tensor) -> "CKKSWrapper":
        # Create the instance
        instance = torch.Tensor._make_subclass(cls, plaintext_data.to("meta"))

        return instance

    # Special methods
    # NOTE: This will just create unencrypted wrapper
    def __init__(self, plaintext_data: torch.Tensor) -> None:
        # Call the super constructor
        super(CKKSWrapper, self).__init__()

        # Set the plaintext data
        self.plaintext_data = plaintext_data

    # Special methods
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ENCRYPTED)" if self.is_encrypted() else f"{self.__class__.__name__}(plaintext_data={self.plaintext_data})"

    # Special methods
    @classmethod
    def __torch_dispatch__(cls, func: Callable, types: Iterable[Type], args: Tuple = (), kwargs: Dict = {}) -> PyTree:
        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    # Overridden methods
    def clone(self) -> "CKKSWrapper":
        # Clone the ckks_data
        ckks_data = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.copy()
        )

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # Overridden methods
    def view_as(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Resize the ckks_data
        ckks_data = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.reshape(other.ckks_data.shape)
        )

        # Resize the plaintext data
        plaintext_data = self.plaintext_data.view_as(other)

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation (with shape change)
    def ckks_matrix_multiplication(self, other: torch.Tensor) -> "CKKSWrapper":
        # Apply the linear transformation to the encrypted input
        ckks_data = self.ckks_data.mm(other.tolist())

        # Create an empty plaintext data tensor
        plaintext_data = torch.zeros(ckks_data.shape)

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation (with shape change)
    def ckks_encrypted_matrix_multiplication(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the linear transformation to the encrypted input
        ckks_data = self.ckks_data.mm(other.ckks_data)

        # Create an empty plaintext data tensor
        plaintext_data = torch.zeros(ckks_data.shape)

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation
    def ckks_addition(self, other: torch.Tensor) -> "CKKSWrapper":
        # Apply the addition function to the encrypted input
        ckks_data = self.ckks_data.add(other.tolist())

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation
    def ckks_encrypted_addition(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the addition function to the encrypted input
        ckks_data = self.ckks_data.add(other.ckks_data)

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation
    def ckks_encrypted_negation(self) -> "CKKSWrapper":
        # Apply the negation function to the encrypted input
        ckks_data = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.neg()
        )

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation
    def ckks_encrypted_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        ckks_data = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.square()
        )

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation
    def ckks_encrypted_polynomial(self, polyval_coeffs: np.ndarray) -> "CKKSWrapper":
        # Apply the activation function to the encrypted input
        ckks_data = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.polyval(
                polyval_coeffs.tolist()
            )
        )

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation
    def ckks_encrypted_softmax(self, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int) -> "CKKSWrapper":
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
        ckks_data = act_x_copy.mul(matrix_inverse_sum)

        # Clone the plaintext data
        plaintext_data = self.plaintext_data.clone()

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # CKKS Operation (with shape change)
    def ckks_encrypted_negative_log_likelihood_loss(
        self, enc_target: "CKKSWrapper", log_coeffs: np.ndarray, batch_size: int
    ) -> "CKKSWrapper":
        # Apply log softmax to the output
        enc_log_output = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.polyval(
                log_coeffs.tolist()
            )
        )

        # Calculate the negative log likelihood loss
        enc_log_probs = typing.cast(
            ts.CKKSTensor,
            enc_log_output.mul(enc_target.ckks_data).sum(axis=1)
        )

        # Calculate the loss
        ckks_data = typing.cast(
            ts.CKKSTensor, enc_log_probs.sum(axis=0)
        ).mul(-1 / batch_size)

        # Create an empty plaintext data tensor
        plaintext_data = torch.zeros(ckks_data.shape)

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # Encrypt-Decrypt Operations
    def encrypt(self) -> "CKKSWrapper":
        # Get the state of the CKKS
        state = CKKSState()

        # Create the new encrypted tensor
        ckks_data = ts.ckks_tensor(state.context, self.plaintext_data.tolist())

        # Create an empty plaintext data tensor
        plaintext_data = torch.zeros(ckks_data.shape)

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance

    # Encrypt-Decrypt Operations
    def decrypt(self) -> "CKKSWrapper":
        # Set the encrypted data to None
        ckks_data = None

        # Decrypt the data
        plaintext_data = torch.tensor(
            self.ckks_data.decrypt().tolist()
        )

        # Create the new instance
        instance = CKKSWrapper.__new__(CKKSWrapper, plaintext_data)

        # Set the ckks data
        instance.ckks_data = ckks_data

        # Set the plaintext data
        instance.plaintext_data = plaintext_data

        return instance
