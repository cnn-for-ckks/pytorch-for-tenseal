import typing
import numpy as np
import torch
import tenseal as ts


# TODO: Move overridden methods to the __torch_dispatch__ method
class CKKSWrapper(torch.Tensor):
    __ckks_data: ts.CKKSTensor

    # Properties
    @property
    def ckks_data(self) -> ts.CKKSTensor:
        return self.__ckks_data

    # Properties
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
    # NOTE: Will automatically encrypt the data tensor
    def __init__(self, context: ts.Context, data: torch.Tensor) -> None:
        # Call the super constructor
        super(CKKSWrapper, self).__init__()

        # Create the encrypted tensor
        new_ckks_data = ts.ckks_tensor(context, data.tolist())

        # Set the ckks_data
        self.ckks_data = new_ckks_data

        # Create an empty data tensor
        new_data_tensor = torch.zeros(new_ckks_data.shape)

        # Blur the data
        self.data = new_data_tensor.data

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

        # Create an empty data tensor
        new_data_tensor = torch.zeros(new_ckks_tensor.shape)

        # Reshape the data
        self.data = new_data_tensor.data

        return self

    # CKKS Operation (with shape change)
    def do_encrypted_matrix_multiplication(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the linear transformation to the encrypted input
        new_ckks_tensor = self.ckks_data.mm(other.ckks_data)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Create an empty data tensor
        new_data_tensor = torch.zeros(new_ckks_tensor.shape)

        # Reshape the data
        self.data = new_data_tensor.data

        return self

    # CKKS Operation
    def do_addition(self, other: torch.Tensor) -> "CKKSWrapper":
        # Apply the addition function to the encrypted input
        new_ckks_tensor = self.ckks_data.add(other.tolist())

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_encrypted_addition(self, other: "CKKSWrapper") -> "CKKSWrapper":
        # Apply the addition function to the encrypted input
        new_ckks_tensor = self.ckks_data.add(other.ckks_data)

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_encrypted_negation(self) -> "CKKSWrapper":
        # Apply the negation function to the encrypted input
        new_ckks_tensor = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.neg()
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_encrypted_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        new_ckks_tensor = typing.cast(
            ts.CKKSTensor,
            self.ckks_data.square()
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_encrypted_polynomial(self, polyval_coeffs: np.ndarray) -> "CKKSWrapper":
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
    def do_encrypted_softmax(self, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int) -> "CKKSWrapper":
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

    # CKKS Operation (with shape change)
    def do_encrypted_negative_log_likelihood_loss(
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
        enc_loss = typing.cast(
            ts.CKKSTensor, enc_log_probs.sum(axis=0)
        ).mul(-1 / batch_size)

        # Update the encrypted data
        self.ckks_data = enc_loss

        # Create an empty data tensor
        new_data_tensor = torch.zeros(enc_loss.shape)

        # Reshape the data
        self.data = new_data_tensor.data

        return self

    # Encrypt-Decrypt Operations
    def do_encryption(self) -> "CKKSWrapper":
        # Get the previous context
        context = self.ckks_data.context()

        # Create the new encrypted tensor
        new_ckks_tensor = ts.ckks_tensor(context, self.data.tolist())

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Create an empty data tensor
        new_data_tensor = torch.zeros(new_ckks_tensor.shape)

        # Blur the data
        self.data = new_data_tensor.data

        return self

    # Encrypt-Decrypt Operations
    def do_decryption(self) -> "CKKSWrapper":
        # Define the new tensor
        new_data_tensor = torch.tensor(
            self.ckks_data.decrypt().tolist(), requires_grad=True
        )

        # Update the data
        self.data = new_data_tensor.data

        return self
