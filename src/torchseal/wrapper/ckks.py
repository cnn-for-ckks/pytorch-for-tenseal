from typing import Optional
from tenseal import CKKSTensor

import torch
import tenseal as ts
import numpy as np
import time


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
        ckks_data: CKKSTensor = self.ckks_data.copy()  # type: ignore

        # Create the new instance
        instance = CKKSWrapper(data, ckks_data)

        return instance

    # Overridden methods
    def view_as(self, other: "CKKSWrapper") -> "CKKSWrapper":
        self.data = super(torch.Tensor, self).view_as(other)

        return self

    # CKKS Operation
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

    # CKKS Operation
    def do_linear(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> "CKKSWrapper":
        print("Linear with Bias" if bias is not None else "Linear without Bias")
        print(f"Weight: {weight.shape}")
        print(f"Data: {self.ckks_data.shape}")

        # Apply the linear transformation to the encrypted input
        start_time = time.perf_counter()

        new_ckks_tensor = self.ckks_data.mm(
            ts.plain_tensor(weight.t().tolist())
        ).add(
            ts.plain_tensor(bias.tolist())
        ) if bias is not None else self.ckks_data.mm(
            ts.plain_tensor(weight.t().tolist())
        )

        end_time = time.perf_counter()

        print(
            f"Time taken for linear transformation: {end_time - start_time:.6f} seconds\n"
        )

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        # Change the shape of the data
        tensor = torch.zeros(new_ckks_tensor.shape)

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation
    def do_sigmoid(self) -> "CKKSWrapper":
        # Apply the sigmoid function to the data
        # TODO: Create an adjustable approximation to calculate the sigmoid function
        new_ckks_tensor: CKKSTensor = self.ckks_data.polyval(
            [0.5, 0.197, 0, -0.004]
        )  # type: ignore

        # Update the encrypted data
        self.ckks_data = new_ckks_tensor

        return self

    # CKKS Operation
    def do_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        new_ckks_tensor: CKKSTensor = self.ckks_data.square()  # type: ignore

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
    def do_sigmoid_backward(self) -> "CKKSWrapper":
        # Apply the sigmoid backward function to the data
        # TODO: Create an adjustable approximation to calculate the sigmoid backward function
        new_tensor = torch.tensor(
            np.vectorize(lambda x: 0.197 - 0.008 * x)(self.data)
        )

        # Update the data
        self.data = new_tensor.data

        return self

    # Data Operation
    def do_square_backward(self) -> "CKKSWrapper":
        # Apply the square backward function to the data
        new_tensor = torch.tensor(np.vectorize(lambda x: 2 * x)(self.data))

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
