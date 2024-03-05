from torch import Tensor
from tenseal import CKKSVector

import torch
import tenseal as ts


class CKKSWrapper(Tensor):
    __ckks_data: CKKSVector

    @property
    def ckks_data(self) -> CKKSVector:
        return self.__ckks_data

    @ckks_data.setter
    def ckks_data(self, ckks_data: CKKSVector) -> None:
        self.__ckks_data = ckks_data

    # # Logging method
    # @classmethod
    # def __torch_function__(cls, func, types, *args, **kwargs):
    #     # Print the function and the types
    #     print(f"Function: {func}")

    #     return super(CKKSWrapper, cls).__torch_function__(func, types, *args, **kwargs)

    # Overridden methods
    def __new__(cls, _data: Tensor, _ckks_data: CKKSVector, *args, **kwargs) -> "CKKSWrapper":
        # Create the tensor
        data = super(CKKSWrapper, cls).__new__(cls, *args, **kwargs)

        # Create the instance
        instance = torch.Tensor._make_subclass(cls, data)

        return instance

    # Overridden methods
    def __init__(self, data: Tensor, ckks_data: CKKSVector, *args, **kwargs) -> None:
        # Call the super constructor
        super(Tensor, self).__init__(*args, **kwargs)

        # Set the data
        self.data = data
        self.ckks_data = ckks_data

    # Overridden methods
    def clone(self, *args, **kwargs) -> "CKKSWrapper":
        # Clone the data
        data = super(Tensor, self).clone(*args, **kwargs)
        ckks_data: CKKSVector = self.ckks_data.copy()  # type: ignore

        # Create the new instance
        instance = CKKSWrapper(data, ckks_data)

        return instance

    # Overridden methods

    def view_as(self, other: "CKKSWrapper") -> "CKKSWrapper":
        self.data = super(Tensor, self).view_as(other)

        return self

    # CKKS Operation
    def do_conv2d(self, weight: Tensor, bias: Tensor, windows_nb: int) -> "CKKSWrapper":
        # Apply the convolution to the encrypted input
        new_ckks_vector = CKKSVector.pack_vectors([
            self.ckks_data.conv2d_im2col(kernel, windows_nb).add(bias) for kernel, bias in zip(weight.tolist(), bias.tolist())
        ])

        # Change the shape of the data
        tensor = torch.rand(new_ckks_vector.size())

        # Update the CKKS data
        self.ckks_data = new_ckks_vector

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation
    def do_linear(self, weight: Tensor, bias: Tensor) -> "CKKSWrapper":
        # Apply the linear transformation to the encrypted input
        new_ckks_vector = self.ckks_data.matmul(
            weight.t().tolist()
        ).add(
            bias.tolist()
        )

        # Change the shape of the data
        tensor = torch.rand(new_ckks_vector.size())

        # Update the data
        self.ckks_data = new_ckks_vector

        # Update the data
        self.data = tensor.data

        return self

    # CKKS Operation
    def do_sigmoid(self) -> "CKKSWrapper":
        # TODO: Do approximation of the sigmoid function using the polynomial approximation
        new_ckks_vector: CKKSVector = self.ckks_data.polyval(
            [0.5, 0.197, 0, -0.004]
        )  # type: ignore

        # Update the data
        self.ckks_data = new_ckks_vector

        return self

    # CKKS Operation
    def do_square(self) -> "CKKSWrapper":
        # Apply the square function to the encrypted input
        new_ckks_vector: CKKSVector = self.ckks_data.square()  # type: ignore

        # Update the data
        self.ckks_data = new_ckks_vector

        return self

    # CKKS Operation
    def do_encryption(self, context: ts.Context) -> "CKKSWrapper":
        # Define the new CKKS vector
        new_ckks_vector = ts.ckks_vector(context, self.data.tolist())

        # Update the data
        self.ckks_data = new_ckks_vector

        return self

    # Data Operation
    def do_sigmoid_backward(self) -> "CKKSWrapper":
        # Apply the sigmoid backward function to the data
        new_tensor = torch.tensor(
            list(map(lambda x: 0.197 - 0.008 * x, self.data))
        )

        # Update the data
        self.data = new_tensor.data

        return self

    # Data Operation
    def do_square_backward(self) -> "CKKSWrapper":
        # Apply the square backward function to the data
        new_tensor = torch.tensor(
            list(map(lambda x: 2 * x, self.data))
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

    # Data Operation
    def do_decryption(self) -> "CKKSWrapper":
        # Define the new tensor
        new_tensor = torch.tensor(
            self.ckks_data.decrypt(), requires_grad=True
        )

        # Update the data
        self.data = new_tensor.data

        return self
