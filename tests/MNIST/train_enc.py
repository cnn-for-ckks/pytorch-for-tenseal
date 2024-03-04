from typing import Tuple, Optional
from torch.nn import Module
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from tenseal import CKKSVector
from torchseal.function import SquareFunction
from torchseal.utils import seed_worker
from torchseal.wrapper.ckks import CKKSWrapper
from train import ConvNet

import torch
import torchseal
import tenseal as ts
import numpy as np
import random


class EncConvNet(Module):
    def __init__(self, hidden=64, output=10, torch_nn: Optional[ConvNet] = None) -> None:
        super(EncConvNet, self).__init__()

        # Create the encrypted model
        self.conv1 = torchseal.nn.Conv2d(
            torch_nn.conv1.out_channels,
            (torch_nn.conv1.kernel_size[0], torch_nn.conv1.kernel_size[1]),
            torch_nn.conv1.weight.data.view(
                torch_nn.conv1.out_channels,
                torch_nn.conv1.kernel_size[0],
                torch_nn.conv1.kernel_size[1]
            ),
            torch_nn.conv1.bias.data if torch_nn.conv1.bias is not None else None,
        ) if torch_nn is not None else torchseal.nn.Conv2d(4, (7, 7))

        self.fc1 = torchseal.nn.Linear(
            torch_nn.fc1.in_features,
            torch_nn.fc1.out_features,
            torch_nn.fc1.weight.T.data,
            torch_nn.fc1.bias.data,
        ) if torch_nn is not None else torchseal.nn.Linear(256, hidden)

        self.fc2 = torchseal.nn.Linear(
            torch_nn.fc2.in_features,
            torch_nn.fc2.out_features,
            torch_nn.fc2.weight.T.data,
            torch_nn.fc2.bias.data,
        ) if torch_nn is not None else torchseal.nn.Linear(hidden, output)

    def forward(self, enc_x: CKKSWrapper, windows_nb: int) -> CKKSWrapper:
        # Convolutional layer
        first_result = self.conv1.forward(enc_x, windows_nb)

        # Square activation function
        first_result_squared: CKKSWrapper = SquareFunction.apply(
            first_result
        )  # type: ignore

        # Fully connected layer
        second_result = self.fc1.forward(first_result_squared)

        # Square activation function
        second_result_squared: CKKSWrapper = SquareFunction.apply(
            second_result
        )  # type: ignore

        # Fully connected layer
        third_result = self.fc2.forward(second_result_squared)

        return third_result


def enc_train(context: ts.Context, enc_model: EncConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, kernel_shape: Tuple[int, int], stride: int, n_epochs: int = 10) -> EncConvNet:
    # Model in training mode
    enc_model.train()

    # Unpack the kernel shape
    kernel_shape_h, kernel_shape_w = kernel_shape

    for epoch in range(1, n_epochs+1):
        train_loss = 0.

        for data, target in train_loader:
            optimizer.zero_grad()

            # Encoding and encryption
            result: Tuple[CKKSVector, int] = ts.im2col_encoding(
                context,
                data.view(28, 28).tolist(),
                kernel_shape_h,
                kernel_shape_w,
                stride
            )  # type: ignore

            # Unpack the result
            enc_x, windows_nb = result
            enc_x_wrapper = CKKSWrapper(enc_x)

            # Encrypted evaluation
            enc_output = enc_model.forward(enc_x_wrapper, windows_nb)

            # Decryption of result
            output = enc_output.do_decryption()

            loss = criterion.forward(output, target)
            loss.backward()  # BUG: Backward autograd not called
            optimizer.step()  # BUG: Weight update not called
            train_loss += loss.item()

        # Calculate average losses
        train_loss = train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    # Model in evaluation mode
    enc_model.eval()

    return enc_model


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Load the data
    train_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )

    # Create the samplers
    sampler = RandomSampler(train_data, num_samples=10)

    # Set the batch size
    batch_size = 1

    # Create the data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler, worker_init_fn=seed_worker
    )

    # Create the model, criterion, and optimizer
    enc_model = EncConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(enc_model.parameters(), lr=0.001)

    # Controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # # NOTE: Check the weights and biases of the model
    # print("\n".join(list(map(str, enc_model.parameters()))))

    # Train the model
    enc_model = enc_train(
        context,
        enc_model,
        train_loader,
        criterion,
        optimizer,
        kernel_shape=(7, 7),
        stride=3,
        n_epochs=10
    )

    # # NOTE: Check the weights and biases of the model
    # print("\n".join(list(map(str, enc_model.parameters()))))

    # Save the model
    torch.save(enc_model.state_dict(), "./parameters/MNIST/model-enc.pth")
