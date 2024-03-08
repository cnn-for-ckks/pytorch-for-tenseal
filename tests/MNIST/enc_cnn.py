from typing import Tuple, Optional
from torch.nn import Module
from torch.utils.data import DataLoader
from tenseal import CKKSVector
from torchseal.function import SquareFunction
from torchseal.utils import im2col_encoding
from torchseal.wrapper.ckks import CKKSWrapper
from cnn import ConvNet

import torch
import torchseal
import tenseal as ts
import numpy as np


# TODO: Encrypt weight and bias of the model when training the model
class EncConvNet(Module):
    def __init__(self, hidden=64, output=10, torch_nn: Optional[ConvNet] = None) -> None:
        super(EncConvNet, self).__init__()

        # Create the encrypted model
        if torch_nn is not None:
            self.conv1 = torchseal.nn.Conv2d(
                output_size=(28, 28),  # NOTE: Hardcoded for MNIST dataset
                kernel_size=(
                    torch_nn.conv1.kernel_size[0], torch_nn.conv1.kernel_size[1]
                ),
                stride=torch_nn.conv1.stride[0],
                padding=0,  # NOTE: Hardcoded for MNIST dataset
                weight=torch_nn.conv1.weight.data.view(
                    torch_nn.conv1.in_channels,
                    torch_nn.conv1.out_channels,
                    torch_nn.conv1.kernel_size[0],
                    torch_nn.conv1.kernel_size[1]
                ),
                bias=torch_nn.conv1.bias.data if torch_nn.conv1.bias is not None else None,
            )

            self.fc1 = torchseal.nn.Linear(
                torch_nn.fc1.in_features,
                torch_nn.fc1.out_features,
                torch_nn.fc1.weight.data,
                torch_nn.fc1.bias.data,
            )

            self.fc2 = torchseal.nn.Linear(
                torch_nn.fc2.in_features,
                torch_nn.fc2.out_features,
                torch_nn.fc2.weight.data,
                torch_nn.fc2.bias.data,
            )

        else:
            self.conv1 = torchseal.nn.Conv2d(
                output_size=(28, 28), kernel_size=(7, 7), stride=3, padding=0
            )
            self.fc1 = torchseal.nn.Linear(64, hidden)
            self.fc2 = torchseal.nn.Linear(hidden, output)

    def forward(self, enc_x: CKKSWrapper, num_row: int, num_col: int) -> CKKSWrapper:
        # Convolutional layer
        first_result = self.conv1.forward(enc_x, num_row, num_col)

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


def enc_train(context: ts.Context, enc_model: EncConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, kernel_size: Tuple[int, int], stride: int, n_epochs: int = 10) -> EncConvNet:
    # Model in training mode
    enc_model.train()

    for epoch in range(1, n_epochs+1):
        train_loss = 0.

        for raw_data, raw_target in train_loader:
            optimizer.zero_grad()

            # Encoding and encryption
            result: Tuple[CKKSVector, int, int] = im2col_encoding(
                context,
                raw_data.view(1, 28, 28),
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )

            # Unpack the result
            enc_unfolded_image, num_row, num_col = result
            enc_x_wrapper = CKKSWrapper(
                torch.rand(enc_unfolded_image.size()),
                enc_unfolded_image
            )

            # Encrypted evaluation
            enc_output = enc_model.forward(
                enc_x_wrapper, num_row, num_col
            )

            # Decryption of result
            output = enc_output.do_decryption()

            # Compute loss
            target = raw_target[0]
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average losses
        train_loss = 0 if len(
            train_loader
        ) == 0 else train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    # Model in evaluation mode
    enc_model.eval()

    return enc_model


def enc_test(context: ts.Context, enc_model: EncConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, kernel_size: Tuple[int, int], stride: int) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # Model in evaluation mode
    enc_model.eval()

    for raw_data, raw_target in test_loader:
        # Encoding and encryption
        result: Tuple[CKKSVector, int, int] = im2col_encoding(
            context,
            raw_data.view(1, 28, 28),
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

        # Unpack the result
        enc_unfolded_image, num_row, num_col = result
        enc_x_wrapper = CKKSWrapper(
            torch.rand(enc_unfolded_image.size()),
            enc_unfolded_image
        )

        # Encrypted evaluation
        enc_output = enc_model.forward(
            enc_x_wrapper, num_row, num_col
        )
        # Decryption of result using client secret key
        output = enc_output.do_decryption()

        # Compute loss
        target = raw_target[0]
        loss = criterion.forward(output, target)
        test_loss += loss.item()

        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 0)

        # Compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # Calculate test accuracy for each object class
        label = target.data.item()
        class_correct[label] += correct.item()
        class_total[label] += 1

    # Calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f"Test Loss for Encrypted Data: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {0 if class_total[label] == 0 else int(100 * class_correct[label] / class_total[label])}% "
            f"({int(class_correct[label])}/{int(class_total[label])})"
        )

    print(
        f"\nTest Accuracy for Encrypted Data (Overall): {0 if np.sum(class_total) == 0 else int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )
