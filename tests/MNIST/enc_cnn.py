from typing import Tuple, Optional
from torch.nn import Module
from torch.utils.data import DataLoader
from tenseal import CKKSVector
from torchseal.function import SquareFunction
from torchseal.wrapper.ckks import CKKSWrapper
from cnn import ConvNet

import torch
import torchseal
import tenseal as ts
import numpy as np


class EncConvNet(Module):
    def __init__(self, hidden=64, output=10, torch_nn: Optional[ConvNet] = None) -> None:
        super(EncConvNet, self).__init__()

        # Create the encrypted model
        self.conv1 = torchseal.nn.Conv2d(
            (torch_nn.conv1.kernel_size[0], torch_nn.conv1.kernel_size[1]),
            torch_nn.conv1.weight.data.view(
                torch_nn.conv1.kernel_size[0],
                torch_nn.conv1.kernel_size[1]
            ),
            torch_nn.conv1.bias.data if torch_nn.conv1.bias is not None else None,
        ) if torch_nn is not None else torchseal.nn.Conv2d((7, 7))

        self.fc1 = torchseal.nn.Linear(
            torch_nn.fc1.in_features,
            torch_nn.fc1.out_features,
            torch_nn.fc1.weight.data,
            torch_nn.fc1.bias.data,
        ) if torch_nn is not None else torchseal.nn.Linear(256, hidden)

        self.fc2 = torchseal.nn.Linear(
            torch_nn.fc2.in_features,
            torch_nn.fc2.out_features,
            torch_nn.fc2.weight.data,
            torch_nn.fc2.bias.data,
        ) if torch_nn is not None else torchseal.nn.Linear(hidden, output)

    def forward(self, enc_x: CKKSWrapper, windows_nb: int, stride: int, padding: int) -> CKKSWrapper:
        # Convolutional layer
        first_result = self.conv1.forward(enc_x, windows_nb, stride, padding)

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


def enc_train(context: ts.Context, enc_model: EncConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, kernel_shape: Tuple[int, int], stride: int, padding: int, n_epochs: int = 10) -> EncConvNet:
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
            enc_x_wrapper = CKKSWrapper(torch.rand(len(data)), enc_x)

            # Encrypted evaluation
            enc_output = enc_model.forward(
                enc_x_wrapper, windows_nb, stride, padding)

            # Decryption of result
            output = enc_output.do_decryption()

            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()  # BUG: Weight and bias are not accurately updated
            train_loss += loss.item()

        # Calculate average losses
        train_loss = 0 if len(
            train_loader
        ) == 0 else train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    # Model in evaluation mode
    enc_model.eval()

    return enc_model


def enc_test(context: ts.Context, enc_model: EncConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, kernel_shape: Tuple[int, int], stride: int, padding: int, ) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # Unpack the kernel shape
    kernel_shape_h, kernel_shape_w = kernel_shape

    # Model in evaluation mode
    enc_model.eval()

    for data, target in test_loader:
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
        enc_x_wrapper = CKKSWrapper(torch.rand(len(data)), enc_x)

        # Encrypted evaluation
        enc_output = enc_model.forward(
            enc_x_wrapper, windows_nb, stride, padding
        )

        # Decryption of result using client secret key
        output = enc_output.do_decryption().view(1, -1)

        # Compute loss
        loss = criterion.forward(output, target)
        test_loss += loss.item()

        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # Calculate test accuracy for each object class
        label = target.data[0]
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
