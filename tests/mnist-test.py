from typing import Tuple
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from tenseal import CKKSVector

import numpy as np
import torch
import torchseal
import tenseal as ts


class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        # the model uses the square activation function
        x = x * x

        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)

        return x


def train(model: ConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, n_epochs=10):
    # model in training mode
    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    # model in evaluation mode
    model.eval()
    return model


def test(model: ConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


class EncConvNet:
    def __init__(self, torch_nn: ConvNet) -> None:
        # Unpack the kernel size
        kernel_size_h, kernel_size_w = torch_nn.conv1.kernel_size

        # Create the encrypted model
        self.conv1 = torchseal.nn.Conv2d(
            torch_nn.conv1.out_channels,
            (kernel_size_h, kernel_size_w),
            torch_nn.conv1.weight.data.view(
                torch_nn.conv1.out_channels,
                kernel_size_h,
                kernel_size_w
            ),
            torch_nn.conv1.bias.data if torch_nn.conv1.bias is not None else None,
        )
        self.fc1 = torchseal.nn.Linear(
            torch_nn.fc1.in_features,
            torch_nn.fc1.out_features,
            torch_nn.fc1.weight.T.data,
            torch_nn.fc1.bias.data,
        )
        self.fc2 = torchseal.nn.Linear(
            torch_nn.fc2.in_features,
            torch_nn.fc2.out_features,
            torch_nn.fc2.weight.T.data,
            torch_nn.fc2.bias.data,
        )

    def forward(self, enc_x: CKKSVector, windows_nb: int) -> CKKSVector:
        # Convolutional layer
        first_result = self.conv1.forward(enc_x, windows_nb)

        # Square activation function
        first_result.square_()

        # Fully connected layer
        second_result = self.fc1.forward(first_result)

        # Square activation function
        second_result.square_()

        # Fully connected layer
        third_result = self.fc2.forward(second_result)

        return third_result


def enc_test(context: ts.Context, enc_model: EncConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, kernel_shape: Tuple[int, int], stride: int):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # unpack the kernel shape
    kernel_shape_h, kernel_shape_w = kernel_shape

    for data, target in test_loader:
        # Encoding and encryption
        result: Tuple[CKKSVector, int] = ts.im2col_encoding(
            context,
            data.view(28, 28).tolist(),
            kernel_shape_h,
            kernel_shape_w,
            stride
        )  # type: ignore[error from tenseal library]

        # Unpack the result
        x_enc, windows_nb = result

        # Encrypted evaluation
        enc_output = enc_model.forward(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)

    # Load the data
    train_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Set the batch size
    batch_size = 64

    # Create the samplers
    train_sampler = RandomSampler(train_data)
    test_sampler = RandomSampler(test_data, num_samples=50)

    # Create the data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, sampler=test_sampler
    )

    # Create the model, criterion, and optimizer
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train(model, train_loader, criterion, optimizer, 10)

    # Test the model
    test(model, test_loader, criterion)

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

    # Create the data loaders for encrypted evaluation
    enc_test_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

    # Create the encrypted model
    enc_model = EncConvNet(model)

    # Encrypted evaluation
    enc_test(
        context,
        enc_model,
        enc_test_loader,
        criterion,
        (7, 7),
        3
    )
