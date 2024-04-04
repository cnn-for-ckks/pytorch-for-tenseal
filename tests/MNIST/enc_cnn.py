from typing import Optional
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchseal.wrapper.ckks import CKKSWrapper
from cnn import ConvNet
from utils import seed_worker

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


class EncConvNet(torch.nn.Module):
    def __init__(self, hidden=32, output=10, torch_nn: Optional[ConvNet] = None) -> None:
        super(EncConvNet, self).__init__()

        # Define the layers
        self.act1 = torchseal.nn.Square()
        self.act2 = torchseal.nn.Square()

        # Create the encrypted model
        if torch_nn is not None:
            self.conv1 = torchseal.nn.Conv2d(
                # Required parameters
                in_channel=torch_nn.conv1.in_channels,
                out_channel=torch_nn.conv1.out_channels,
                kernel_size=(
                    torch_nn.conv1.kernel_size[0], torch_nn.conv1.kernel_size[1]
                ),
                output_size=torch.Size([1, 28, 28]),
                stride=torch_nn.conv1.stride[0],
                padding=0,

                # Optional parameters
                weight=torch_nn.conv1.weight.data,
                bias=torch_nn.conv1.bias.data if torch_nn.conv1.bias is not None else None,
            )

            self.avg_pool = torchseal.nn.AvgPool2d(
                n_channel=torch_nn.conv1.out_channels,
                kernel_size=(2, 2),
                output_size=torch.Size([2, 8, 8]),
                stride=2,
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
                in_channel=1, out_channel=2, kernel_size=(7, 7), stride=3, output_size=torch.Size([1, 28, 28])
            )
            self.avg_pool = torchseal.nn.AvgPool2d(
                n_channel=2,
                kernel_size=(2, 2),
                output_size=torch.Size([2, 8, 8]),
                stride=2,
            )
            self.fc1 = torchseal.nn.Linear(32, hidden)
            self.fc2 = torchseal.nn.Linear(hidden, output)

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        # Convolutional layer
        first_result = self.conv1.forward(enc_x)

        # Average pooling layer
        first_result_averaged = self.avg_pool.forward(first_result)

        # Square activation function
        first_result_squared = self.act1.forward(first_result_averaged)

        # Fully connected layer
        second_result = self.fc1.forward(first_result_squared)

        # Square activation function
        second_result_squared = self.act2.forward(second_result)

        # Fully connected layer
        third_result = self.fc2.forward(second_result_squared)

        return third_result


def enc_train(context: ts.Context, enc_model: EncConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, n_epochs: int = 10) -> EncConvNet:
    # Model in training mode
    enc_model.train()

    for epoch in range(n_epochs):
        train_loss = 0.

        for raw_data, raw_target in train_loader:
            optimizer.zero_grad()

            # Encoding and encryption
            data = raw_data.view(-1).tolist()
            enc_data = ts.ckks_vector(context, data)
            enc_data_wrapper = CKKSWrapper(
                torch.rand(enc_data.size()), enc_data
            )

            # Encrypted evaluation
            enc_output = enc_model.forward(enc_data_wrapper)

            # Decryption of result
            output = enc_output.do_decryption()

            # Compute loss
            target = raw_target[0]  # TODO: Handle larger batch sizes
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average losses
        train_loss = 0 if len(
            train_loader
        ) == 0 else train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch + 1, train_loss))

    # Model in evaluation mode
    enc_model.eval()

    return enc_model


def enc_test(context: ts.Context, enc_model: EncConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # Model in evaluation mode
    enc_model.eval()

    for raw_data, raw_target in test_loader:
        # Encoding and encryption
        data = raw_data.view(-1).tolist()
        enc_data = ts.ckks_vector(context, data)
        enc_data_wrapper = CKKSWrapper(
            torch.rand(enc_data.size()), enc_data
        )

        # Encrypted evaluation
        enc_output = enc_model.forward(enc_data_wrapper)

        # Decryption of result using client secret key
        output = enc_output.do_decryption()

        # Compute loss
        target = raw_target[0]  # TODO: Handle larger batch sizes
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


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

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

    # Load the data
    train_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Subset the data
    subset_train_data = Subset(train_data, list(range(20)))
    subset_test_data = Subset(test_data, list(range(20)))

    # Set the batch size
    batch_size = 1  # TODO: Handle larger batch sizes

    # Create the data loaders
    subset_train_loader = DataLoader(
        subset_train_data, batch_size=batch_size, worker_init_fn=seed_worker
    )
    subset_test_loader = DataLoader(
        subset_test_data, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Create the original model
    original_model = ConvNet()
    original_model.load_state_dict(
        torch.load(
            "./parameters/MNIST/original-model.pth"
        )
    )

    # Create the encrypted data loaders
    enc_subset_train_loader = DataLoader(
        subset_train_data, batch_size=batch_size, worker_init_fn=seed_worker
    )
    enc_subset_test_loader = DataLoader(
        subset_test_data, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Create the model, criterion, and optimizer
    enc_model = EncConvNet(torch_nn=original_model)
    enc_criterion = torch.nn.CrossEntropyLoss()
    enc_optimizer = torch.optim.Adam(enc_model.parameters(), lr=0.001)

    # Print the weights and biases of the model
    print("Ciphertext Model (Before Training):")
    print("Number of Parameters:", len(list(map(str, enc_model.parameters()))))
    print("\n".join(list(map(str, enc_model.parameters()))))
    print()

    # Encrypted training
    enc_model = enc_train(
        context,
        enc_model,
        enc_subset_train_loader,
        enc_criterion,
        enc_optimizer,
        n_epochs=10
    )
    print()

    # Print the weights and biases of the model
    print("Ciphertext Model (After Training):")
    print("\n".join(list(map(str, enc_model.parameters()))))
    print()

    # Encrypted evaluation
    enc_test(
        context,
        enc_model,
        enc_subset_test_loader,
        enc_criterion,
    )

    # Save the model
    torch.save(
        enc_model.state_dict(),
        "./parameters/MNIST/enc-trained-model.pth"
    )
