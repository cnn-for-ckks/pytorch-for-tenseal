from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

import torch
import numpy as np


class ConvNet(Module):
    def __init__(self, hidden=64, output=10) -> None:
        super(ConvNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1.forward(x)

        # Flatten the data
        x = x.view(-1, 256)

        # Apply the activation function
        x = x * x

        # Apply the fully connected layers
        x = self.fc1.forward(x)

        # Apply the activation function
        x = x * x

        # Apply the fully connected layers
        x = self.fc2.forward(x)

        return x


def train(model: ConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, n_epochs: int = 10) -> ConvNet:
    # Model in training mode
    model.train()

    for epoch in range(n_epochs):
        train_loss = 0.

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model.forward(data)
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
    model.eval()

    return model


def test(model: ConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # Model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model.forward(data)
        loss = criterion.forward(output, target)
        test_loss += loss.item()

        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # Calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[0]
            class_correct[label] += correct.item()
            class_total[label] += 1

    # Calculate and print avg test loss
    test_loss = 0 if len(test_loader) == 0 else test_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {0 if class_total[label] == 0 else int(100 * class_correct[label] / class_total[label])}% "
            f"({int(class_correct[label])}/{int(class_total[label])})"
        )

    print(
        f"\nTest Accuracy (Overall): {0 if np.sum(class_total) == 0 else int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )
