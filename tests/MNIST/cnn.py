from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import seed_worker

import torch
import numpy as np
import random


class ConvNet(torch.nn.Module):
    def __init__(self, hidden=32, output=10) -> None:
        super(ConvNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=(7, 7), stride=3)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = torch.nn.Linear(32, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the convolutional layer
        x = self.conv1.forward(x)

        # Apply the average pooling layer
        x = self.avg_pool.forward(x)

        # Flatten the data
        x = x.view(-1, 32)

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
        for _ in range(len(target)):
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


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

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

    # Create the model, criterion, and optimizer
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save the original model
    torch.save(
        model.state_dict(),
        "./parameters/MNIST/original-model.pth"
    )

    # Print the weights and biases of the model
    print("Plaintext Model (Before Training):")
    print("Number of Parameters:", len(list(map(str, model.parameters()))))
    print("\n".join(list(map(str, model.parameters()))))
    print()

    # Train the model
    model = train(
        model, subset_train_loader, criterion, optimizer, n_epochs=10
    )
    print()

    # Print the weights and biases of the model
    print("Plaintext Model (After Training):")
    print("\n".join(list(map(str, model.parameters()))))
    print()

    # Test the model
    test(model, subset_test_loader, criterion)
    print()

    # Save the model
    torch.save(model.state_dict(), "./parameters/MNIST/trained-model.pth")
