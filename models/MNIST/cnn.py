from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import seed_worker

import torch
import numpy as np
import random


class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10) -> None:
        super(ConvNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(7, 7), stride=3)
        self.fc1 = torch.nn.Linear(64, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the convolutional layer
        x = self.conv1.forward(x)

        # Flatten the data
        x = x.view(-1, 64)

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
        average_train_loss = 0 if len(
            train_loader
        ) == 0 else train_loss / len(train_loader)

        print(
            "Average Training Loss for epoch {} (Plaintext): {:.6f}\n".format(
                epoch + 1, average_train_loss
            )
        )

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
        num_target = len(target)

        if num_target > 1:
            for i in range(num_target):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        else:
            label = target.data.item()
            class_correct[label] += correct.item()
            class_total[label] += 1

    # Calculate and print avg test loss
    average_test_loss = 0 if sum(
        class_total
    ) == 0 else test_loss / sum(class_total)
    print(f"Average Test Loss (Plaintext): {average_test_loss:.6f}\n")

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

    # Set the batch size
    batch_size = 2

    # Subset the data
    # NOTE: Remove subset to use the entire dataset
    subset_test_data = Subset(test_data, list(range(20)))

    # Create the data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        subset_test_data, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Create the model, criterion, and optimizer
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Train the model
    # trained_model = train(
    #     model, train_loader, criterion, optimizer, n_epochs=10
    # )

    # # Save the model
    # torch.save(
    #     trained_model.state_dict(),
    #     "./parameters/MNIST/trained-model.pth"
    # )

    # Load the model and test the model
    trained_model = ConvNet()
    trained_model.load_state_dict(
        torch.load(
            "./parameters/MNIST/trained-model.pth"
        )
    )

    # Test the model
    test(trained_model, test_loader, criterion)
