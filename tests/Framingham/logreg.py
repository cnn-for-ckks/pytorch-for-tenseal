from torch.utils.data import DataLoader, Subset, random_split

from dataloader import FraminghamDataset
from utils import seed_worker

import torch
import numpy as np
import random


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features: int) -> None:
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear.forward(x)

        x = torch.sigmoid(x)

        return x


def train(model: LogisticRegression, train_loader: DataLoader, criterion: torch.nn.BCELoss, optimizer: torch.optim.SGD, n_epochs: int = 10) -> LogisticRegression:
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

        print(
            "Epoch: {} \tTraining Loss (Plaintext): {:.6f}".format(
                epoch + 1, train_loss
            )
        )

    # Model in evaluation mode
    model.eval()

    return model


def test(model: LogisticRegression, test_loader: DataLoader, criterion: torch.nn.BCELoss) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.

    # Model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model.forward(data)
        loss = criterion.forward(output, target)
        test_loss += loss.item()

    # Calculate and print avg test loss
    test_loss = 0 if len(test_loader) == 0 else test_loss / len(test_loader)
    print(f"Average Test Loss (Plaintext): {test_loss:.6f}")


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Load the data
    dataset = FraminghamDataset(csv_file="./data/framingham.csv")

    # Take subset of the data
    subdataset = Subset(dataset, list(range(20)))

    # Split the data into training and testing
    generator = torch.Generator().manual_seed(73)
    train_dataset, test_dataset = random_split(
        subdataset, [0.5, 0.5], generator=generator
    )

    # Set the batch size
    batch_size = 1  # TODO: Handle larger batch sizes

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Get the number of features
    n_features = dataset.features.shape[1]

    # Create the model, criterion, and optimizer
    model = LogisticRegression(n_features)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCELoss()

    # Save the original model
    torch.save(
        model.state_dict(),
        "./parameters/framingham/original-model.pth"
    )

    # Print the weights and biases of the model
    print("Plaintext Model (Before Training):")
    print("\n".join(list(map(str, model.parameters()))))
    print()

    # Train the model
    model = train(model, train_loader, criterion, optim, n_epochs=10)
    print()

    # Print the weights and biases of the model
    print("Plaintext Model (After Training):")
    print("\n".join(list(map(str, model.parameters()))))
    print()

    # Test the model
    test(model, test_loader, criterion)
    print()

    # Save the model
    torch.save(model.state_dict(), "./parameters/framingham/trained-model.pth")
