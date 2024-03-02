from torch.utils.data import DataLoader, random_split
from torchseal.utils import seed_worker
from dataloader import FraminghamDataset

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
        train_loss = 0.0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average losses
        train_loss = train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch + 1, train_loss))

    # Model in evaluation mode
    model.eval()

    return model


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Load the data
    dataset = FraminghamDataset(csv_file="./data/framingham.csv")

    # Split the data into training and testing
    generator = torch.Generator().manual_seed(73)
    train_dataset, _ = random_split(dataset, [0.9, 0.1], generator=generator)

    # Set the batch size
    batch_size = 64

    # Create the data loader
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Get the number of features
    n_features = dataset.features.shape[1]

    # Create the model, criterion, and optimizer
    model = LogisticRegression(n_features)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCELoss()

    # Train the model
    model = train(model, train_loader, criterion, optim, n_epochs=5)

    # Save the model
    torch.save(model.state_dict(), "./parameters/framingham/model.pth")
