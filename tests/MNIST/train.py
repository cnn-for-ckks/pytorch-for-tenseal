from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchseal.utils import seed_worker

import torch
import numpy as np
import random


class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10) -> None:
        super(ConvNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1.forward(x)

        # the model uses the square activation function
        x = x * x

        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1.forward(x)
        x = x * x
        x = self.fc2.forward(x)

        return x


def train(model: ConvNet, train_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, n_epochs: int = 10) -> ConvNet:
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
    train_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )

    # Set the batch size
    batch_size = 64

    # Create the data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Create the model, criterion, and optimizer
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train(model, train_loader, criterion, optimizer, n_epochs=10)

    # Save the model
    torch.save(model.state_dict(), "./parameters/MNIST/model.pth")
