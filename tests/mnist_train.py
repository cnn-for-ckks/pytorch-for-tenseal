from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import torch


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
            output = model.forward(data)
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


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)

    # Load the data
    train_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )

    # Set the batch size
    batch_size = 64

    # Create the data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    # Create the model, criterion, and optimizer
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train(model, train_loader, criterion, optimizer, 10)

    # Save the model
    torch.save(model.state_dict(), "./parameters/MNIST/model.pth")