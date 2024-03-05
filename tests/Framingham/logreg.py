from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

import torch


class LogisticRegression(Module):
    def __init__(self, n_features: int) -> None:
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x: Tensor) -> Tensor:
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
        train_loss = train_loss / len(train_loader)

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
    test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss (Plaintext): {test_loss:.6f}")
