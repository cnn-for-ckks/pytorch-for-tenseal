from torch.utils.data import DataLoader, random_split
from torchseal.utils import seed_worker
from dataloader import FraminghamDataset
from train import LogisticRegression

import torch
import numpy as np
import random


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
    print(f"Test Loss: {test_loss:.6f}\n")


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Load the data
    dataset = FraminghamDataset(csv_file="./data/framingham.csv")

    # Split the data into training and testing
    generator = torch.Generator().manual_seed(73)
    _, test_dataset = random_split(dataset, [0.9, 0.1], generator=generator)

    # Set the batch size
    batch_size = 64

    # Create the data loader
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Get the number of features
    n_features = dataset.features.shape[1]

    # Load the model
    model = LogisticRegression(n_features)
    model.load_state_dict(torch.load("./parameters/framingham/model.pth"))

    # Loss function
    criterion = torch.nn.BCELoss()

    # Test the model
    test(model, test_loader, criterion)
