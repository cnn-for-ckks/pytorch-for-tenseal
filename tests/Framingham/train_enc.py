from typing import Optional
from torch.nn import Module
from torch.utils.data import DataLoader, RandomSampler, random_split
from tenseal import CKKSVector
from torchseal.function import SigmoidFunction
from torchseal.utils import seed_worker
from train import LogisticRegression
from dataloader import FraminghamDataset

import torch
import tenseal as ts
import numpy as np
import random
import torchseal


class EncLogisticRegression(Module):
    def __init__(self, n_features: int, torch_nn: Optional[LogisticRegression] = None) -> None:
        super(EncLogisticRegression, self).__init__()

        self.linear = torchseal.nn.Linear(
            n_features,
            1,
            torch_nn.linear.weight.T.data,
            torch_nn.linear.bias.data
        ) if torch_nn is not None else torchseal.nn.Linear(n_features, 1)

    def forward(self, x: CKKSVector) -> CKKSVector:
        # Fully connected layer
        first_result = self.linear.forward(x)

        # Sigmoid activation function
        first_result_activated: CKKSVector = SigmoidFunction.apply(
            first_result
        )  # type: ignore

        return first_result_activated


def enc_train(context: ts.Context, enc_model: EncLogisticRegression, train_loader: DataLoader, criterion: torch.nn.BCELoss, optimizer: torch.optim.SGD, n_epochs: int = 10) -> EncLogisticRegression:
    # Model in training mode
    enc_model.train()

    for epoch in range(n_epochs):
        train_loss = 0.

        for data, target in train_loader:
            optimizer.zero_grad()

            # Encrypt the data
            raw_data = data[0].tolist()
            enc_data = ts.ckks_vector(context, raw_data)

            # Forward pass
            output = enc_model.forward(enc_data)

            # Encrypted evaluation
            enc_output = enc_model.forward(enc_data)

            # Decryption of result
            output = torch.tensor(
                # Clip the result to be in [0, 1]
                list(
                    map(
                        lambda x: 1. if x > 1 else 0. if x < 0 else x,
                        enc_output.decrypt()
                    )
                ),
                requires_grad=True
            ).view(1, -1)

            loss = criterion.forward(output, target)
            loss.backward()  # BUG: Gradient is not computed correctly
            optimizer.step()
            train_loss += loss.item()

        # Calculate average losses
        train_loss = train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch + 1, train_loss))

    # Model in evaluation mode
    enc_model.eval()

    return enc_model


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

    # Create the samplers
    sampler = RandomSampler(train_dataset, num_samples=20)

    # Set the batch size
    batch_size = 1

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, worker_init_fn=seed_worker
    )

    # Get the number of features
    n_features = dataset.features.shape[1]

    # Create the model, criterion, and optimizer
    model = EncLogisticRegression(n_features)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCELoss()

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

    # Train the model
    enc_model = enc_train(context, model, train_loader, criterion, optim)

    # Save the model
    torch.save(model.state_dict(), "./parameters/framingham/model-enc.pth")
