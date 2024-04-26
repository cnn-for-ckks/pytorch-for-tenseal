from torch.utils.data import DataLoader, Subset, random_split
from torchseal.wrapper import CKKSWrapper
from torchseal.nn import Linear, Sigmoid

from logreg import LogisticRegression
from dataloader import FraminghamDataset
from utils import seed_worker

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


# NOTE: The weights and biases are not encrypted in this example (the secret key must be used in the training process for calculating convolution).
# NOTE: To create a secure model, the weights and biases must be transferred from plaintext model to encrypted model.
# NOTE: Because of this, we have to redefine the research sequencem).
class EncLogisticRegression(torch.nn.Module):
    def __init__(self, torch_nn: LogisticRegression) -> None:
        super(EncLogisticRegression, self).__init__()

        self.linear = Linear(
            in_features=torch_nn.linear.in_features,
            out_features=torch_nn.linear.out_features,
            weight=torch_nn.linear.weight.data,
            bias=torch_nn.linear.bias.data
        )
        self.activation_function = Sigmoid(
            start=-5, stop=5, num_of_sample=5, degree=3, approximation_type="minimax"
        )

    def forward(self, x: CKKSWrapper) -> CKKSWrapper:
        # Fully connected layer
        first_result = self.linear.forward(x)

        # Sigmoid activation function
        first_result_activated = self.activation_function.forward(first_result)

        return first_result_activated


def enc_test(context: ts.Context, enc_model: EncLogisticRegression, test_loader: DataLoader, criterion: torch.nn.BCELoss) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.

    # Model in evaluation mode
    enc_model.eval()

    for raw_data, raw_target in test_loader:
        # Encryption
        enc_data_wrapper = torchseal.ckks_wrapper(
            context, raw_data
        )

        # Encrypted evaluation
        enc_output = enc_model.forward(enc_data_wrapper)

        # Decryption using client secret key
        output = enc_output.do_decryption().clamp(0, 1)

        # Compute loss
        loss = criterion.forward(output, raw_target)
        test_loss += loss.item()

    # Calculate and print avg test loss
    average_test_loss = 0 if len(
        test_loader
    ) == 0 else test_loss / len(test_loader)
    print(f"\nAverage Test Loss (Ciphertext): {average_test_loss:.6f}")


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

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

    # Load the data
    dataset = FraminghamDataset(csv_file="./data/Framingham.csv")

    # Take subset of the data
    subdataset = Subset(dataset, list(range(100)))

    # Split the data into training and testing
    generator = torch.Generator().manual_seed(73)
    train_dataset, test_dataset = random_split(
        subdataset, [0.8, 0.2], generator=generator
    )

    # Set the batch size
    batch_size = 2

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Get the number of features
    n_features = dataset.features.shape[1]

    # Create the original model
    trained_model = LogisticRegression(n_features)
    trained_model.load_state_dict(
        torch.load(
            "./parameters/Framingham/trained-model.pth"
        )
    )

    # Create the model, criterion, and optimizer
    enc_model = EncLogisticRegression(torch_nn=trained_model)
    enc_optim = torch.optim.SGD(enc_model.parameters(), lr=0.1)
    enc_criterion = torch.nn.BCELoss()

    # Test the model
    enc_test(context, enc_model, test_loader, enc_criterion)
