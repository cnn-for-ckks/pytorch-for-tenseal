from torch.utils.data import DataLoader, RandomSampler, random_split
from torchseal.utils import seed_worker
from torchseal.wrapper.ckks import CKKSWrapper
from dataloader import FraminghamDataset
from train_enc import EncLogisticRegression

import torch
import tenseal as ts
import numpy as np
import random


def enc_test(context: ts.Context, enc_model: EncLogisticRegression, test_loader: DataLoader, criterion: torch.nn.BCELoss) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.

    # Model in evaluation mode
    enc_model.eval()

    for raw_data, raw_target in test_loader:
        # Encryption
        data = raw_data[0].tolist()
        enc_data = ts.ckks_vector(context, data)
        enc_data_wrapper = CKKSWrapper(torch.rand(len(data)), enc_data)

        # Encrypted evaluation
        enc_output = enc_model.forward(enc_data_wrapper)

        # Decryption using client secret key
        output = enc_output.do_decryption()

        # Compute loss
        target = raw_target[0]
        loss = criterion.forward(output, target)
        test_loss += loss.item()

        print(f"Current Test Loss: {loss.item():.6f}")

    # Calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f"\nAverage Test Loss: {test_loss:.6f}")


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

    # Create the samplers
    sampler = RandomSampler(test_dataset, num_samples=20)

    # Set the batch size
    batch_size = 1

    # Create the data loader
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, sampler=sampler, worker_init_fn=seed_worker
    )

    # Get the number of features
    n_features = dataset.features.shape[1]

    # Load the model
    enc_model = EncLogisticRegression(n_features)
    enc_model.load_state_dict(
        torch.load(
            "./parameters/framingham/model-enc.pth"
        )
    )

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

    # Loss function
    criterion = torch.nn.BCELoss()

    # Test the model
    enc_test(context, enc_model, test_loader, criterion)
