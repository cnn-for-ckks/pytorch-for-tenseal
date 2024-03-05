from torch.utils.data import DataLoader, RandomSampler, random_split
from torchseal.utils import seed_worker
from dataloader import FraminghamDataset
from logreg import LogisticRegression, train, test
from enc_logreg import EncLogisticRegression, enc_train, enc_test

import torch
import tenseal as ts
import numpy as np
import random

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
    dataset = FraminghamDataset(csv_file="./data/framingham.csv")

    # Split the data into training and testing
    generator = torch.Generator().manual_seed(73)
    train_dataset, test_dataset = random_split(
        dataset, [0.9, 0.1], generator=generator
    )

    # Create the samplers
    train_sampler = RandomSampler(
        train_dataset, num_samples=10, generator=generator)
    test_sampler = RandomSampler(
        test_dataset, num_samples=10, generator=generator)

    # Set the batch size
    batch_size = 1

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler, worker_init_fn=seed_worker
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

    # Create the original model
    original_model = LogisticRegression(n_features)
    original_model.load_state_dict(
        torch.load(
            "./parameters/framingham/original-model.pth"
        )
    )

    # Create the model, criterion, and optimizer
    enc_model = EncLogisticRegression(n_features, torch_nn=original_model)
    enc_optim = torch.optim.SGD(enc_model.parameters(), lr=0.1)
    enc_criterion = torch.nn.BCELoss()

    # Print the weights and biases of the model
    print("Ciphertext Model (Before Training):")
    print("\n".join(list(map(str, enc_model.parameters()))))
    print()

    # Train the model
    enc_model = enc_train(
        context,
        enc_model,
        train_loader,
        enc_criterion,
        enc_optim,
        n_epochs=10
    )
    print()

    # Print the weights and biases of the model
    print("Ciphertext Model (After Training):")
    print("\n".join(list(map(str, enc_model.parameters()))))
    print()

    # Test the model
    enc_test(context, enc_model, test_loader, enc_criterion)

    # Save the model
    torch.save(
        enc_model.state_dict(),
        "./parameters/framingham/enc-trained-model.pth"
    )
