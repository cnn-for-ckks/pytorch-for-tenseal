from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchseal.utils import seed_worker
from cnn import ConvNet, train, test
from enc_cnn import EncConvNet, enc_train, enc_test

import torch
import numpy as np
import random
import tenseal as ts

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
    train_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Subset the data
    subset_train_data = Subset(train_data, list(range(50)))
    subset_test_data = Subset(test_data, list(range(50)))

    # Set the batch size
    batch_size = 1

    # Create the data loaders
    subset_train_loader = DataLoader(
        subset_train_data, batch_size=batch_size, worker_init_fn=seed_worker
    )
    subset_test_loader = DataLoader(
        subset_test_data, batch_size=batch_size, worker_init_fn=seed_worker
    )

    # Create the model, criterion, and optimizer
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save the original model
    torch.save(
        model.state_dict(),
        "./parameters/MNIST/original-model.pth"
    )

    # Print the weights and biases of the model
    print("Plaintext Model (Before Training):")
    print("\n".join(list(map(str, model.parameters()))))
    print()

    # Train the model
    model = train(
        model, subset_train_loader, criterion, optimizer, n_epochs=10
    )
    print()

    # Print the weights and biases of the model
    print("Plaintext Model (After Training):")
    print("\n".join(list(map(str, model.parameters()))))
    print()

    # Test the model
    test(model, subset_test_loader, criterion)
    print()

    # Save the model
    torch.save(model.state_dict(), "./parameters/MNIST/trained-model.pth")

    # # Create the trained model
    # original_model = ConvNet()
    # original_model.load_state_dict(
    #     torch.load(
    #         "./parameters/MNIST/original-model.pth"
    #     )
    # )

    # # Set the batch size
    # enc_batch_size = 1  # NOTE: This is set to 1 to allow for encryption

    # # Create the encrypted data loaders
    # enc_subset_train_loader = DataLoader(
    #     subset_train_data, batch_size=enc_batch_size, worker_init_fn=seed_worker
    # )
    # enc_subset_test_loader = DataLoader(
    #     subset_test_data, batch_size=enc_batch_size, worker_init_fn=seed_worker
    # )

    # # Create the model, criterion, and optimizer
    # enc_model = EncConvNet(torch_nn=original_model)
    # enc_criterion = torch.nn.CrossEntropyLoss()
    # enc_optimizer = torch.optim.Adam(enc_model.parameters(), lr=0.001)

    # # Encrypted training
    # enc_model = enc_train(
    #     context,
    #     enc_model,
    #     enc_subset_train_loader,
    #     enc_criterion,
    #     enc_optimizer,
    #     kernel_shape=(7, 7),
    #     stride=3,
    #     padding=0,
    #     n_epochs=10
    # )
    # print()

    # # Encrypted evaluation
    # enc_test(
    #     context,
    #     enc_model,
    #     enc_subset_test_loader,
    #     enc_criterion,
    #     kernel_shape=(7, 7),
    #     stride=3,
    #     padding=0
    # )
