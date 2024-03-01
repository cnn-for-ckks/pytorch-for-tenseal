from typing import Tuple
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from tenseal import CKKSVector
from mnist_train import ConvNet
from mnist_train_enc import EncConvNet

import numpy as np
import torch
import tenseal as ts


def enc_test(context: ts.Context, enc_model: EncConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, kernel_shape: Tuple[int, int], stride: int):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # unpack the kernel shape
    kernel_shape_h, kernel_shape_w = kernel_shape

    for data, target in test_loader:
        # Encoding and encryption
        result: Tuple[CKKSVector, int] = ts.im2col_encoding(
            context,
            data.view(28, 28).tolist(),
            kernel_shape_h,
            kernel_shape_w,
            stride
        )  # type: ignore[error from tenseal library]

        # Unpack the result
        x_enc, windows_nb = result

        # Encrypted evaluation
        enc_output = enc_model.forward(x_enc, windows_nb)

        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion.forward(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f"Test Loss for Encrypted Data: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy for Encrypted Data (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)

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
    test_data = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Set the batch size
    batch_size = 64

    # Create the samplers
    sampler = RandomSampler(test_data, num_samples=50, replacement=True)

    # Create the data loaders
    test_loader = DataLoader(
        test_data, batch_size=batch_size, sampler=sampler
    )

    # Load the model
    model = ConvNet()
    model.load_state_dict(torch.load("./parameters/MNIST/model.pth"))

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Create the data loaders for encrypted evaluation
    enc_test_loader = DataLoader(test_data, batch_size=1, sampler=sampler)

    # Create the encrypted model
    enc_model = EncConvNet(model)

    # Encrypted evaluation
    enc_test(
        context,
        enc_model,
        enc_test_loader,
        criterion,
        (7, 7),
        3
    )
