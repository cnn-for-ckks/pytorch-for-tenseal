from torchseal.nn import AvgPool2d as EncryptedAvgPool2d
from torch.nn import AvgPool2d as PlainAvgPool2d

import typing
import torch
import numpy as np
import random
import tenseal as ts
import torchseal
from torchseal.wrapper.ckks import CKKSWrapper


def test_avgpool2d():
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

    # Set the context
    torchseal.set_context(context)

    # Declare parameters
    # NOTE: High number of padding will cause the test to fail (due to the added noise by generating near-zero values)
    n_channels = 2
    kernel_height = 3
    kernel_width = 3
    stride = 1
    padding = 0

    # Declare input dimensions
    batch_size = 1
    input_height = 4
    input_width = 4

    # Create the input tensor
    input_tensor = torch.randn(
        batch_size, n_channels, input_height, input_width, requires_grad=True
    )

    print()
    print("Input tensor:", input_tensor)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor.view(batch_size, -1), do_encryption=True
    )
    enc_input_tensor.requires_grad_(True)

    # Create the plaintext average pooling layer
    plain_avgpool2d = PlainAvgPool2d(
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding
    )

    # Create the encrypted average pooling layer
    enc_avgpool2d = EncryptedAvgPool2d(
        n_channels=n_channels,
        input_size=(input_height, input_width),
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding
    )

    # Calculate the output
    output = plain_avgpool2d.forward(input_tensor)
    enc_output = enc_avgpool2d.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt().plaintext_data.clone()

    # Reshape the output
    output_resized = output.view(
        batch_size, -1,
    )

    print()
    print("Encrypted output:", dec_output)
    print("Plaintext output:", output_resized)
    print(f"Max difference: {torch.max(torch.abs(dec_output - output_resized)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(dec_output - output_resized)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(dec_output - output_resized)).item():.4f}")

    # Compare the results (with a tolerance of 5e-2)
    assert torch.allclose(
        dec_output,
        output_resized,
        atol=5e-2,
        rtol=0
    ), "Average pooling layer failed!"

    # Create random grad_output
    grad_output = torch.randn_like(output)

    # Encrypt the grad_output
    enc_grad_output = torchseal.ckks_wrapper(
        grad_output.clone().view(batch_size, -1), do_encryption=True
    )

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # Check the correctness of input gradients (with a tolerance of 5e-2)
    assert enc_input_tensor.grad is not None and input_tensor.grad is not None, "Input gradients are None!"

    # Reshape the input gradients
    input_grad_expanded = input_tensor.grad.view(
        batch_size, -1
    )

    # Decrypt the input gradients
    enc_input_grad = typing.cast(
        CKKSWrapper, enc_input_tensor.grad
    ).decrypt().plaintext_data.clone()

    print()
    print("Encrypted input gradient:", enc_input_grad)
    print("Plaintext input gradient:", input_grad_expanded)
    print(f"Max difference: {torch.max(torch.abs(enc_input_grad - input_grad_expanded)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(enc_input_grad - input_grad_expanded)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(enc_input_grad - input_grad_expanded)).item():.4f}")

    assert torch.allclose(
        enc_input_grad,
        input_grad_expanded,
        atol=5e-2,
        rtol=0
    ), "Input gradient failed!"
