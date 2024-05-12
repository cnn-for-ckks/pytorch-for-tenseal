from torchseal.nn import Linear as EncryptedLinear
from torch.nn import Linear as PlainLinear

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_linear_train():
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

    # Declare parameters
    out_features = 8
    in_features = 8

    # Declare input dimensions
    batch_size = 1

    # Create weight and bias
    weight = torch.randn(out_features, in_features, requires_grad=True)
    bias = torch.randn(out_features, requires_grad=True)

    # Create the input tensor
    input_tensor = torch.randn(batch_size, in_features, requires_grad=True)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor.clone().view(batch_size, -1)
    )

    # Create the plaintext linear layer
    linear = PlainLinear(in_features, out_features)
    linear.weight = torch.nn.Parameter(weight)
    linear.bias = torch.nn.Parameter(bias)

    # Create the encrypted linear layer
    enc_linear = EncryptedLinear(
        in_features=in_features,
        out_features=out_features,
        weight=torch.nn.Parameter(
            weight.clone().detach().requires_grad_(True)
        ),
        bias=torch.nn.Parameter(
            bias.clone().detach().requires_grad_(True)
        )
    )

    # Set both layer on training mode
    linear.train()
    enc_linear.train()

    # Calculate the output
    output = linear.forward(input_tensor)
    enc_output = enc_linear.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.do_decryption()

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Linear layer failed!"

    # Create random grad_output
    grad_output = torch.randn_like(output)

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(grad_output)

    # TODO: Check the correctness of input gradients

    # Check the correctness of weight gradients (with a tolerance of 5e-2)
    assert enc_linear.weight.grad is not None and linear.weight.grad is not None, "Weight gradients are None!"

    assert torch.allclose(
        enc_linear.weight.grad, linear.weight.grad, atol=5e-2, rtol=0
    ), "Weight gradient failed!"

    # Check the correctness of bias gradients (with a tolerance of 5e-2)
    assert enc_linear.bias.grad is not None and linear.bias.grad is not None, "Bias gradients are None!"

    assert torch.allclose(
        enc_linear.bias.grad, linear.bias.grad, atol=5e-2, rtol=0
    ), "Bias gradient failed!"


def test_linear_eval():
    # TODO: Add evaluation test

    pass
