from torchseal.nn import Linear as EncryptedLinear
from torchseal.optim import SGD as EncryptedSGD
from torch.nn import Linear as PlainLinear
from torch.optim import SGD as PlainSGD

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_sgd():
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
        input_tensor.clone().view(batch_size, -1), do_encryption=True
    )

    # Create the plaintext linear layer
    linear = PlainLinear(in_features, out_features)
    linear.weight = torch.nn.Parameter(weight)
    linear.bias = torch.nn.Parameter(bias)

    # Create the encrypted linear layer
    enc_linear = EncryptedLinear(
        in_features=in_features,
        out_features=out_features,
        weight=torchseal.ckks_wrapper(
            weight.clone(), do_encryption=False
        ),
        bias=torchseal.ckks_wrapper(
            bias.clone(), do_encryption=False
        )
    )

    # Create the optimizer
    optimizer = PlainSGD(linear.parameters(), lr=0.1)
    enc_optimizer = EncryptedSGD(enc_linear.parameters(), lr=0.1)

    # Set both layer on training mode
    linear.train()
    enc_linear.train()

    # Calculate the output
    output = linear.forward(input_tensor)
    enc_output = enc_linear.forward(enc_input_tensor)

    # Create random grad_output
    grad_output = torch.randn_like(output)
    enc_grad_output = torchseal.ckks_wrapper(
        grad_output.clone().view(batch_size, -1), do_encryption=True
    )

    # Set the gradients to none
    optimizer.zero_grad()
    enc_optimizer.zero_grad()

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # Update the weights
    optimizer.step()
    enc_optimizer.step()

    # Compare the weights
    assert torch.allclose(
        enc_linear.weight.decrypt().plaintext_data.clone(),
        linear.weight,
        atol=5e-2,
        rtol=0
    ), "Weights are not equal"

    # Compare the biases
    assert torch.allclose(
        enc_linear.bias.decrypt().plaintext_data.clone(),
        linear.bias,
        atol=5e-2,
        rtol=0
    ), "Biases are not equal"
