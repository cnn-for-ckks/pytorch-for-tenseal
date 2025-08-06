from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.nn import Sigmoid as EncryptedSigmoid
from torch.nn import Sigmoid as PlainSigmoid

import typing
import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_sigmoid():
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
    start = -3
    stop = 3
    num_of_sample = 100
    degree = 3
    approximation_type = "least-squares"

    # Declare input dimensions
    input_length = 10
    batch_size = 1

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length, requires_grad=True)

    print()
    print("Input tensor:", input_tensor)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor, do_encryption=True
    )
    enc_input_tensor.requires_grad_(True)

    # Create the plaintext sigmoid layer
    sigmoid = PlainSigmoid()

    # Create the encrypted sigmoid layer
    enc_sigmoid = EncryptedSigmoid(
        start=start,
        stop=stop,
        num_of_sample=num_of_sample,
        degree=degree,
        approximation_type=approximation_type,
    )

    # Calculate the output
    output = sigmoid.forward(input_tensor)
    enc_output = enc_sigmoid.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt().plaintext_data.clone()

    print()
    print("Output:", output)
    print("Decrypted output:", dec_output)
    print(f"Max difference: {torch.max(torch.abs(output - dec_output)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(output - dec_output)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(output - dec_output)).item():.4f}")

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Sigmoid layer failed!"

    # Create random grad_output
    grad_output = torch.randn_like(output)

    # Encrypt the grad_output
    enc_grad_output = torchseal.ckks_wrapper(
        grad_output.clone(), do_encryption=True
    )

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # Check the correctness of input gradients (with a tolerance of 5e-2)
    assert enc_input_tensor.grad is not None and input_tensor.grad is not None, "Input gradients are None!"

    # Decrypt the input gradients
    dec_input_grad = typing.cast(
        CKKSWrapper,
        enc_input_tensor.grad
    ).decrypt().plaintext_data.clone()

    print()
    print("Encrypted input gradient:", dec_input_grad)
    print("Plaintext input gradient:", input_tensor.grad)
    print(f"Max difference: {torch.max(torch.abs(dec_input_grad - input_tensor.grad)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(dec_input_grad - input_tensor.grad)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(dec_input_grad - input_tensor.grad)).item():.4f}")

    assert torch.allclose(
        dec_input_grad,
        input_tensor.grad,
        atol=5e-2,
        rtol=0
    ), "Input gradients are incorrect!"
