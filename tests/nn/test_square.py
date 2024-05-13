from torchseal.nn import Square as EncryptedSquare

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_square():
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

    # Declare input dimensions
    input_length = 10
    batch_size = 1

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor.view(batch_size, -1), do_encryption=True
    )

    # Create the encrypted square layer
    enc_square = EncryptedSquare()

    # Calculate the output
    output = input_tensor.square()
    enc_output = enc_square.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt()

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Square layer failed!"

    # TODO: Do backward pass and check the correctness of the input gradients
