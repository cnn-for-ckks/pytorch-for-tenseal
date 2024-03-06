from torch.nn.functional import unfold, fold

import tenseal as ts
import torch

if __name__ == "__main__":
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

    # Randomize the tensor
    weight = torch.rand(1, 7, 7)
    print("weight.shape:", weight.shape, end="\n\n")

    image = torch.rand(1, 28, 28)
    print("image.shape:", image.shape, end="\n\n")

    # NOTE: Plaintext im2col
    unfolded_image = unfold(
        image,
        kernel_size=(7, 7),
        stride=(7, 7),
        padding=(0, 0)
    )  # NOTE: Must not overlap
    print("unfolded_image.shape:", unfolded_image.shape, end="\n\n")

    # NOTE: Plaintext convolution
    # Create the convolutional weight
    conv_weight = weight.view(1, -1)

    # Perform the convolution
    conv_output = conv_weight.matmul(
        unfolded_image
    ).view(-1)

    # print the result
    print("conv_output:", conv_output.tolist(), end="\n\n")

    # NOTE: Encrypted im2col
    enc_unfolded_image = ts.enc_matmul_encoding(
        context, unfolded_image.t()
    )
    num_row = unfolded_image.shape[0]
    num_col = unfolded_image.shape[1]

    # NOTE: Encrypted convolution
    # Create the convolutional weight
    enc_conv_weight = ts.plain_tensor(weight.view(-1))

    # Perform the convolution
    enc_result = enc_unfolded_image.enc_matmul_plain(
        enc_conv_weight.tolist(), num_col
    )

    # Decrypt the result
    print("enc_result:", enc_result.decrypt(), end="\n\n")

    # NOTE: Plaintext col2im
    raw_dec_unfolded_image = enc_unfolded_image.decrypt()
    dec_unfolded_image = torch.tensor(raw_dec_unfolded_image).reshape(
        len(raw_dec_unfolded_image) // num_col, num_col
    )
    print("dec_unfolded_image.shape", dec_unfolded_image.shape, end="\n\n")

    # Throw away the extra rows
    dec_unfolded_image_clipped = dec_unfolded_image[:num_row, :]

    # Fold (Inverse operation)
    dec_image = fold(
        dec_unfolded_image_clipped,
        output_size=(28, 28),
        kernel_size=(7, 7),
        stride=(7, 7),
        padding=(0, 0)
    )  # NOTE: Must not overlap
    print("dec_image.shape:", dec_image.shape, end="\n\n")

    # Check if the original and folded tensors are equal
    error_sum = (image - dec_image).abs().sum()
    print("error_sum:", error_sum, end="\n\n")
