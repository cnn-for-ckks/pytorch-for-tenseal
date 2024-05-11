from torchseal.utils import create_block_diagonal_matrix

import torch


def create_inverse_padding_transformation_matrix(num_channels: int, input_height: int, input_width: int, padding: int) -> torch.Tensor:
    # Add padding to the input size
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Initialize the inverse transformation matrix with zeros
    # NOTE: Possibility of TenSEAL parameter mismatch when using torch.zeros
    inverse_transform_matrix = torch.zeros(
        input_height * input_width, padded_input_height * padded_input_width
    )

    # Fill the inverse transformation matrix to map each padded pixel back to the original image
    for i in range(input_height):
        for j in range(input_width):
            # Original position in the flattened image
            orig_pos = i * input_width + j

            # Corresponding position in the flattened padded image
            padded_pos = (i + padding) * padded_input_width + (j + padding)

            # Update the inverse transformation matrix
            inverse_transform_matrix[orig_pos, padded_pos] = 1

    # Repeat the inverse transformation matrix for each channel
    repeated_inverse_transform_matrix = create_block_diagonal_matrix(
        inverse_transform_matrix, num_channels
    )

    # Transpose the inverse transformation matrix
    transposed_inverse_transform_matrix = repeated_inverse_transform_matrix.t()

    return transposed_inverse_transform_matrix


def create_padding_transformation_matrix(num_channels: int, input_height: int, input_width: int, padding: int) -> torch.Tensor:
    # Add padding to the input size
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Initialize the transformation matrix with zeros
    # NOTE: Possibility of TenSEAL parameter mismatch when using torch.zeros
    transform_matrix = torch.zeros(
        padded_input_height * padded_input_width, input_height * input_width
    )

    # Fill the transformation matrix to map each original pixel to the correct position in the padded image
    for i in range(input_height):
        for j in range(input_width):
            # Original position in the flattened image
            orig_pos = i * input_width + j

            # New position in the flattened padded image
            new_pos = (i + padding) * padded_input_width + (j + padding)

            # Update the transformation matrix
            transform_matrix[new_pos, orig_pos] = 1

    # Repeat the transformation matrix for each channel
    repeated_transform_matrix = create_block_diagonal_matrix(
        transform_matrix, num_channels
    )

    # Transpose the transformation matrix
    transposed_transform_matrix = repeated_transform_matrix.t()

    return transposed_transform_matrix
