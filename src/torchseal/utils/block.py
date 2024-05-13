import torch


def create_block_diagonal_matrix(base_matrix: torch.Tensor, num_blocks: int) -> torch.Tensor:
    # Get dimensions of the base matrix
    rows, cols = base_matrix.shape

    # Create a large zero matrix to hold the entire block diagonal matrix
    # NOTE: Possibility of TenSEAL parameter mismatch when using torch.zeros
    block_diag_matrix = torch.zeros(num_blocks * rows, num_blocks * cols)

    # Fill the block diagonal matrix
    for i in range(num_blocks):
        start_index = i * rows
        block_diag_matrix[
            start_index:start_index + rows, i * cols:i * cols + cols
        ] = base_matrix

    return block_diag_matrix