import torch


def create_transformation_matrix(n: int, length: int):
    # Calculate the number of groups
    num_groups = length // n

    # Create the transformation matrix
    transformation_matrix = torch.zeros((length, length))

    for i in range(num_groups):
        for j in range(n):
            row = i * n + j
            group_start = i * n
            transformation_matrix[row, group_start:group_start + n] = 1

    return transformation_matrix


x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
transformation_matrix = create_transformation_matrix(n=4, length=8)
print("transformation_matrix:\n", transformation_matrix)

# Apply the transformation matrix to vector x
transformed_x = torch.matmul(transformation_matrix, x)
print("transformed_x:\n", transformed_x)
