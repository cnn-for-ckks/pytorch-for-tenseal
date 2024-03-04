import torch


if __name__ == "__main__":
    predictions = torch.tensor(
        [0.5, 0.9, 0.5, 0.9], dtype=torch.float32, requires_grad=True)
    matrix = torch.tensor(
        [
            [0.5, 0.197, -0.004, 0],
            [0.5, 0.197, -0.004, 0],
            [0.5, 0.197, -0.004, 0]
        ],
        dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([0, 1, 0], dtype=torch.float32, requires_grad=True)

    criterion = torch.nn.BCELoss()

    output = matrix.matmul(predictions)
    loss = criterion.forward(output, targets)

    loss.backward()

    print(f"Loss: {loss}")
    print(f"Predictions grad:\n{matrix.grad}")
