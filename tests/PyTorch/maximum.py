import torch

if __name__ == "__main__":
    image = torch.tensor(
        [[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6]
        ]],
        dtype=torch.float32, requires_grad=True
    )
    avg_layer = torch.nn.MaxPool2d(kernel_size=3, stride=1)
    output = avg_layer.forward(image)

    criterion = torch.nn.L1Loss()
    target = torch.tensor(
        [[
            [0, 0],
            [0, 0],
        ]],
        dtype=torch.float32, requires_grad=True
    )
    loss = criterion.forward(output, target)
    loss.backward()

    print(f"Target grad:\n{target.grad}")
    print(f"Image grad:\n{image.grad}")
