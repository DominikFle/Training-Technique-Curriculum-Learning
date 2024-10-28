import torch.nn as nn

from models.ConvNet import ConvNet


def print_params(model: nn.Module):
    for name, param in model.named_parameters():
        print(name)


if __name__ == "__main__":
    model = ConvNet(
        input_size=(28, 28),
        num_classes=10,
        model_channels=[(1, 32), (32, 64)],
        strides=[1, 2],
        learning_rate=0.01,
        weight_decay=0.1,
    )
    print_params(model)
