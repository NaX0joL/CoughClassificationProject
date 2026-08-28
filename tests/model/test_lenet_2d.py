import torch
from torch import nn

from core.model.architectures.LeNet2D import LeNet2D


def test_lenet_2d_applies_convolutions_across_time_and_features() -> None:
    model = LeNet2D(
        conv_channels=[4, 8],
        linear_dims=[16, 8],
        dropout=0.3,
        output_dim=2,
    )

    outputs = model(torch.rand(2, 42, 40))
    conv_layers = [
        layer
        for layer in model.conv_layers
        if isinstance(layer, nn.Conv2d)
    ]
    conv_dropouts = [
        layer
        for layer in model.conv_layers
        if isinstance(layer, nn.Dropout2d)
    ]
    linear_dropouts = [
        layer
        for layer in model.linear_layers
        if isinstance(layer, nn.Dropout)
    ]

    assert outputs.shape == (2, 2)
    assert conv_layers[0].in_channels == 1
    assert len(conv_layers) == 2
    assert len(conv_dropouts) == 2
    assert len(linear_dropouts) == 2
    assert all(layer.p == 0.3 for layer in conv_dropouts + linear_dropouts)
