import torch
from torch import nn

from core.model.architectures.LeNet import LeNet


def test_lenet_applies_configured_dropout() -> None:
    model = LeNet(
        conv_channels=[4, 8],
        linear_dims=[16, 8],
        dropout=0.3,
        output_dim=2,
    )

    outputs = model(torch.rand(2, 42, 40))
    conv_dropouts = [
        layer
        for layer in model.conv_layers
        if isinstance(layer, nn.Dropout1d)
    ]
    linear_dropouts = [
        layer
        for layer in model.linear_layers
        if isinstance(layer, nn.Dropout)
    ]

    assert outputs.shape == (2, 2)
    assert len(conv_dropouts) == 2
    assert len(linear_dropouts) == 2
    assert all(layer.p == 0.3 for layer in conv_dropouts + linear_dropouts)
