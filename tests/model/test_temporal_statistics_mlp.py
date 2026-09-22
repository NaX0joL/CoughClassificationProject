import pytest
import torch
from torch import nn

from core.model.architectures.TemporalStatisticsMLP import TemporalStatisticsMLP


def test_temporal_statistics_mlp_accepts_frames_and_coefficients() -> None:
    model = TemporalStatisticsMLP(
        linear_dims=[16, 8],
        dropout=0.2,
        output_dim=3,
    )

    outputs = model(torch.rand(2, 51, 40))

    assert outputs.shape == (2, 3)
    assert model.layers[0].in_features == 80


@pytest.mark.parametrize(
    ("dropout", "output_dim"),
    [(0.0, 1), (0.4, 2)],
)
def test_temporal_statistics_mlp_configures_dropout_and_output_shape(
    dropout:float,
    output_dim:int,
) -> None:
    model = TemporalStatisticsMLP(
        linear_dims=[16],
        dropout=dropout,
        output_dim=output_dim,
    )

    outputs = model(torch.rand(2, 51, 40))
    dropouts = [layer for layer in model.layers if isinstance(layer, nn.Dropout)]

    assert outputs.shape == (2, output_dim)
    assert len(dropouts) == 1
    assert dropouts[0].p == dropout


def test_temporal_statistics_mlp_rejects_empty_linear_dims() -> None:
    with pytest.raises(ValueError, match="linear_dims"):
        TemporalStatisticsMLP(
            linear_dims=[],
            dropout=0.2,
            output_dim=2,
        )
