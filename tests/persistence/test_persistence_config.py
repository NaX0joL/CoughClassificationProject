import pytest

from config_plan import log_mel_spectrogram_persistence_config, mfcc_persistence_config
from core.persistence import PersistenceConfig


def test_feature_persistence_plans_describe_their_feature_representation() -> None:
    assert mfcc_persistence_config.x_axis_label == "Frame"
    assert mfcc_persistence_config.y_axis_label == "MFCC coefficient"
    assert mfcc_persistence_config.colorbar_label == "MFCC value"
    assert log_mel_spectrogram_persistence_config.y_axis_label == "Mel band"
    assert log_mel_spectrogram_persistence_config.colorbar_label == "Log-mel energy"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"feature_color_percentile": 0}, "feature_color_percentile"),
        ({"feature_colormap": " "}, "feature_colormap"),
        ({"x_axis_label": " "}, "x_axis_label"),
        ({"y_axis_label": " "}, "y_axis_label"),
        ({"colorbar_label": ""}, "colorbar_label"),
    ],
)
def test_persistence_config_rejects_invalid_feature_figure_settings(arguments, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PersistenceConfig(**arguments)
