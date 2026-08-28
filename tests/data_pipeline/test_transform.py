import numpy as np
import pytest

from core.data_pipeline.intermediary import Example
from core.data_pipeline.preprocessing import LogMelSpectrogram, MFCC


def test_log_mel_spectogram_produces_log_mel_features_and_preserves_example_data() -> None:
    example = Example(
        value=np.ones(1_600, dtype=np.float32),
        label=3,
        metadata={"patient_id": "patient-1"},
    )

    transformed_example = LogMelSpectrogram(n_mels=24).transform([example])[0]

    assert transformed_example.value.ndim == 2
    assert transformed_example.value.shape[1] == 24
    assert np.isfinite(transformed_example.value).all()
    assert transformed_example.label == example.label
    assert transformed_example.metadata == example.metadata


def test_log_mel_spectogram_rejects_empty_waveforms() -> None:
    example = Example(value=np.array([], dtype=np.float32), label=0, metadata={})

    with pytest.raises(ValueError, match="LogMelSpectrogram requires at least one audio sample"):
        LogMelSpectrogram().transform([example])


def test_mfcc_uses_configured_feature_counts() -> None:
    example = Example(value=np.ones(1_600, dtype=np.float32), label=0, metadata={})

    transformed_example = MFCC(n_mels=24, n_mfcc=13).transform([example])[0]

    assert transformed_example.value.shape[1] == 13


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (lambda: LogMelSpectrogram(n_fft=200, win_length=400), "win_length cannot exceed n_fft"),
        (lambda: MFCC(n_mels=12, n_mfcc=13), "n_mfcc cannot exceed n_mels"),
        (lambda: LogMelSpectrogram(log_offset=0), "log_offset must be positive"),
    ],
)
def test_feature_transformers_reject_invalid_parameters(transformer, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        transformer()
