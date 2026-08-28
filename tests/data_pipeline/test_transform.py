import numpy as np
import pytest

from core.data_pipeline.intermediary import Example
from core.data_pipeline.preprocessing import (
    FeatureWiseNormalization,
    FeatureWiseStandardization,
    LogMelSpectrogram,
    MFCC,
)


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


def test_feature_wise_standardization_standardizes_each_feature() -> None:
    example = Example(
        value=np.array([
            [1.0, 10.0, 4.0],
            [2.0, 30.0, 4.0],
            [3.0, 50.0, 4.0],
        ]),
        label=1,
        metadata={"patient_id": "patient-1"},
    )

    transformed_example = FeatureWiseStandardization().transform([example])[0]

    assert transformed_example.value.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    assert transformed_example.value.std(axis=0) == pytest.approx([1.0, 1.0, 0.0])
    assert transformed_example.label == example.label
    assert transformed_example.metadata == example.metadata


def test_feature_wise_normalization_normalizes_each_feature() -> None:
    example = Example(
        value=np.array([
            [1.0, 10.0, 4.0],
            [2.0, 30.0, 4.0],
            [3.0, 50.0, 4.0],
        ]),
        label=1,
        metadata={"patient_id": "patient-1"},
    )

    transformed_example = FeatureWiseNormalization().transform([example])[0]

    np.testing.assert_allclose(
        transformed_example.value,
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    assert transformed_example.label == example.label
    assert transformed_example.metadata == example.metadata


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (
            FeatureWiseStandardization(),
            "FeatureWiseStandardization requires at least one value",
        ),
        (
            FeatureWiseNormalization(),
            "FeatureWiseNormalization requires at least one value",
        ),
    ],
)
def test_feature_scalers_reject_empty_examples(transformer, message:str) -> None:
    example = Example(value=np.array([]), label=0, metadata={})

    with pytest.raises(ValueError, match=message):
        transformer.transform([example])


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
