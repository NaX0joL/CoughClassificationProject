import numpy as np
import pytest

from core.data_pipeline_2 import (
    DownSampler,
    Example,
    FeatureWiseNormalization,
    FeatureWiseStandardization,
    LogMelSpectrogram,
    MFCC,
)
from core.data_pipeline_2.abstract import Transformer


def make_example(value:np.ndarray) -> Example:
    return Example(
        value=value,
        label=2,
        metadata={"patient_id": "patient-1"},
    )


@pytest.mark.parametrize(
    ("transformer", "number_of_features"),
    [
        (LogMelSpectrogram(n_mels=24), 24),
        (MFCC(n_mels=24, n_mfcc=13), 13),
    ],
)
def test_audio_feature_transformer_preserves_example_data(
    transformer:Transformer,
    number_of_features:int,
) -> None:
    example = make_example(np.ones(1_600, dtype=np.float32))

    transformed_example = transformer.transform([example])[0]

    assert transformed_example.value.ndim == 2
    assert transformed_example.value.shape[1] == number_of_features
    assert np.isfinite(transformed_example.value).all()
    assert transformed_example.label == example.label
    assert transformed_example.metadata is example.metadata


def test_feature_wise_standardization_standardizes_each_feature() -> None:
    example = make_example(np.array([
        [1.0, 10.0, 4.0],
        [2.0, 30.0, 4.0],
        [3.0, 50.0, 4.0],
    ]))

    transformed_example = FeatureWiseStandardization().transform([example])[0]

    assert transformed_example.value.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    assert transformed_example.value.std(axis=0) == pytest.approx([1.0, 1.0, 0.0])
    assert transformed_example.label == example.label
    assert transformed_example.metadata is example.metadata


def test_feature_wise_normalization_normalizes_each_feature() -> None:
    example = make_example(np.array([
        [1.0, 10.0, 4.0],
        [2.0, 30.0, 4.0],
        [3.0, 50.0, 4.0],
    ]))

    transformed_example = FeatureWiseNormalization().transform([example])[0]

    np.testing.assert_allclose(
        transformed_example.value,
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    assert transformed_example.metadata is example.metadata


def test_downsampler_resamples_waveform_and_preserves_example_data() -> None:
    example = make_example(np.ones(16_000, dtype=np.float32))
    transformer = DownSampler(
        original_sampling_rate=16_000,
        target_sampling_rate=8_000,
    )

    transformed_example = transformer.transform([example])[0]

    assert len(transformed_example.value) == 8_000
    assert transformed_example.label == example.label
    assert transformed_example.metadata is example.metadata


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (MFCC(), "MFCC requires at least one audio sample"),
        (
            LogMelSpectrogram(),
            "LogMelSpectrogram requires at least one audio sample",
        ),
        (
            FeatureWiseStandardization(),
            "FeatureWiseStandardization requires at least one value",
        ),
        (
            FeatureWiseNormalization(),
            "FeatureWiseNormalization requires at least one value",
        ),
        (
            DownSampler(16_000, 8_000),
            "DownSampler requires a non-empty audio array.",
        ),
    ],
)
def test_transformers_reject_empty_examples(
    transformer:Transformer,
    message:str,
) -> None:
    example = make_example(np.array([], dtype=np.float32))

    with pytest.raises(ValueError, match=message):
        transformer.transform([example])


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (
            lambda:LogMelSpectrogram(n_fft=200, win_length=400),
            "win_length cannot exceed n_fft",
        ),
        (
            lambda:MFCC(n_mels=12, n_mfcc=13),
            "n_mfcc cannot exceed n_mels",
        ),
        (lambda:LogMelSpectrogram(log_offset=0), "log_offset must be positive"),
        (lambda:DownSampler(0, 8_000), "Sample rates must be positive integers."),
    ],
)
def test_transformers_reject_invalid_parameters(
    transformer:object,
    message:str,
) -> None:
    with pytest.raises(ValueError, match=message):
        transformer()
