import numpy as np

from core.data_pipeline_3.example_constructor._utils.segment_transformer import (
    FeatureWiseNormalization,
    FeatureWiseStandardization,
    LogMelSpectrogram,
    MFCC,
)
from core.data_pipeline_3.intermediary import SeriesSegment



def _make_segment(value:np.ndarray) -> SeriesSegment:
    return SeriesSegment(
        value=value,
        original_index=(10, 20),
    )


def test_feature_wise_standardization_preserves_original_index() -> None:
    segment = _make_segment(np.asarray([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ]))

    transformed = FeatureWiseStandardization().transform([segment])[0]

    np.testing.assert_allclose(
        transformed.value.mean(axis=0),
        np.zeros(2),
        atol=1e-7,
    )
    assert transformed.original_index == segment.original_index


def test_feature_wise_normalization_scales_each_feature() -> None:
    segment = _make_segment(np.asarray([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ]))

    transformed = FeatureWiseNormalization().transform([segment])[0]

    np.testing.assert_allclose(
        transformed.value,
        np.asarray([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]),
    )


def test_mfcc_transforms_waveform_into_feature_sequence() -> None:
    segment = _make_segment(np.ones(1_600, dtype=np.float32))
    transformer = MFCC(
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=16,
        n_mfcc=8,
    )

    transformed = transformer.transform([segment])[0]

    assert transformed.value.ndim == 2
    assert transformed.value.shape[1] == 8


def test_log_mel_transforms_waveform_into_feature_sequence() -> None:
    segment = _make_segment(np.ones(1_600, dtype=np.float32))
    transformer = LogMelSpectrogram(
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=16,
    )

    transformed = transformer.transform([segment])[0]

    assert transformed.value.ndim == 2
    assert transformed.value.shape[1] == 16
