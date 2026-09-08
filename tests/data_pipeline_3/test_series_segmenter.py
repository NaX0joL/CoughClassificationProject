import numpy as np

from core.data_pipeline_3.example_constructor._utils.series_segmenter import (
    SeriesSegmenter,
)



def test_segmenter_drops_incomplete_last_segment() -> None:
    series = np.arange(10)

    segments = SeriesSegmenter(
        window_size=4,
        stride=4,
        drop_last=True,
    ).segment(series)

    assert [segment.original_index for segment in segments] == [
        (0, 4),
        (4, 8),
    ]


def test_segmenter_zero_pads_incomplete_last_segment() -> None:
    series = np.arange(10)

    segments = SeriesSegmenter(
        window_size=4,
        stride=4,
        drop_last=False,
    ).segment(series)

    assert [segment.original_index for segment in segments] == [
        (0, 4),
        (4, 8),
        (8, 12),
    ]
    np.testing.assert_array_equal(
        segments[-1].value,
        np.asarray([8, 9, 0, 0]),
    )


def test_segmenter_keeps_last_segment_when_it_fits_exactly() -> None:
    series = np.arange(8)

    segments = SeriesSegmenter(
        window_size=4,
        stride=4,
        drop_last=False,
    ).segment(series)

    assert [segment.original_index for segment in segments] == [
        (0, 4),
        (4, 8),
    ]
