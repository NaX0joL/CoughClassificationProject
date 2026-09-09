import numpy as np

from core.data_pipeline_3.example_constructor._utils.series_segmenter import (
    SlidingWindowSegmenter,
)



def test_segmenter_drops_incomplete_last_segment() -> None:
    series = np.arange(10)

    segments = SlidingWindowSegmenter(
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

    segments = SlidingWindowSegmenter(
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

    segments = SlidingWindowSegmenter(
        window_size=4,
        stride=4,
        drop_last=False,
    ).segment(series)

    assert [segment.original_index for segment in segments] == [
        (0, 4),
        (4, 8),
    ]


def test_segmenter_pads_final_window_when_length_is_multiple_of_stride() -> None:
    series = np.arange(12)

    segments = SlidingWindowSegmenter(
        window_size=5,
        stride=4,
        drop_last=False,
    ).segment(series)

    assert [segment.original_index for segment in segments] == [
        (0, 5),
        (4, 9),
        (8, 13),
    ]
    np.testing.assert_array_equal(
        segments[-1].value,
        np.asarray([8, 9, 10, 11, 0]),
    )


def test_segmenter_pads_series_shorter_than_window() -> None:
    series = np.arange(3)

    segments = SlidingWindowSegmenter(
        window_size=5,
        stride=3,
        drop_last=False,
    ).segment(series)

    assert [segment.original_index for segment in segments] == [(0, 5)]
    np.testing.assert_array_equal(
        segments[0].value,
        np.asarray([0, 1, 2, 0, 0]),
    )
