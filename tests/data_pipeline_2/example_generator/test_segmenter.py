import numpy as np
import pytest

from core.data_pipeline_2 import (
    CenteredCoughSegmenter,
    SlidingWindowSegmenter,
    SourceSeries,
)


def make_source(
    values:list[float],
    annotations:object=None,
    include_annotations:bool=True,
) -> SourceSeries:
    metadata:dict[str, object] = {"is_infectious": False}
    if include_annotations:
        metadata["detected_cough_segments"] = annotations

    return SourceSeries(
        value=np.asarray(values, dtype=np.float32),
        metadata=metadata,
    )


def test_sliding_windows_preserve_stride_and_share_source_memory() -> None:
    source = make_source(list(range(6)), annotations=[])
    segmenter = SlidingWindowSegmenter(window_size=4, stride=2)

    segments = segmenter.segment([source])

    assert [segment.original_index for segment in segments] == [(0, 4), (2, 6)]
    np.testing.assert_array_equal(segments[1].value, [2, 3, 4, 5])
    assert np.shares_memory(segments[0].value, source.value)


def test_sliding_windows_can_keep_short_trailing_segment() -> None:
    source = make_source(list(range(5)), annotations=[])
    segmenter = SlidingWindowSegmenter(
        window_size=4,
        stride=4,
        keep_short_segments=True,
    )

    segments = segmenter.segment([source])

    assert len(segments) == 2
    assert segments[1].original_index == (4, 8)
    np.testing.assert_array_equal(segments[1].value, [4])


def test_centered_segmenter_makes_one_overlapping_window_per_annotation() -> None:
    source = make_source(
        list(range(7)),
        annotations=[(2, 2), (3, 3)],
    )
    segmenter = CenteredCoughSegmenter(window_size=4)

    segments = segmenter.segment([source])

    assert len(segments) == 2
    np.testing.assert_array_equal(segments[0].value, [1, 2, 3, 4])
    np.testing.assert_array_equal(segments[1].value, [2, 3, 4, 5])
    assert segments[0].cough_annotations == [(2, 2)]
    assert segments[1].cough_annotations == [(3, 3)]
    assert np.shares_memory(segments[0].value, source.value)


def test_centered_segmenter_cuts_long_cough_around_its_midpoint() -> None:
    source = make_source(list(range(9)), annotations=[(1, 7)])
    segmenter = CenteredCoughSegmenter(window_size=3)

    segments = segmenter.segment([source])

    np.testing.assert_array_equal(segments[0].value, [3, 4, 5])


@pytest.mark.parametrize(
    ("annotation", "expected", "window_start", "window_end"),
    [
        ((0, 0), [0, 0, 1, 2], -1, 3),
        ((4, 4), [3, 4, 0, 0], 3, 7),
    ],
)
def test_centered_segmenter_zero_pads_audio_boundaries(
    annotation:tuple[int, int],
    expected:list[int],
    window_start:int,
    window_end:int,
) -> None:
    source = make_source(list(range(5)), annotations=[annotation])
    segmenter = CenteredCoughSegmenter(window_size=4)

    segment = segmenter.segment([source])[0]

    np.testing.assert_array_equal(segment.value, expected)
    assert len(segment.value) == 4
    assert segment.original_index == (window_start, window_end)
    assert not np.shares_memory(segment.value, source.value)


def test_centered_segmenter_returns_no_segments_without_annotations() -> None:
    source = make_source(
        list(range(5)),
        include_annotations=False,
    )

    segments = CenteredCoughSegmenter(window_size=3).segment([source])

    assert segments == []


def test_centered_segmenter_returns_no_segments_for_empty_annotations() -> None:
    source = make_source(list(range(5)), annotations=[])

    segments = CenteredCoughSegmenter(window_size=3).segment([source])

    assert segments == []


def test_sliding_segmenter_validates_annotations_when_no_window_is_produced() -> None:
    source = make_source([0, 1], annotations=None)
    segmenter = SlidingWindowSegmenter(window_size=4, stride=4)

    with pytest.raises(ValueError, match="detected_cough_segments"):
        segmenter.segment([source])


@pytest.mark.parametrize(
    "annotations",
    [
        None,
        [[0, 1]],
        [(-1, 1)],
        [(2, 1)],
        [(0, 5)],
    ],
)
def test_centered_segmenter_rejects_malformed_or_invalid_annotations(
    annotations:object,
) -> None:
    source = make_source(list(range(5)), annotations=annotations)

    with pytest.raises(ValueError, match="detected_cough_segments"):
        CenteredCoughSegmenter(window_size=3).segment([source])


@pytest.mark.parametrize(
    ("window_size", "stride", "message"),
    [
        (0, 1, "window_size"),
        (1, 0, "stride"),
    ],
)
def test_sliding_segmenter_rejects_non_positive_configuration(
    window_size:int,
    stride:int,
    message:str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SlidingWindowSegmenter(window_size=window_size, stride=stride)


def test_centered_segmenter_rejects_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        CenteredCoughSegmenter(window_size=0)
