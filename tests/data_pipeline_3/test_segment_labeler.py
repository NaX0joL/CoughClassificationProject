import numpy as np

from core.data_pipeline_3.example_constructor._utils.segment_labeler import (
    OverlapLabeler,
)
from core.data_pipeline_3.intermediary import SeriesSegment



def _make_segment(start:int, end:int) -> SeriesSegment:
    return SeriesSegment(
        value=np.zeros(end - start, dtype=np.float32),
        original_index=(start, end),
    )


def test_no_overlap_uses_no_overlap_label() -> None:
    labeler = OverlapLabeler(overlap_threshold=0.5, no_overlap_label=0)

    labels = labeler.label(
        segments=[_make_segment(0, 10)],
        cough_intervals=[(10, 20)],
        source_label=2,
    )

    assert labels == [0]


def test_overlap_below_threshold_uses_no_overlap_label() -> None:
    labeler = OverlapLabeler(overlap_threshold=0.5, no_overlap_label=0)

    labels = labeler.label(
        segments=[_make_segment(0, 10)],
        cough_intervals=[(7, 12)],
        source_label=2,
    )

    assert labels == [0]


def test_overlap_at_threshold_uses_source_label() -> None:
    labeler = OverlapLabeler(overlap_threshold=0.5, no_overlap_label=0)

    labels = labeler.label(
        segments=[_make_segment(0, 10)],
        cough_intervals=[(5, 15)],
        source_label=2,
    )

    assert labels == [2]


def test_each_segment_is_labeled_independently() -> None:
    labeler = OverlapLabeler(overlap_threshold=0.5, no_overlap_label=0)

    labels = labeler.label(
        segments=[
            _make_segment(0, 10),
            _make_segment(10, 20),
            _make_segment(20, 30),
        ],
        cough_intervals=[(4, 16)],
        source_label=1,
    )

    assert labels == [1, 1, 0]
