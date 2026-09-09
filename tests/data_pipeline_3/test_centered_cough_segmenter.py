from pathlib import Path
from unittest.mock import Mock

import numpy as np

from core.data_pipeline_3.example_constructor.example_constructor import (
    ExampleConstructor,
)
from core.data_pipeline_3.example_constructor._utils.segment_labeler import (
    OverlapLabeler,
)
from core.data_pipeline_3.example_constructor._utils.series_segmenter import (
    CenteredCoughSegmenter,
    SlidingWindowSegmenter,
)
from core.data_pipeline_3.intermediary import SourceRecord



def test_segmenter_returns_one_centered_window_per_cough_interval() -> None:
    series = np.arange(20)

    segments = CenteredCoughSegmenter(window_size=6).segment(
        series,
        cough_intervals=[(5, 7), (12, 12)],
    )

    assert [segment.original_index for segment in segments] == [
        (3, 9),
        (9, 15),
    ]
    np.testing.assert_array_equal(segments[0].value, np.arange(3, 9))
    np.testing.assert_array_equal(segments[1].value, np.arange(9, 15))


def test_segmenter_pads_centered_windows_at_recording_boundaries() -> None:
    series = np.arange(10, 20)

    segments = CenteredCoughSegmenter(window_size=5).segment(
        series,
        cough_intervals=[(0, 0), (9, 9)],
    )

    assert [segment.original_index for segment in segments] == [
        (-2, 3),
        (7, 12),
    ]
    np.testing.assert_array_equal(
        segments[0].value,
        np.asarray([0, 0, 10, 11, 12]),
    )
    np.testing.assert_array_equal(
        segments[1].value,
        np.asarray([17, 18, 19, 0, 0]),
    )


def test_constructor_returns_one_labeled_example_per_cough_interval(
    tmp_path:Path,
    monkeypatch,
) -> None:
    sliding_window_segmenter = SlidingWindowSegmenter(
        window_size=6,
        stride=3,
    )
    constructor = ExampleConstructor(
        sampling_rate=16_000,
        train_segmenter=sliding_window_segmenter,
        validation_segmenter=sliding_window_segmenter,
        test_segmenter=CenteredCoughSegmenter(window_size=6),
        segment_labeler=OverlapLabeler(
            overlap_threshold=0.7,
            no_overlap_label=0,
        ),
        cache_directory=tmp_path / "cache",
    )
    monkeypatch.setattr(
        constructor,
        "_load_audio",
        lambda path: np.arange(20),
    )
    constructor.raw_audio_visualizer = Mock()
    source_record = SourceRecord(
        metadata={
            "DetectedCoughSegments": [(5, 7), (12, 12)],
            "isInfectious": True,
        },
        audio_path=Path("audio.wav"),
        label=1,
    )

    examples = constructor.construct([source_record], type="test")

    assert len(examples) == 2
    assert [example.label for example in examples] == [1, 1]
    np.testing.assert_array_equal(examples[0].value, np.arange(3, 9))
    np.testing.assert_array_equal(examples[1].value, np.arange(9, 15))
