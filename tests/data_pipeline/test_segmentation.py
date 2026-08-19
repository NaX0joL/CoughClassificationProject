import numpy as np

from core.data_pipeline.preprocessing.segmentation import SlidingWindowSegmenter
from core.data_pipeline.intermediary import ORIGINAL_LABEL_METADATA_KEY, SourceSeries


def _make_series(signal:list[float], label:int=0, metadata:dict|None=None) -> SourceSeries:
    return SourceSeries(
        value=np.array(signal, dtype=np.float32),
        label=label,
        metadata=metadata or {},
    )


class TestSlidingWindowSegmenterBasic:

    def test_single_window_fits_exactly(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=4)
        series = _make_series([1.0, 2.0, 3.0, 4.0])
        examples = segmenter.segment([series])

        assert len(examples) == 1
        np.testing.assert_array_equal(examples[0].value, [1.0, 2.0, 3.0, 4.0])

    def test_multiple_non_overlapping_windows(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=3, stride=3)
        series = _make_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        examples = segmenter.segment([series])

        assert len(examples) == 2
        np.testing.assert_array_equal(examples[0].value, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(examples[1].value, [4.0, 5.0, 6.0])

    def test_multiple_series(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=2, stride=2)
        series_a = _make_series([1.0, 2.0, 3.0, 4.0], label=0)
        series_b = _make_series([5.0, 6.0, 7.0, 8.0], label=1)
        examples = segmenter.segment([series_a, series_b])

        assert len(examples) == 4
        assert examples[2].label == 1


class TestSlidingWindowSegmenterOverlap:

    def test_overlap_when_stride_less_than_window(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=2)
        series = _make_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        examples = segmenter.segment([series])

        assert len(examples) == 2
        np.testing.assert_array_equal(examples[0].value, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(examples[1].value, [3.0, 4.0, 5.0, 6.0])


class TestSlidingWindowSegmenterShortSegments:

    def test_discard_short_trailing_by_default(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=4)
        series = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        examples = segmenter.segment([series])

        assert len(examples) == 1
        np.testing.assert_array_equal(examples[0].value, [1.0, 2.0, 3.0, 4.0])

    def test_keep_short_trailing_when_enabled(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=4, keep_short_segments=True)
        series = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        examples = segmenter.segment([series])

        assert len(examples) == 2
        np.testing.assert_array_equal(examples[1].value, [5.0])

    def test_discard_short_trailing_with_overlap(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=2)
        series = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        examples = segmenter.segment([series])

        assert len(examples) == 1
        np.testing.assert_array_equal(examples[0].value, [1.0, 2.0, 3.0, 4.0])



class TestSlidingWindowSegmenterMetadata:

    def test_original_label_preserved(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=2, stride=2)
        series = _make_series([1.0, 2.0], label=1)
        examples = segmenter.segment([series])

        assert examples[0].metadata[ORIGINAL_LABEL_METADATA_KEY] == "1"

    def test_original_label_from_metadata_takes_precedence(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=2, stride=2)
        series = _make_series(
            [1.0, 2.0],
            label=1,
            metadata={ORIGINAL_LABEL_METADATA_KEY: "POSITIVE"},
        )
        examples = segmenter.segment([series])

        assert examples[0].metadata[ORIGINAL_LABEL_METADATA_KEY] == "POSITIVE"

    def test_kept_metadata_forwarded(self) -> None:
        segmenter = SlidingWindowSegmenter(
            window_size=2,
            stride=2,
            kept_metadata_key=["patient_id"],
        )
        series = _make_series(
            [1.0, 2.0],
            metadata={"patient_id": "P001", "other": "ignored"},
        )
        examples = segmenter.segment([series])

        assert examples[0].metadata["patient_id"] == "P001"
        assert "other" not in examples[0].metadata

    def test_metadata_shared_across_windows(self) -> None:
        segmenter = SlidingWindowSegmenter(
            window_size=2,
            stride=2,
            kept_metadata_key=["patient_id"],
        )
        series = _make_series(
            [1.0, 2.0, 3.0, 4.0],
            metadata={"patient_id": "P001"},
        )
        examples = segmenter.segment([series])

        assert examples[0].metadata is examples[1].metadata


class TestSlidingWindowSegmenterEdgeCases:

    def test_empty_signal_returns_no_examples(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=2)
        series = _make_series([])
        examples = segmenter.segment([series])

        assert len(examples) == 0

    def test_empty_signal_with_keep_short(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=2, keep_short_segments=True)
        series = _make_series([])
        examples = segmenter.segment([series])

        assert len(examples) == 0

    def test_signal_shorter_than_window_discarded(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=4)
        series = _make_series([1.0, 2.0])
        examples = segmenter.segment([series])

        assert len(examples) == 0

    def test_signal_shorter_than_window_kept(self) -> None:
        segmenter = SlidingWindowSegmenter(window_size=4, stride=4, keep_short_segments=True)
        series = _make_series([1.0, 2.0])
        examples = segmenter.segment([series])

        assert len(examples) == 1
        np.testing.assert_array_equal(examples[0].value, [1.0, 2.0])
