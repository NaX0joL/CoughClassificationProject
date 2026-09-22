import numpy as np

from ..abstract import Segmenter
from ..intermediary import Segment, SourceSeries
from ._annotation import get_cough_annotations



class SlidingWindowSegmenter(Segmenter):

    def __init__(
        self,
        window_size:int,
        stride:int,
        keep_short_segments:bool=False,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if stride <= 0:
            raise ValueError("stride must be greater than 0")

        self.window_size = window_size
        self.stride = stride
        self.keep_short_segments = keep_short_segments
        return

    def segment(self, source_series:list[SourceSeries]) -> list[Segment]:
        segments:list[Segment] = []

        for series in source_series:
            cough_annotations = get_cough_annotations(series)
            segments.extend(self._segment_source_series(
                source_series=series,
                cough_annotations=cough_annotations,
            ))

        return segments

    def _segment_source_series(
        self,
        source_series:SourceSeries,
        cough_annotations:list[tuple[int, int]],
    ) -> list[Segment]:
        segments:list[Segment] = []

        for window_start in range(0, len(source_series.value), self.stride):
            window_end = window_start + self.window_size
            segment_value = source_series.value[window_start:window_end]

            if len(segment_value) < self.window_size and not self.keep_short_segments:
                continue

            segments.append(Segment(
                value=segment_value,
                source_series=source_series,
                original_index=(window_start, window_end),
                cough_annotations=cough_annotations,
            ))

        return segments



class CenteredCoughSegmenter(Segmenter):

    def __init__(self, window_size:int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        self.window_size = window_size
        return

    def segment(self, source_series:list[SourceSeries]) -> list[Segment]:
        segments:list[Segment] = []

        for series in source_series:
            cough_annotations = get_cough_annotations(series)
            for annotation in cough_annotations:
                segments.append(self._segment_annotation(series, annotation))

        return segments

    def _segment_annotation(
        self,
        source_series:SourceSeries,
        annotation:tuple[int, int],
    ) -> Segment:
        annotation_start, annotation_end = annotation
        annotation_midpoint = (annotation_start + annotation_end) // 2

        samples_before_midpoint = (self.window_size - 1) // 2
        window_start = annotation_midpoint - samples_before_midpoint
        window_end = window_start + self.window_size

        source_start = max(window_start, 0)
        source_end = min(window_end, len(source_series.value))
        segment_value = source_series.value[source_start:source_end]

        left_padding = max(0, -window_start)
        right_padding = max(0, window_end - len(source_series.value))
        if left_padding or right_padding:
            segment_value = np.pad(
                segment_value,
                (left_padding, right_padding),
                mode="constant",
                constant_values=0,
            )

        return Segment(
            value=segment_value,
            source_series=source_series,
            original_index=(window_start, window_end),
            cough_annotations=[annotation],
        )
