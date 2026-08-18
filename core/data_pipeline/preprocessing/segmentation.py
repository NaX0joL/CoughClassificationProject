
from typing import TypeGuard

from ..abstract import Segmenter
from ..intermediary import ORIGINAL_LABEL_METADATA_KEY, SourceSeries, Example



class CoughSegmenter(Segmenter):
    
    def __init__(self, kept_metadata_key:list[str]) -> None:
        self.kept_metadata_key = kept_metadata_key
        return
    
    def segment(self, source_series:list[SourceSeries]) -> list[Example]:
        examples = []
        
        for series in source_series:
            example = self._segment_by_detected_cough_segments(series)
            examples.extend(example)
            
        return examples
    
    def _segment_by_detected_cough_segments(self, source_series:SourceSeries) -> list[Example]:
        kept_metadata = {
            key: source_series.metadata[key]
            for key in self.kept_metadata_key
        }
        kept_metadata[ORIGINAL_LABEL_METADATA_KEY] = source_series.metadata.get(
            ORIGINAL_LABEL_METADATA_KEY,
            str(source_series.label),
        )
        examples = []
        
        cough_intervals = source_series.metadata.get("detected_cough_segments")
        if not self._is_cough_intervals(cough_intervals):
            raise ValueError(
                "detected_cough_segments must be a list of integer intervals"
            )

        for start, end in cough_intervals:
            example = Example(
                value=source_series.value[start:end + 1],
                label=source_series.label,
                metadata=kept_metadata,
            )
            examples.append(example)
        
        return examples

    @staticmethod
    def _is_cough_intervals(
        value: object,
    ) -> TypeGuard[list[tuple[int, int]]]:
        if not isinstance(value, list):
            return False

        return all(
            isinstance(interval, tuple)
            and len(interval) == 2
            and isinstance(interval[0], int)
            and isinstance(interval[1], int)
            for interval in value
        )



class SlidingWindowSegmenter(Segmenter):

    def __init__(
        self,
        window_size:int,
        stride:int,
        keep_short_segments:bool=False,
        kept_metadata_key:list[str]|None=None,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.keep_short_segments = keep_short_segments
        self.kept_metadata_key = kept_metadata_key if kept_metadata_key is not None else []
        return

    def segment(self, source_series:list[SourceSeries]) -> list[Example]:
        examples = []

        for series in source_series:
            examples.extend(self._segment_by_sliding_window(series))

        return examples

    def _segment_by_sliding_window(self, source_series:SourceSeries) -> list[Example]:
        kept_metadata = {
            key: source_series.metadata[key]
            for key in self.kept_metadata_key
        }
        kept_metadata[ORIGINAL_LABEL_METADATA_KEY] = source_series.metadata.get(
            ORIGINAL_LABEL_METADATA_KEY,
            str(source_series.label),
        )

        signal = source_series.value
        total_samples = len(signal)
        examples = []

        for start in range(0, total_samples, self.stride):
            end = start + self.window_size
            segment = signal[start:end]

            if len(segment) < self.window_size and not self.keep_short_segments:
                continue

            examples.append(Example(
                value=segment,
                label=source_series.label,
                metadata=kept_metadata,
            ))

        return examples
