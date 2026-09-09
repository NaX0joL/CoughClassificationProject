import numpy as np

from ...intermediary import SeriesSegment



class SlidingWindowSegmenter():
    
    def __init__(self, window_size:int, stride:int, drop_last:bool=True) -> None:
        self.window_size = window_size
        self.stride = stride
        self.drop_last = drop_last
        return
    
    def segment(
        self,
        series:np.ndarray,
        cough_intervals:list[tuple[int, int]]|None=None,
    ) -> list[SeriesSegment]:
        segments = []
        final_segment_start = len(series) - self.window_size

        for segment_start in range(0, final_segment_start + 1, self.stride):
            segment_end = segment_start + self.window_size
            segment_value = series[segment_start:segment_end].copy()
            
            series_segment = SeriesSegment(
                value=segment_value,
                original_index=(segment_start, segment_end),
            )
            segments.append(series_segment)

        if self.drop_last:
            return segments

        next_segment_start = len(segments) * self.stride
        if next_segment_start >= len(series):
            return segments

        last_segment = self._add_last_padded_segment(
            series,
            next_segment_start,
        )
        segments.append(last_segment)

        return segments
    
    def _add_last_padded_segment(
        self,
        series:np.ndarray,
        segment_start:int,
    ) -> SeriesSegment:
        segment_end = segment_start + self.window_size
        segment_value = series[segment_start:].copy()
        
        segment_value = np.pad(
            segment_value,
            pad_width=(0, self.window_size - len(segment_value)),
            mode="constant",
            constant_values=0,
        )
        return SeriesSegment(
            value=segment_value,
            original_index=(segment_start, segment_end),
        )



class CenteredCoughSegmenter():

    def __init__(self, window_size:int) -> None:
        self.window_size = window_size
        return

    def segment(
        self,
        series:np.ndarray,
        cough_intervals:list[tuple[int, int]],
    ) -> list[SeriesSegment]:
        segments = []

        for cough_start, cough_end in cough_intervals:
            cough_center = (cough_start + cough_end) // 2
            segment_start = cough_center - self.window_size // 2
            segment_end = segment_start + self.window_size
            source_start = max(0, segment_start)
            source_end = min(len(series), segment_end)
            segment_value = series[source_start:source_end].copy()
            segment_value = np.pad(
                segment_value,
                pad_width=(
                    max(0, -segment_start),
                    max(0, segment_end - len(series)),
                ),
                mode="constant",
                constant_values=0,
            )
            segments.append(SeriesSegment(
                value=segment_value,
                original_index=(segment_start, segment_end),
            ))

        return segments
