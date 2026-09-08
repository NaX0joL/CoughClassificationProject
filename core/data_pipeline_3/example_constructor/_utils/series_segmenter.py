import numpy as np

from ...intermediary import SeriesSegment



class SeriesSegmenter():
    
    def __init__(self, window_size:int, stride:int, drop_last:bool=True) -> None:
        self.window_size = window_size
        self.stride = stride
        self.drop_last = drop_last
        return
    
    def segment(self, series:np.ndarray) -> list[SeriesSegment]:
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

        remainder_size = len(series) % self.stride
        if self.drop_last or remainder_size == 0:
            return segments

        last_segment = self._add_last_padded_segment(series, remainder_size)
        segments.append(last_segment)

        return segments
    
    def _add_last_padded_segment(self, series:np.ndarray, remainder_size:int) -> SeriesSegment:
        segment_start = len(series) - remainder_size
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
