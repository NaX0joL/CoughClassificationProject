
from ...intermediary import SourceRecord, SeriesSegment, Example



class OverlapLabeler():
    
    def __init__(self, overlap_threshold:float, no_overlap_label:int) -> None:
        self.overlap_threshold = overlap_threshold
        self.no_overlap_label = no_overlap_label
        return
    
    def label(
        self, 
        segments:list[SeriesSegment], 
        cough_intervals:list[tuple[int, int]],
        source_label:int,
    ) -> list[int]:
        labels = []
        
        for segment in segments:
            overlap_ratio = self._get_largest_overlap_ratio(segment, cough_intervals)
            
            if overlap_ratio >= self.overlap_threshold:
                label = source_label
            else:
                label = self.no_overlap_label
                
            labels.append(label)
        
        return labels

    def _get_largest_overlap_ratio(
        self,
        segment:SeriesSegment, 
        cough_intervals:list[tuple[int, int]]
    ) -> float:    
        segment_start, segment_end = segment.original_index
        segment_size = segment_end - segment_start
        largest_overlap_ratio = 0.0
        
        for cough_start, cough_end in cough_intervals:
            overlap_start = max(segment_start, cough_start)
            overlap_end = min(segment_end, cough_end)
            
            overlap_size = max(0, overlap_end - overlap_start)
            overlap_ratio = overlap_size / segment_size

            largest_overlap_ratio = max(
                largest_overlap_ratio,
                overlap_ratio,
            )
        
        return largest_overlap_ratio