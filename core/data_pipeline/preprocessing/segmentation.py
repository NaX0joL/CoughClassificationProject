
from ..intermediary import SourceSeries, Example



class CoughSegmenter():
    
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
        kept_metadata = {key: value for key, value in source_series.metadata.items()}
        examples = []
        
        for start, end in source_series.metadata["detected_cough_segments"]:
            example = Example(
                value=source_series.value[start:end + 1],
                label=source_series.label,
                metadata=kept_metadata,
            )
            examples.append(example)
        
        return examples