from pathlib import Path
from typing import cast
import numpy as np

from ..intermediary import SourceRecord, Example
from ._utils.audio_file_reader import AudioFileReader
from ._utils.cache import AudioWaveformCache
from ._utils.series_segmenter import SeriesSegmenter
from ._utils.segment_labeler import OverlapLabeler
from ._utils.transformer import ExampleTransformer



DEFAULT_CACHE_DIRECTORY = Path("outputs/cache/data_pipeline_3/audio")



class ExampleConstructor():
    
    def __init__(
        self, 
        sampling_rate:int,
        window_size:int,
        stride:int,
        drop_last:bool,
        overlap_threshold:float,
        metadata_to_label_mapping:dict[bool, int],
        transformer:ExampleTransformer|list[ExampleTransformer]|None=None,
        cache_directory:Path=DEFAULT_CACHE_DIRECTORY,
    ) -> None:
        self.metadata_to_label_mapping = metadata_to_label_mapping

        if transformer is None:
            self.transformers = []
        elif isinstance(transformer, list):
            self.transformers = transformer
        else:
            self.transformers = [transformer]
        
        self.audio_file_reader = AudioFileReader(sampling_rate)
        self.audio_cache = AudioWaveformCache(cache_directory, sampling_rate)
        self.series_segmenter = SeriesSegmenter(window_size, stride, drop_last)
        self.segment_labeler = OverlapLabeler(overlap_threshold, no_overlap_label=0)
        return
    
    def construct(self, source_records:list[SourceRecord]) -> list[Example]:
        examples = []
        
        for source_record in source_records:
            local_examples = self._construct_from_source_record(source_record)
            examples.extend(local_examples)
        
        return examples
    
    def _construct_from_source_record(self, source_record:SourceRecord) -> list[Example]:
        audio_waveform = self._load_audio(source_record.audio_path)
        
        # slide window across the audio waveform
        segments = self.series_segmenter.segment(audio_waveform)
        
        # give labels
        cough_intervals = self._get_cough_intervals(source_record)
        source_label = self._get_source_label(source_record)
        labels = self.segment_labeler.label(segments, cough_intervals, source_label)
        
        examples = []
        for index in range(len(segments)):
            example = Example(
                value=segments[index].value,
                label=labels[index],
                metadata=source_record.metadata,
            )
            examples.append(example)

        for transformer in self.transformers:
            examples = transformer.transform(examples)

        return examples
    
    def _load_audio(self, path:Path) -> np.ndarray:
        cached_audio = self.audio_cache.read(path)
        if cached_audio is not None:
            return cached_audio

        audio = self.audio_file_reader.read(path)
        self.audio_cache.write(path, audio)
        return audio
    
    def _get_source_label(self, source_record:SourceRecord) -> int:
        metadata_value = source_record.metadata.get("isInfectious")
        metadata_value = cast(bool, metadata_value)
        source_label = self.metadata_to_label_mapping[metadata_value]
        return source_label
    
    def _get_cough_intervals(self, source_record:SourceRecord) -> list[tuple[int, int]]:
        cough_intervals = source_record.metadata.get("DetectedCoughSegments", [])
        return cast(list[tuple[int, int]], cough_intervals)
