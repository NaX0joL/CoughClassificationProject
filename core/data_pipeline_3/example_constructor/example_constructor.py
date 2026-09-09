from pathlib import Path
from typing import cast, Literal
import numpy as np

from ..intermediary import SourceRecord, Example
from ._utils.audio_file_reader import AudioFileReader
from ._utils.cache import AudioWaveformCache
from ._utils.raw_audio_visualizer import RawAudioVisualizer
from ._utils.series_segmenter import CenteredCoughSegmenter, SlidingWindowSegmenter
from ._utils.segment_labeler import OverlapLabeler
from ._utils.segment_transformer import SegmentTransformer



DEFAULT_CACHE_DIRECTORY = Path(
    "outputs/cache/elderly_cough_audio_waveform_16khz_v3"
)



class ExampleConstructor():

    def __init__(
        self,
        sampling_rate:int,
        train_segmenter:SlidingWindowSegmenter|CenteredCoughSegmenter,
        validation_segmenter:SlidingWindowSegmenter|CenteredCoughSegmenter,
        test_segmenter:SlidingWindowSegmenter|CenteredCoughSegmenter,
        segment_labeler:OverlapLabeler,
        transformer:SegmentTransformer|list[SegmentTransformer]|None=None,
        cache_directory:Path=DEFAULT_CACHE_DIRECTORY,
        verbose:bool=False,
    ) -> None:
        self.verbose = verbose
        self.raw_audio_visualizer = RawAudioVisualizer()

        if transformer is None:
            self.transformers = []
        elif isinstance(transformer, list):
            self.transformers = transformer
        else:
            self.transformers = [transformer]
        
        self.audio_file_reader = AudioFileReader(sampling_rate)
        self.audio_cache = AudioWaveformCache(cache_directory, sampling_rate)
        self.train_segmenter = train_segmenter
        self.validation_segmenter = validation_segmenter
        self.test_segmenter = test_segmenter
        self.segment_labeler = segment_labeler
        return
    
    def construct(
        self,
        source_records:list[SourceRecord],
        type:Literal["train", "validation", "test"],
    ) -> list[Example]:

        if type == "train":
            segmenter = self.train_segmenter
        elif type == "validation":
            segmenter = self.validation_segmenter
        elif type == "test":
            segmenter = self.test_segmenter
        else:
            raise ValueError(f"invalid example type, got: {type}")

        examples = []
        
        for source_record in source_records:
            try:
                local_examples = self._construct_from_source_record(
                    source_record,
                    segmenter,
                    type,
                )
                examples.extend(local_examples)

            except Exception as e:
                if self.verbose:
                    print(e)
                continue
        
        return examples
    
    def _construct_from_source_record(
        self,
        source_record:SourceRecord,
        segmenter:SlidingWindowSegmenter|CenteredCoughSegmenter,
        type:Literal["train", "validation", "test"],
    ) -> list[Example]:

        audio_waveform = self._load_audio(source_record.audio_path)
        
        cough_intervals = self._get_cough_intervals(source_record)
        self.raw_audio_visualizer.visualize(
            audio_waveform,
            source_record.audio_path,
        )
        segments = segmenter.segment(audio_waveform, cough_intervals)

        source_label = source_record.label
        if type == "test":
            labels = [source_label for segment in segments]
        else:
            labels = self.segment_labeler.label(
                segments,
                cough_intervals,
                source_label,
            )

        # transform value
        for transformer in self.transformers:
            segments = transformer.transform(segments)

        examples = []
        for index in range(len(segments)):
            example = Example(
                value=segments[index].value,
                label=labels[index],
                metadata=source_record.metadata,
            )
            examples.append(example)

        return examples
    
    def _load_audio(self, path:Path) -> np.ndarray:
        cached_audio = self.audio_cache.read(path)
        if cached_audio is not None:
            return cached_audio

        audio = self.audio_file_reader.read(path)
        self.audio_cache.write(path, audio)
        return audio
    
    def _get_cough_intervals(self, source_record:SourceRecord) -> list[tuple[int, int]]:
        cough_intervals = source_record.metadata.get("DetectedCoughSegments", [])
        return cast(list[tuple[int, int]], cough_intervals)
