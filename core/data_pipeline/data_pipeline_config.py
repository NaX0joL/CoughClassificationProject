from dataclasses import dataclass

from .abstract import SourceReader, Segmenter, Transformer, Padder, Splitter



@dataclass
class DataPipelineConfig:
    source_reader:SourceReader
    segmenter:Segmenter
    transformer:Transformer | list[Transformer]
    padder:Padder
    splitter:Splitter

    @classmethod
    def default(cls) -> "DataPipelineConfig":
        from .preprocessing import CoughSegmenter, MFCC, ZeroPadder
        from .source_reader import ElderlyCoughAudioSourceReader
        from .stratifier import DataSplitter

        data_pipeline_config = cls(
            source_reader=ElderlyCoughAudioSourceReader(),
            segmenter=CoughSegmenter(
                kept_metadata_key=["patient_id", "cough_audio"],
            ),
            transformer=MFCC(
                sample_rate=16_000,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=40,
                n_mfcc=40,
            ),
            padder=ZeroPadder(
                target_length=820,
                padding_type="random",
                random_seed=42,
            ),
            splitter=DataSplitter(
                group_metadata_key="patient_id",
                test_ratio=0.1,
                number_of_folds=5,
                random_seed=42,
            ),
        )
        return data_pipeline_config
