from dataclasses import dataclass

from .abstract import SourceReader, Segmenter, Transformer, Padder, Splitter



@dataclass
class DataPipelineConfig:
    source_reader:SourceReader
    segmenter:Segmenter
    transformer:Transformer
    padder:Padder
    splitter:Splitter
