
from dataclasses import dataclass

from .abstract import Padder, Segmenter, SourceReader, Transformer
from .dataset import ExampleDataset
from .intermediary import Example



@dataclass
class DataPipeline:
    source_reader: SourceReader
    segmenter: Segmenter
    transformer: Transformer
    padder: Padder

    def get_examples(self) -> list[Example]:
        source_series = self.source_reader.get_source_series()
        examples = self.segmenter.segment(source_series)
        examples = self.transformer.transform(examples)
        examples = self.padder.pad(examples)
        return examples

    def get_dataset(self) -> ExampleDataset:
        examples = self.get_examples()
        return ExampleDataset(examples)
