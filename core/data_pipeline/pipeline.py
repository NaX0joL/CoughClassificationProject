
from dataclasses import dataclass

from .abstract import Padder, Segmenter, SourceReader, Transformer
from .dataset import ExampleDataset
from .intermediary import DataSplit, Example
from .stratifier import DataSplitter



@dataclass
class DataPipeline:
    source_reader: SourceReader
    segmenter: Segmenter
    transformer: Transformer
    padder: Padder
    splitter: DataSplitter

    def get_examples(self) -> list[Example]:
        source_series = self.source_reader.get_source_series()
        examples = self.segmenter.segment(source_series)
        examples = self.transformer.transform(examples)
        examples = self.padder.pad(examples)
        return examples

    def get_dataset(self) -> ExampleDataset:
        examples = self.get_examples()
        return ExampleDataset(examples)

    def get_data_split(self) -> DataSplit:
        examples = self.get_examples()
        data_split = self.splitter.split(examples)
        return data_split
