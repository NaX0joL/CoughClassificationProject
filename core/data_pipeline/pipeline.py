
from dataclasses import dataclass

from .abstract import Padder, Segmenter, SourceReader, Splitter, Transformer
from .data_pipeline_config import DataPipelineConfig
from .dataset import ExampleDataset
from .intermediary import DataSplit, Example



@dataclass
class DataPipeline:
    source_reader: SourceReader
    segmenter: Segmenter
    transformer: Transformer
    padder: Padder
    splitter:Splitter

    @classmethod
    def create(
        cls,
        config:DataPipelineConfig,
    ) -> "DataPipeline":
        pipeline = cls(
            source_reader=config.source_reader,
            segmenter=config.segmenter,
            transformer=config.transformer,
            padder=config.padder,
            splitter=config.splitter,
        )
        return pipeline

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
