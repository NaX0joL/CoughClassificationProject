

from dataclasses import dataclass

from .abstract import Padder, Segmenter, SourceReader, Splitter, Transformer
from .data_pipeline_config import DataPipelineConfig
from .dataset import ExampleDataset
from .intermediary import DataSplit, Example



@dataclass
class DataPipeline:
    source_reader: SourceReader
    segmenter: Segmenter
    transformer: Transformer | list[Transformer]
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

    def _apply_transformers(self, examples:list[Example]) -> list[Example]:
        if isinstance(self.transformer, list):
            for t in self.transformer:
                examples = t.transform(examples)
            return examples

        return self.transformer.transform(examples)

    def get_examples(self) -> list[Example]:
        source_series = self.source_reader.get_source_series()
        examples = self.segmenter.segment(source_series)
        examples = self._apply_transformers(examples)
        examples = self.padder.pad(examples)
        return examples

    def get_dataset(self) -> ExampleDataset:
        examples = self.get_examples()
        return ExampleDataset(examples)

    def get_data_split(self) -> DataSplit:
        examples = self.get_examples()
        data_split = self.splitter.split(examples)
        return data_split
