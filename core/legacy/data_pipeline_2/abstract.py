from abc import ABC, abstractmethod

from .intermediary import DataSplit, Example, FoldPartition, Segment, SourceSeries



class SourceReader(ABC):

    @abstractmethod
    def get_source_series(self) -> list[SourceSeries]:
        pass



class Partitioner(ABC):

    @abstractmethod
    def partition(self, source_series:list[SourceSeries]) -> list[FoldPartition]:
        pass



class Segmenter(ABC):

    @abstractmethod
    def segment(self, source_series:list[SourceSeries]) -> list[Segment]:
        pass



class Labeler(ABC):

    @abstractmethod
    def label(self, segments:list[Segment]) -> list[Example]:
        pass



class Transformer(ABC):

    @abstractmethod
    def transform(self, examples:list[Example]) -> list[Example]:
        pass



class Padder(ABC):

    @abstractmethod
    def pad(self, examples:list[Example]) -> list[Example]:
        pass



class Balancer(ABC):

    @abstractmethod
    def balance(self, examples:list[Example]) -> list[Example]:
        pass



class Splitter(ABC):

    @abstractmethod
    def split(self, examples:list[Example]) -> DataSplit:
        """Split processed examples into test and development datasets."""
