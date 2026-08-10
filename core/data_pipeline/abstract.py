from abc import ABC, abstractmethod

from .intermediary import Example, SourceSeries



class SourceReader(ABC):
    
    @abstractmethod
    def get_source_series(self) -> list[SourceSeries]:
        pass


class Segmenter(ABC):

    @abstractmethod
    def segment(self, source_series: list[SourceSeries]) -> list[Example]:
        pass


class Transformer(ABC):

    @abstractmethod
    def transform(self, examples: list[Example]) -> list[Example]:
        pass


class Padder(ABC):

    @abstractmethod
    def pad(self, examples: list[Example]) -> list[Example]:
        pass
