from abc import ABC, abstractmethod
from typing import Literal

from .intermediary import Example, FoldPartition, SourceRecord
from .loading.data_module import DataModule



class AbstractSourceReader(ABC):
    
    @abstractmethod
    def get_source_records(self) -> list[SourceRecord]:
        raise NotImplementedError


class AbstractPartitioner(ABC):
    
    @abstractmethod
    def partition(self, source_records:list[SourceRecord]) -> list[FoldPartition]:
        raise NotImplementedError


class AbstractExampleConstructor(ABC):
    
    @abstractmethod
    def construct(
        self,
        source_records:list[SourceRecord],
        type:Literal["train", "validation", "test"],
    ) -> list[Example]:
        raise NotImplementedError


class AbstractDataPipeline(ABC):

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_data_module(
        self,
        index:int,
        *,
        batch_size:int,
        num_workers:int=0,
        drop_last_batch:bool=False,
        pin_memory:bool=False,
        random_seed:int|None=None,
    ) -> DataModule:
        raise NotImplementedError
