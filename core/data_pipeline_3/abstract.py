from abc import ABC, abstractmethod

from .intermediary import SourceRecord, FoldPartition



class AbstractSourceReader(ABC):
    
    @abstractmethod
    def get_source_records(self) -> list[SourceRecord]:
        raise NotImplementedError


class AbstractPartitioner(ABC):
    
    @abstractmethod
    def partition(self, source_records:list[SourceRecord]) -> list[FoldPartition]:
        raise NotImplementedError