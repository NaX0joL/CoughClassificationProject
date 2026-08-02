

from abc import ABC, abstractmethod



class SourceSeries(ABC):
    @abstractmethod
    def __init__(self, value:tuple[int, ...], metadata:dict[str, str]) -> None:
        self.value = value
        self.metadata = metadata
        return



class CanonicalSeries(SourceSeries):
    def __init__(self, value:tuple[int, ...], metadata:dict[str, str]) -> None:
        super().__init__(value, metadata)
        return
