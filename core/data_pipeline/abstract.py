
from abc import ABC, abstractmethod



class SourceReader(ABC):
    
    @abstractmethod
    def get_source_series(self):
        return
