
from .abstract import AbstractSourceReader, AbstractPartitioner



class DataPipeline():
    
    def __init__(
        self,
        source_reader:AbstractSourceReader,
        partitioner:AbstractPartitioner,
    ) -> None:
        self.source_reader = source_reader
        self.partitioner = partitioner
        return
    
    def get_data_module(self):
        source_records = self.source_reader.get_source_records()
        fold_partitions = self.partitioner.partition(source_records)
        return