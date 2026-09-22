from .abstract import Partitioner, SourceReader
from .example_generator import ExampleGenerator
from .intermediary import ExampleBundle, FoldPartition



class DataPipeline:

    def __init__(
        self,
        source_reader:SourceReader,
        partitioner:Partitioner,
        example_generator:ExampleGenerator,
    ) -> None:
        self.source_reader = source_reader
        self.partitioner = partitioner
        self.example_generator = example_generator
        self.fold_partitions:list[FoldPartition]|None = None
        return

    def initialize(self) -> None:
        source_series = self.source_reader.get_source_series()
        self.fold_partitions = self.partitioner.partition(source_series)
        return

    def build_fold(self, fold_index:int) -> ExampleBundle:
        fold_partition = self._get_fold_partition(fold_index)
        return self.example_generator.generate(fold_partition)

    def _get_fold_partition(self, fold_index:int) -> FoldPartition:
        if self.fold_partitions is None:
            raise RuntimeError("DataPipeline must be initialized before building a fold")

        if fold_index < 0 or fold_index >= len(self.fold_partitions):
            raise IndexError(f"Fold index {fold_index} is out of range")

        return self.fold_partitions[fold_index]
