from ..abstract import (
    AbstractDataPipeline,
    AbstractExampleConstructor,
    AbstractPartitioner,
    AbstractSourceReader,
)
from ..intermediary import FoldPartition, SourceRecord
from ..loading.data_module import DataModule
from ..loading.loader_creator import create_data_module
from ..loading.oversampler import UniformOversampler



class StandardDataPipeline(AbstractDataPipeline):

    def __init__(
        self,
        source_reader:AbstractSourceReader,
        partitioner:AbstractPartitioner,
        example_constructor:AbstractExampleConstructor,
        oversampler:UniformOversampler|None=None,
    ) -> None:
        self.source_reader = source_reader
        self.partitioner = partitioner
        self.example_constructor = example_constructor
        self.oversampler = oversampler
        
        self._is_initialized = False
        self._source_records:list[SourceRecord]|None = None
        self._fold_partitions:list[FoldPartition]|None = None
        return

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(self) -> None:
        if self.is_initialized:
            raise RuntimeError("data pipeline is already initialized")

        source_records = self.source_reader.get_source_records()
        fold_partitions = self.partitioner.partition(source_records)
        if not source_records:
            raise ValueError("data pipeline source records cannot be empty")
        if not fold_partitions:
            raise ValueError("data pipeline fold partitions cannot be empty")

        self._source_records = source_records
        self._fold_partitions = fold_partitions
        self._is_initialized = True
        return

    def __len__(self) -> int:
        self._require_initialization()
        fold_partitions = self._get_fold_partitions()
        
        return len(fold_partitions)

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
        self._require_initialization()
        fold_partitions = self._get_fold_partitions()
        
        if index < 0 or index >= len(fold_partitions):
            raise IndexError(f"fold index is out of range: {index}")

        fold_partition = fold_partitions[index]
        train_examples = self.example_constructor.construct(
            fold_partition.train,
            type="train",
        )
        validation_examples = self.example_constructor.construct(
            fold_partition.validation,
            type="validation",
        )
        test_examples = self.example_constructor.construct(
            fold_partition.test,
            type="test",
        )

        if self.oversampler is not None:
            train_examples = self.oversampler.oversample(train_examples)

        return create_data_module(
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last_batch=drop_last_batch,
            pin_memory=pin_memory,
            random_seed=random_seed,
        )

    def _require_initialization(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("data pipeline is not initialized")
        return

    def _get_fold_partitions(self) -> list[FoldPartition]:
        if self._fold_partitions is None:
            raise RuntimeError("data pipeline is not initialized")
        return self._fold_partitions
