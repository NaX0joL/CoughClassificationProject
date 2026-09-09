import torch
from torch.utils.data import DataLoader

from .abstract import (
    AbstractExampleConstructor,
    AbstractPartitioner,
    AbstractSourceReader,
)
from .data_module import DataModule
from .dataset import ExampleDataset
from .oversampler import UniformOversampler



class DataPipeline():
    
    def __init__(
        self,
        source_reader:AbstractSourceReader,
        partitioner:AbstractPartitioner,
        example_constructor:AbstractExampleConstructor,
        batch_size:int,
        oversampler:UniformOversampler|None=None,
    ) -> None:
        self.source_reader = source_reader
        self.partitioner = partitioner
        self.example_constructor = example_constructor
        self.batch_size = batch_size
        self.oversampler = oversampler

        self.init_fold_partitions()
        return

    def init_fold_partitions(self) -> None:
        source_records = self.source_reader.get_source_records()
        self.fold_partitions = self.partitioner.partition(source_records)
        return

    def __len__(self) -> int:
        return len(self.fold_partitions)

    def get_data_module(
        self,
        index:int,
        *,
        num_workers:int=0,
        drop_last_batch:bool=False,
        pin_memory:bool=False,
        random_seed:int|None=None,
    ) -> DataModule:
        if index < 0 or index >= len(self.fold_partitions):
            raise IndexError(f"fold index is out of range: {index}")

        fold_partition = self.fold_partitions[index]
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

        generator = None
        if random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(random_seed)

        return DataModule(
            train_loader=DataLoader(
                dataset=ExampleDataset(train_examples),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=drop_last_batch,
                pin_memory=pin_memory,
                generator=generator,
            ),
            validation_loader=DataLoader(
                dataset=ExampleDataset(validation_examples),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            test_loader=DataLoader(
                dataset=ExampleDataset(test_examples),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
        )
