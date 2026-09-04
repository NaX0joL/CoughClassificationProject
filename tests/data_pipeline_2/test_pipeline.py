import numpy as np
import pytest

import core.data_pipeline_2 as data_pipeline_2
from core.data_pipeline_2 import (
    DataPipeline,
    Example,
    ExampleGenerator,
    FoldPartition,
    SourceSeries,
    TrainExampleGenerator,
    ValidationExampleGenerator,
)
from core.data_pipeline_2.abstract import Partitioner, SourceReader



class FakeSourceReader(SourceReader):

    def __init__(self, source_series:list[SourceSeries]) -> None:
        self.source_series = source_series
        self.number_of_calls = 0
        return

    def get_source_series(self) -> list[SourceSeries]:
        self.number_of_calls += 1
        return self.source_series



class FakePartitioner(Partitioner):

    def __init__(self, fold_partitions:list[FoldPartition]) -> None:
        self.fold_partitions = fold_partitions
        self.received_source_series:list[SourceSeries]|None = None
        return

    def partition(self, source_series:list[SourceSeries]) -> list[FoldPartition]:
        self.received_source_series = source_series
        return self.fold_partitions



class FakeTrainExampleGenerator(TrainExampleGenerator):

    def __init__(self) -> None:
        self.received_source_series:list[SourceSeries]|None = None
        return

    def generate(self, source_series:list[SourceSeries]) -> list[Example]:
        self.received_source_series = source_series
        return make_examples(source_series, "train")



class FakeValidationExampleGenerator(ValidationExampleGenerator):

    def __init__(self) -> None:
        self.received_source_series:list[SourceSeries]|None = None
        return

    def generate(self, source_series:list[SourceSeries]) -> list[Example]:
        self.received_source_series = source_series
        return make_examples(source_series, "validation")



class FakeTestExampleGenerator(data_pipeline_2.TestExampleGenerator):

    def __init__(self) -> None:
        self.received_source_series:list[SourceSeries]|None = None
        return

    def generate(self, source_series:list[SourceSeries]) -> list[Example]:
        self.received_source_series = source_series
        return make_examples(source_series, "test")



def make_source(identifier:str) -> SourceSeries:
    return SourceSeries(
        value=np.array([1.0]),
        metadata={"identifier": identifier},
    )


def make_examples(
    source_series:list[SourceSeries],
    split_name:str,
) -> list[Example]:
    return [
        Example(
            value=source.value,
            label=0,
            metadata={"split": split_name},
        )
        for source in source_series
    ]


def make_example_generator(
    train_generator:TrainExampleGenerator|None=None,
    validation_generator:ValidationExampleGenerator|None=None,
    test_generator:data_pipeline_2.TestExampleGenerator|None=None,
) -> ExampleGenerator:
    return ExampleGenerator(
        train_generator=train_generator or FakeTrainExampleGenerator(),
        validation_generator=(
            validation_generator or FakeValidationExampleGenerator()
        ),
        test_generator=test_generator or FakeTestExampleGenerator(),
    )


def test_pipeline_initializes_source_partitions_once() -> None:
    source_series = [make_source("source")]
    fold_partitions = [FoldPartition(train=[], validation=[], test=[])]
    source_reader = FakeSourceReader(source_series)
    partitioner = FakePartitioner(fold_partitions)
    pipeline = DataPipeline(
        source_reader=source_reader,
        partitioner=partitioner,
        example_generator=make_example_generator(),
    )

    pipeline.initialize()

    assert source_reader.number_of_calls == 1
    assert partitioner.received_source_series is source_series
    assert pipeline.fold_partitions is fold_partitions


def test_pipeline_creates_each_split_only_when_fold_is_requested() -> None:
    train = [make_source("train")]
    validation = [make_source("validation")]
    test = [make_source("test")]
    train_example_generator = FakeTrainExampleGenerator()
    validation_example_generator = FakeValidationExampleGenerator()
    test_example_generator = FakeTestExampleGenerator()
    pipeline = DataPipeline(
        source_reader=FakeSourceReader([]),
        partitioner=FakePartitioner(
            [FoldPartition(train=train, validation=validation, test=test)],
        ),
        example_generator=make_example_generator(
            train_generator=train_example_generator,
            validation_generator=validation_example_generator,
            test_generator=test_example_generator,
        ),
    )
    pipeline.initialize()

    examples = pipeline.build_fold(0)

    assert train_example_generator.received_source_series is train
    assert validation_example_generator.received_source_series is validation
    assert test_example_generator.received_source_series is test
    assert examples.train[0].metadata["split"] == "train"
    assert examples.validation[0].metadata["split"] == "validation"
    assert examples.test[0].metadata["split"] == "test"


def test_pipeline_rejects_fold_request_before_initialization() -> None:
    pipeline = DataPipeline(
        source_reader=FakeSourceReader([]),
        partitioner=FakePartitioner([]),
        example_generator=make_example_generator(),
    )

    with pytest.raises(RuntimeError, match="must be initialized"):
        pipeline.build_fold(0)


def test_pipeline_rejects_invalid_fold_index() -> None:
    pipeline = DataPipeline(
        source_reader=FakeSourceReader([]),
        partitioner=FakePartitioner([]),
        example_generator=make_example_generator(),
    )
    pipeline.initialize()

    with pytest.raises(IndexError, match="out of range"):
        pipeline.build_fold(0)
