from collections import Counter
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np

from core.data_pipeline_3.data_module import DataModule
from core.data_pipeline_3.dataset import ExampleDataset
from core.data_pipeline_3.intermediary import Example, FoldPartition, SourceRecord
from core.data_pipeline_3.oversampler import UniformOversampler
from core.data_pipeline_3.pipeline import DataPipeline



def _make_example(label:int) -> Example:
    return Example(
        value=np.asarray([1.0, 2.0]),
        label=label,
        metadata={},
    )


def test_pipeline_returns_data_module_for_fold() -> None:
    train_records = [SourceRecord({}, Path("train.wav"), label=0)]
    validation_records = [SourceRecord({}, Path("validation.wav"), label=1)]
    test_records = [SourceRecord({}, Path("test.wav"), label=1)]
    source_reader = Mock()
    source_reader.get_source_records.return_value = (
        train_records + validation_records + test_records
    )
    partitioner = Mock()
    partitioner.partition.return_value = [FoldPartition(
        train=train_records,
        validation=validation_records,
        test=test_records,
    )]
    example_constructor = Mock()
    example_constructor.construct.side_effect = [
        [_make_example(0), _make_example(0), _make_example(1)],
        [_make_example(1)],
        [_make_example(1)],
    ]
    pipeline = DataPipeline(
        source_reader=source_reader,
        partitioner=partitioner,
        example_constructor=example_constructor,
        batch_size=2,
        oversampler=UniformOversampler(random_seed=42),
    )

    data_module = pipeline.get_data_module(index=0)

    assert isinstance(data_module, DataModule)
    train_dataset = data_module.train_loader.dataset
    assert isinstance(train_dataset, ExampleDataset)
    train_counts = Counter(
        example.label
        for example in train_dataset.examples
    )
    assert train_counts == {0: 2, 1: 2}
    assert len(data_module.validation_loader.dataset) == 1
    assert len(data_module.test_loader.dataset) == 1
    assert data_module.train_loader.batch_size == 2
    assert data_module.validation_loader.batch_size == 2
    assert data_module.test_loader.batch_size == 2
    assert example_constructor.construct.call_args_list == [
        call(train_records, type="train"),
        call(validation_records, type="validation"),
        call(test_records, type="test"),
    ]
