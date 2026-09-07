from unittest.mock import MagicMock, call

import numpy as np

from core.data_pipeline_2 import (
    DataPipeline,
    Example,
    ExampleBundle,
    MFCC,
    UniformOversamplingBalancer,
)
from scripts.analysis import print_data_pipeline_examples as print_examples_script
from scripts.analysis.print_data_pipeline_examples import (
    print_data_pipeline_examples,
)



def test_print_data_pipeline_examples_prints_every_fold_split(capsys) -> None:
    train_example = _make_example("train")
    validation_example = _make_example("validation")
    test_example = _make_example("test")
    data_pipeline = MagicMock(spec=DataPipeline)
    data_pipeline.fold_partitions = [object(), object()]
    data_pipeline.build_fold.return_value = ExampleBundle(
        train=[train_example],
        validation=[validation_example],
        test=[test_example],
    )

    print_data_pipeline_examples(data_pipeline)

    data_pipeline.initialize.assert_called_once_with()
    assert data_pipeline.build_fold.call_args_list == [call(0), call(1)]
    output = capsys.readouterr().out
    assert "fold=0 split=train index=0" in output
    assert str(train_example) in output
    assert "fold=0 split=validation index=0" in output
    assert str(validation_example) in output
    assert "fold=0 split=test index=0" in output
    assert str(test_example) in output
    assert "fold=1 split=train index=0" in output
    assert "fold=1 split=validation index=0" in output
    assert "fold=1 split=test index=0" in output


def test_main_builds_default_mfcc_data_pipeline(monkeypatch) -> None:
    received_pipelines = []
    monkeypatch.setattr(
        print_examples_script,
        "print_data_pipeline_examples",
        received_pipelines.append,
    )

    print_examples_script.main()

    assert len(received_pipelines) == 1
    data_pipeline = received_pipelines[0]
    assert isinstance(data_pipeline, DataPipeline)
    assert data_pipeline.partitioner._outer_fold_splitter.number_of_folds == 5
    train_generator = data_pipeline.example_generator.train_generator
    assert isinstance(train_generator.transformers[0], MFCC)
    assert isinstance(train_generator.balancer, UniformOversamplingBalancer)
    assert data_pipeline.example_generator.validation_generator.balancer is None
    assert data_pipeline.example_generator.test_generator.balancer is None


def _make_example(split_name:str) -> Example:
    return Example(
        value=np.array([1.0]),
        label=1,
        metadata={"split": split_name},
    )
