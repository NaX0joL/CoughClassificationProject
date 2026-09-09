from pathlib import Path
from unittest.mock import Mock

import pytest

from core.data_pipeline_3.example_constructor._utils.segment_transformer import (
    MFCC,
)
from core.data_pipeline_3.oversampler import UniformOversampler
from core.data_pipeline_3.pipeline import DataPipeline
from core.model.architectures.MLP import MLP
from modules.yaml_experiment import YamlToExperimentConverter, do_experiment



PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = PROJECT_ROOT / "yaml" / "run" / "mfcc_sliding_windows_mlp_v3.yaml"



def test_converter_builds_v3_mlp(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DataPipeline,
        "init_fold_partitions",
        lambda self: setattr(self, "fold_partitions", []),
    )

    experiment = YamlToExperimentConverter().convert(YAML_PATH)

    data_pipeline = experiment.config.data_pipeline_config
    assert isinstance(data_pipeline, DataPipeline)
    assert isinstance(data_pipeline.example_constructor.transformers[0], MFCC)
    assert isinstance(data_pipeline.oversampler, UniformOversampler)
    assert isinstance(experiment.config.model_config.architecture, MLP)
    assert experiment.config.model_config.architecture.output_dim == 3


def test_runner_dispatches_v3_pipeline() -> None:
    experiment = Mock()

    do_experiment(experiment)

    experiment.train_model.assert_called_once_with()


def test_runner_preserves_existing_training_flow() -> None:
    experiment = Mock()

    do_experiment(experiment)

    experiment.train_model.assert_called_once_with()
