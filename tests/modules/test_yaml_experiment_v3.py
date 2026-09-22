from pathlib import Path
from unittest.mock import Mock

import pytest

from core.data_pipeline_3.example_constructor._utils.segment_transformer import (
    MFCC,
)
from core.data_pipeline_3.oversampler import UniformOversampler
from core.data_pipeline_3.pipeline import DataPipeline
from core.model.architectures.MLP import MLP
from modules import yaml_experiment
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


def test_runner_dispatches_v3_pipeline(
    monkeypatch:pytest.MonkeyPatch,
    tmp_path:Path,
) -> None:
    experiment = Mock()
    experiment.run_directory = tmp_path
    _mock_report_dependencies(monkeypatch)

    do_experiment(experiment)

    experiment.train_model.assert_called_once_with()


def test_runner_preserves_existing_training_flow(
    monkeypatch:pytest.MonkeyPatch,
    tmp_path:Path,
) -> None:
    experiment = Mock()
    experiment.run_directory = tmp_path
    _mock_report_dependencies(monkeypatch)

    do_experiment(experiment)

    experiment.train_model.assert_called_once_with()


def test_runner_writes_collapsed_report_after_training(
    monkeypatch:pytest.MonkeyPatch,
    tmp_path:Path,
) -> None:
    experiment = Mock()
    experiment.run_directory = tmp_path
    events:list[str] = []
    experiment.train_model.side_effect = lambda: events.append("train")
    recompute = Mock(side_effect=lambda *args, **kwargs: events.append("recompute") or [
        {"mpkg": "run", "fold": 1, "accuracy": 1.0},
    ])
    create_summary = Mock(side_effect=lambda table: events.append("summary") or table)
    save = Mock(side_effect=lambda *args: events.append("save"))
    monkeypatch.setattr(yaml_experiment, "_recompute_mpkg_folders", recompute)
    monkeypatch.setattr(yaml_experiment, "create_summary_table", create_summary)
    monkeypatch.setattr(yaml_experiment, "save_report", save)

    do_experiment(experiment)

    assert events == ["train", "recompute", "summary", "save"]
    recompute.assert_called_once_with(
        tmp_path,
        {0: 0, 1: 0, 2: 1},
        metric_names=None,
    )
    save.assert_called_once()
    assert save.call_args.args[2] == tmp_path / "collapsed_output_metrics.xlsx"
    assert save.call_args.args[3] == {0: 0, 1: 0, 2: 1}


def test_runner_does_not_write_report_when_training_fails(
    monkeypatch:pytest.MonkeyPatch,
    tmp_path:Path,
) -> None:
    experiment = Mock()
    experiment.run_directory = tmp_path
    experiment.train_model.side_effect = RuntimeError("training failed")
    recompute = Mock()
    create_summary = Mock()
    save = Mock()
    monkeypatch.setattr(yaml_experiment, "_recompute_mpkg_folders", recompute)
    monkeypatch.setattr(yaml_experiment, "create_summary_table", create_summary)
    monkeypatch.setattr(yaml_experiment, "save_report", save)

    with pytest.raises(RuntimeError, match="training failed"):
        do_experiment(experiment)

    recompute.assert_not_called()
    create_summary.assert_not_called()
    save.assert_not_called()


def _mock_report_dependencies(monkeypatch:pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        yaml_experiment,
        "_recompute_mpkg_folders",
        lambda *args, **kwargs: [{"mpkg": "run", "fold": 1}],
    )
    monkeypatch.setattr(
        yaml_experiment,
        "create_summary_table",
        lambda table: table,
    )
    monkeypatch.setattr(yaml_experiment, "save_report", lambda *args: None)
    return
