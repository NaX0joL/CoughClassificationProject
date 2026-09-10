import json
import sys
from types import SimpleNamespace

import pytest

from core.data_pipeline_3.intermediary import Example, FoldPartition
from core.experiment_config import ExperimentConfig
from core.metrics import AccuracyMetric, MetricsConfig
from scripts.analysis import recompute_mpkg_metrics
from scripts.analysis.recompute_mpkg_metrics import (
    extract_one_mpkg,
    recompute_one_mpkg,
)


def test_extract_one_mpkg_returns_saved_metrics_in_fold_order(tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    metrics_directory = mpkg_folder / "json" / "metrics"
    metrics_directory.mkdir(parents=True)
    (metrics_directory / "metrics-fold_10.json").write_text(
        json.dumps({"accuracy": 0.8, "f1_score": 0.7}),
        encoding="utf-8",
    )
    (metrics_directory / "metrics-fold_2.json").write_text(
        json.dumps({"accuracy": 0.6, "f1_score": 0.5}),
        encoding="utf-8",
    )

    fold_rows = extract_one_mpkg(mpkg_folder)

    assert fold_rows == [
        {
            "mpkg": "experiment",
            "folder": str(mpkg_folder),
            "fold": 2,
            "accuracy": 0.6,
            "f1_score": 0.5,
        },
        {
            "mpkg": "experiment",
            "folder": str(mpkg_folder),
            "fold": 10,
            "accuracy": 0.8,
            "f1_score": 0.7,
        },
    ]


def test_extract_one_mpkg_filters_selected_saved_metrics(tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    metrics_directory = mpkg_folder / "json" / "metrics"
    metrics_directory.mkdir(parents=True)
    (metrics_directory / "metrics-fold_1.json").write_text(
        json.dumps({"accuracy": 0.8, "f1_score": 0.7}),
        encoding="utf-8",
    )

    fold_rows = extract_one_mpkg(mpkg_folder, ["f1_score"])

    assert fold_rows == [{
        "mpkg": "experiment",
        "folder": str(mpkg_folder),
        "fold": 1,
        "f1_score": 0.7,
    }]


def test_extract_one_mpkg_rejects_missing_selected_metric(tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    metrics_directory = mpkg_folder / "json" / "metrics"
    metrics_directory.mkdir(parents=True)
    metric_path = metrics_directory / "metrics-fold_1.json"
    metric_path.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")

    with pytest.raises(ValueError, match="Saved metrics are missing f1_score"):
        extract_one_mpkg(mpkg_folder, ["f1_score"])


def test_load_replacement_config_imports_experiment_config(tmp_path) -> None:
    config_path = tmp_path / "replacement.py"
    config_path.write_text(
        "from core.experiment_config import ExperimentConfig\n"
        "from core.metrics import AccuracyMetric, MetricsConfig\n"
        "replacement_config = ExperimentConfig(\n"
        "    data_pipeline_config='replacement-data',\n"
        "    model_config='replacement-model',\n"
        "    training_config='replacement-training',\n"
        "    metrics_config=MetricsConfig(metrics=(AccuracyMetric(),)),\n"
        "    persistence_config='replacement-persistence',\n"
        ")\n",
        encoding="utf-8",
    )

    replacement_config = recompute_mpkg_metrics.load_replacement_config(
        f"{config_path}:replacement_config",
    )

    assert isinstance(replacement_config, ExperimentConfig)
    assert replacement_config.data_pipeline_config == "replacement-data"
    assert replacement_config.metrics_config.metrics[0].name == "accuracy"


def test_main_selects_metrics_from_replacement_config(monkeypatch, tmp_path) -> None:
    replacement_config = ExperimentConfig(
        data_pipeline_config=object(),
        model_config=object(),
        training_config=object(),
        metrics_config=MetricsConfig(metrics=(AccuracyMetric(),)),
        persistence_config=object(),
    )
    mpkg_folder = tmp_path / "experiment"
    calls = []

    def recompute(mpkg_folder, metrics_config, replacement_config):
        calls.append((mpkg_folder, metrics_config, replacement_config))
        return [{
            "mpkg": mpkg_folder.name,
            "folder": str(mpkg_folder),
            "fold": 1,
            "accuracy": 0.9,
        }]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recompute_mpkg_metrics.py",
            str(tmp_path),
            "--recompute",
            "--replacement-config",
            "replacement.py:replacement_config",
        ],
    )
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "find_mpkg_folders",
        lambda folder: [mpkg_folder],
    )
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "load_replacement_config",
        lambda reference: replacement_config,
    )
    monkeypatch.setattr(recompute_mpkg_metrics, "recompute_one_mpkg", recompute)
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "save_excel",
        lambda summary_table, fold_table, excel_path: None,
    )

    recompute_mpkg_metrics.main()

    assert calls == [(
        mpkg_folder,
        replacement_config.metrics_config,
        replacement_config,
    )]


def test_recompute_preserves_saved_model_config(monkeypatch, tmp_path) -> None:
    saved_model_config = object()
    replacement_model_config = object()
    saved_model = object()
    saved_metrics_config = MetricsConfig(metrics=(AccuracyMetric(),))

    class SourceReader:
        def get_source_records(self):
            return []

    class Partitioner:
        def partition(self, source_records):
            return [FoldPartition(train=[], validation=[], test=[])]

    class ExampleConstructor:
        def construct(self, source_records, type):
            return [Example(
                value=[0.0],
                label=0,
                metadata={"population": type},
            )]

    replacement_data_config = {
        "source_reader": SourceReader(),
        "partitioner": Partitioner(),
        "example_constructor": ExampleConstructor(),
        "batch_size": 1,
        "oversampler": None,
    }
    replacement_config = ExperimentConfig(
        data_pipeline_config=replacement_data_config,
        model_config=replacement_model_config,
        training_config=object(),
        metrics_config=saved_metrics_config,
        persistence_config=object(),
    )
    saved_experiment = SimpleNamespace(
        config=SimpleNamespace(
            data_pipeline_config={
                "source_reader": SourceReader(),
                "partitioner": Partitioner(),
                "example_constructor": ExampleConstructor(),
                "batch_size": 1,
                "oversampler": None,
            },
            model_config=saved_model_config,
            metrics_config=MetricsConfig(metrics=(AccuracyMetric(),)),
        ),
        persisted_folds=[SimpleNamespace(
            fold_index=1,
            model=saved_model,
        )],
    )
    evaluated_models = []

    class FakeEvaluator:
        def __init__(self, metrics_config):
            assert metrics_config is replacement_config.metrics_config

        def evaluate_dataloader(self, model, data_loader):
            evaluated_models.append((model, data_loader))
            return SimpleNamespace(
                metrics=SimpleNamespace(to_dict=lambda: {"accuracy": 0.9}),
            )

    monkeypatch.setattr(
        recompute_mpkg_metrics.ExperimentOrchestrator,
        "load",
        lambda folder: saved_experiment,
    )
    monkeypatch.setattr(recompute_mpkg_metrics, "ModelEvaluator", FakeEvaluator)

    rows = recompute_one_mpkg(
        tmp_path,
        replacement_config=replacement_config,
    )

    assert evaluated_models[0][0] is saved_model
    assert evaluated_models[0][1].dataset.examples[0].metadata == {
        "population": "test",
    }
    assert saved_experiment.config.model_config is saved_model_config
    assert replacement_config.model_config is replacement_model_config
    assert rows[0]["accuracy"] == 0.9


def test_main_recomputes_metrics_only_when_requested(monkeypatch, tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    recomputed_folders = []

    def recompute(mpkg_folder, metrics_config):
        recomputed_folders.append(mpkg_folder)
        return [{
            "mpkg": mpkg_folder.name,
            "folder": str(mpkg_folder),
            "fold": 1,
            "accuracy": 0.9,
        }]

    def reject_extraction(mpkg_folder, metric_names=None):
        raise AssertionError("stored metrics should not be extracted")

    monkeypatch.setattr(
        sys,
        "argv",
        ["recompute_mpkg_metrics.py", str(tmp_path), "--recompute"],
    )
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "find_mpkg_folders",
        lambda folder: [mpkg_folder],
    )
    monkeypatch.setattr(recompute_mpkg_metrics, "recompute_one_mpkg", recompute)
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "extract_one_mpkg",
        reject_extraction,
    )
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "save_excel",
        lambda summary_table, fold_table, excel_path: None,
    )

    recompute_mpkg_metrics.main()

    assert recomputed_folders == [mpkg_folder]
