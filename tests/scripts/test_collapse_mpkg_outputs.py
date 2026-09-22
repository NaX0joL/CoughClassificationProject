from types import SimpleNamespace

import numpy as np
import pytest

from core.metrics import AccuracyMetric, F1ScoreMetric, MetricsConfig
from scripts.analysis import collapse_mpkg_outputs
from scripts.analysis.collapse_mpkg_outputs import (
    collapse_predictions,
    collapse_probabilities,
    parse_class_map,
    recompute_one_mpkg,
    validate_class_map,
)


def test_parse_class_map_accepts_whitespace_and_rejects_duplicate_sources() -> None:
    assert parse_class_map("0:0, 1:0, 2:1") == {0: 0, 1: 0, 2: 1}

    with pytest.raises(ValueError, match="appears more than once"):
        parse_class_map("0:0,0:1,1:1")


@pytest.mark.parametrize(
    "value",
    ["0:0,1:0", "0:0,1:1,2:2", "0:0,1:1,"],
)
def test_parse_class_map_rejects_invalid_target_or_syntax(value) -> None:
    with pytest.raises(ValueError):
        parse_class_map(value)


def test_validate_class_map_requires_every_source_column_and_both_targets() -> None:
    with pytest.raises(ValueError, match="missing source classes"):
        validate_class_map({0: 0, 2: 1}, source_class_count=3)
    with pytest.raises(ValueError, match="both target classes"):
        validate_class_map({0: 0, 1: 0, 2: 0}, source_class_count=3)


def test_collapse_probabilities_sums_source_columns() -> None:
    probabilities = np.asarray([
        [0.10, 0.20, 0.30, 0.40],
        [0.70, 0.10, 0.15, 0.05],
    ])

    collapsed = collapse_probabilities(
        probabilities,
        {0: 0, 1: 0, 2: 1, 3: 1},
    )

    np.testing.assert_allclose(collapsed, [[0.30, 0.70], [0.80, 0.20]])


def test_collapse_predictions_remaps_labels_and_argmaxes_collapsed_probabilities() -> None:
    labels = np.asarray([0, 2, 3])
    original_predictions = np.asarray([0, 2, 1])
    probabilities = np.asarray([
        [0.40, 0.35, 0.15, 0.10],
        [0.05, 0.10, 0.45, 0.40],
        [0.45, 0.40, 0.10, 0.05],
    ])

    collapsed_labels, collapsed_predictions, collapsed_probabilities = (
        collapse_predictions(
            labels,
            original_predictions,
            probabilities,
            {0: 0, 1: 0, 2: 1, 3: 1},
        )
    )

    np.testing.assert_array_equal(collapsed_labels, [0, 1, 1])
    np.testing.assert_array_equal(collapsed_predictions, [0, 1, 0])
    np.testing.assert_allclose(
        collapsed_probabilities,
        [[0.75, 0.25], [0.15, 0.85], [0.85, 0.15]],
    )


def test_recompute_one_mpkg_recalculates_binary_metrics_for_each_fold(
    monkeypatch,
    tmp_path,
) -> None:
    class FakePipeline:
        def __len__(self):
            return 1

        def get_data_module(self, index):
            assert index == 0
            return SimpleNamespace(test_loader="test-loader")

    saved_experiment = SimpleNamespace(
        config=SimpleNamespace(
            data_pipeline_config="saved-components",
            metrics_config=MetricsConfig(
                metrics=(AccuracyMetric(), F1ScoreMetric()),
            ),
        ),
        persisted_folds=[SimpleNamespace(fold_index=1, model="saved-model")],
    )
    monkeypatch.setattr(
        collapse_mpkg_outputs.ExperimentOrchestrator,
        "load",
        lambda folder: saved_experiment,
    )
    monkeypatch.setattr(
        collapse_mpkg_outputs,
        "_create_current_data_pipeline",
        lambda config: FakePipeline(),
    )

    def collect_outputs(evaluator, model, data_loader):
        assert model == "saved-model"
        assert data_loader == "test-loader"
        return (
            np.asarray([0, 2, 3]),
            np.asarray([0, 2, 1]),
            np.asarray([
                [0.40, 0.35, 0.15, 0.10],
                [0.05, 0.10, 0.45, 0.40],
                [0.45, 0.40, 0.10, 0.05],
            ]),
            [],
        )

    monkeypatch.setattr(
        collapse_mpkg_outputs.ModelEvaluator,
        "_collect_outputs",
        collect_outputs,
    )

    rows = recompute_one_mpkg(
        tmp_path / "experiment",
        {0: 0, 1: 0, 2: 1, 3: 1},
    )

    assert rows[0]["fold"] == 1
    assert rows[0]["accuracy"] == pytest.approx(2 / 3)
    assert rows[0]["f1_score"] == pytest.approx(2 / 3)
