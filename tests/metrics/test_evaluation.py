from unittest.mock import Mock

import numpy as np

from core.metrics.evaluation import _get_original_labels
from core.metrics.evaluation import ModelEvaluator



def test_original_labels_fall_back_to_v3_infectious_metadata() -> None:
    batch = {
        "metadata": {
            "isInfectious": ["False", "True"],
        },
    }

    original_labels = _get_original_labels(batch)

    assert original_labels == ["False", "True"]


def test_evaluator_accepts_dataloader(monkeypatch) -> None:
    data_loader = Mock()
    model = Mock()
    metrics = Mock()
    metrics_calculator = Mock()
    metrics_calculator.calculate.return_value = metrics
    evaluator = ModelEvaluator.__new__(ModelEvaluator)
    evaluator.metrics_calculator = metrics_calculator
    collect_outputs = Mock(return_value=(
        np.asarray([1, 2]),
        np.asarray([1, 0]),
        np.asarray([
            [0.1, 0.8, 0.1],
            [0.6, 0.1, 0.3],
        ]),
        ["False", "True"],
    ))
    monkeypatch.setattr(evaluator, "_collect_outputs", collect_outputs)

    evaluation = evaluator.evaluate_dataloader(model, data_loader)

    collect_outputs.assert_called_once_with(model, data_loader)
    metric_call = metrics_calculator.calculate.call_args
    np.testing.assert_array_equal(metric_call.args[3], np.asarray([0, 1, 2]))
    assert evaluation.metrics is metrics
