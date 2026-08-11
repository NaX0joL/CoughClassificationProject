from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .abstract import ClassificationMetric, ClassificationMetricInput


@dataclass(frozen=True)
class ClassificationMetrics:
    values:dict[str, float]

    def __getattr__(self, name:str) -> float:
        try:
            return self.values[name]
        except KeyError as error:
            raise AttributeError(f"metric was not calculated: {name}") from error

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)



class ClassificationMetricsCalculator:
    """Calculate only the metric instances selected for a classification result."""

    def __init__(self, metrics:Sequence[ClassificationMetric]) -> None:
        self.metrics = tuple(metrics)
        _validate_metric_selection(self.metrics)
        return

    def calculate(
        self,
        labels:ArrayLike,
        predictions:ArrayLike,
        probabilities:ArrayLike,
        class_labels:ArrayLike|None=None,
    ) -> ClassificationMetrics:
        metric_input = _create_metric_input(
            labels,
            predictions,
            probabilities,
            class_labels,
        )
        values = {
            metric.name: metric.calculate(metric_input)
            for metric in self.metrics
        }
        return ClassificationMetrics(values)


def _validate_metric_selection(metrics:tuple[ClassificationMetric, ...]) -> None:
    metric_names = [metric.name for metric in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metrics must not contain duplicate names")
    return


def _create_metric_input(
    labels:ArrayLike,
    predictions:ArrayLike,
    probabilities:ArrayLike,
    class_labels:ArrayLike|None,
) -> ClassificationMetricInput:
    label_array = np.asarray(labels)
    prediction_array = np.asarray(predictions)
    probability_array = np.asarray(probabilities)
    resolved_class_labels = _resolve_class_labels(
        label_array,
        probability_array,
        class_labels,
    )
    _validate_inputs(
        label_array,
        prediction_array,
        probability_array,
        resolved_class_labels,
    )
    return ClassificationMetricInput(
        labels=label_array,
        predictions=prediction_array,
        probabilities=probability_array,
        class_labels=resolved_class_labels,
    )


def _resolve_class_labels(
    labels:np.ndarray,
    probabilities:np.ndarray,
    class_labels:ArrayLike|None,
) -> np.ndarray:
    if class_labels is not None:
        return np.asarray(class_labels)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")

    inferred_class_labels = np.unique(labels)
    if len(inferred_class_labels) != probabilities.shape[1]:
        raise ValueError(
            "class_labels is required when probabilities include classes absent from labels"
        )
    return inferred_class_labels


def _validate_inputs(
    labels:np.ndarray,
    predictions:np.ndarray,
    probabilities:np.ndarray,
    class_labels:np.ndarray,
) -> None:
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("labels must contain at least one example")
    if len(labels) != len(predictions) or len(labels) != len(probabilities):
        raise ValueError("labels, predictions, and probabilities must have equal lengths")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if len(class_labels) < 2:
        raise ValueError("at least two class labels are required")
    if probabilities.shape[1] != len(class_labels):
        raise ValueError("probabilities columns must match the number of class_labels")
    if len(np.unique(class_labels)) != len(class_labels):
        raise ValueError("class_labels must not contain duplicates")
    if not np.isin(labels, class_labels).all():
        raise ValueError("labels must be present in class_labels")
    if not np.isin(predictions, class_labels).all():
        raise ValueError("predictions must be present in class_labels")
    return
