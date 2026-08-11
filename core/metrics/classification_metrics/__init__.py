from collections.abc import Sequence

from numpy.typing import ArrayLike

from .abstract import ClassificationMetric
from .accuracy import AccuracyMetric
from .calculator import ClassificationMetrics, ClassificationMetricsCalculator
from .f1_score import F1ScoreMetric
from .macro_accuracy import MacroAccuracyMetric
from .macro_f1_score import MacroF1ScoreMetric
from .pr_auc import PRAucMetric
from .precision import PrecisionMetric
from .recall import RecallMetric
from .roc_auc import RocAucMetric
from .specificity import SpecificityMetric


def calculate_classification_metrics(
    labels:ArrayLike,
    predictions:ArrayLike,
    probabilities:ArrayLike,
    class_labels:ArrayLike|None=None,
    metrics:Sequence[ClassificationMetric]|None=None,
) -> ClassificationMetrics:
    selected_metrics = metrics
    if selected_metrics is None:
        from ..metrics_config import MetricsConfig

        selected_metrics = MetricsConfig.default().metrics

    calculator = ClassificationMetricsCalculator(selected_metrics)
    return calculator.calculate(labels, predictions, probabilities, class_labels)


__all__ = [
    "AccuracyMetric",
    "ClassificationMetric",
    "ClassificationMetrics",
    "ClassificationMetricsCalculator",
    "F1ScoreMetric",
    "MacroAccuracyMetric",
    "MacroF1ScoreMetric",
    "PRAucMetric",
    "PrecisionMetric",
    "RecallMetric",
    "RocAucMetric",
    "SpecificityMetric",
    "calculate_classification_metrics",
]
