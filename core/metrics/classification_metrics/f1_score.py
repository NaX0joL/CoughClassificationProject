from sklearn.metrics import f1_score

from ._average_policy import _get_average_parameters
from .abstract import ClassificationMetric, ClassificationMetricInput


class F1ScoreMetric(ClassificationMetric):
    name = "f1_score"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        average_parameters = _get_average_parameters(metric_input.class_labels)
        return float(f1_score(
            metric_input.labels,
            metric_input.predictions,
            **average_parameters,
            zero_division=0,
        ))
