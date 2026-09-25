from sklearn.metrics import precision_score

from ._average_policy import _get_average_parameters
from .abstract import ClassificationMetric, ClassificationMetricInput


class PrecisionMetric(ClassificationMetric):
    name = "precision"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        average_parameters = _get_average_parameters(metric_input.class_labels)
        return float(precision_score(
            metric_input.labels,
            metric_input.predictions,
            **average_parameters,
            zero_division=0,
        ))
