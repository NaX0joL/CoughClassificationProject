from sklearn.metrics import recall_score

from ._average_policy import _get_average_parameters
from .abstract import ClassificationMetric, ClassificationMetricInput


class RecallMetric(ClassificationMetric):
    name = "recall"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        average_parameters = _get_average_parameters(metric_input.class_labels)
        return float(recall_score(
            metric_input.labels,
            metric_input.predictions,
            **average_parameters,
            zero_division=0,
        ))
