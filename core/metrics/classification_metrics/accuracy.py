from sklearn.metrics import accuracy_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class AccuracyMetric(ClassificationMetric):
    name = "accuracy"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        return float(accuracy_score(metric_input.labels, metric_input.predictions))
