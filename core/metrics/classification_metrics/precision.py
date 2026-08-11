from sklearn.metrics import precision_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class PrecisionMetric(ClassificationMetric):
    name = "precision"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        return float(precision_score(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
            average="weighted",
            zero_division=0,
        ))
