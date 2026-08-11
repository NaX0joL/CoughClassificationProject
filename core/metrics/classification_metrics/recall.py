from sklearn.metrics import recall_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class RecallMetric(ClassificationMetric):
    name = "recall"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        return float(recall_score(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
            average="weighted",
            zero_division=0,
        ))
