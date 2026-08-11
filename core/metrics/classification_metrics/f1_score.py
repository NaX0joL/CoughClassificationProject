from sklearn.metrics import f1_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class F1ScoreMetric(ClassificationMetric):
    name = "f1_score"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        return float(f1_score(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
            average="weighted",
            zero_division=0,
        ))
