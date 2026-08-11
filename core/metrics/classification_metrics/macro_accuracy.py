from sklearn.metrics import balanced_accuracy_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class MacroAccuracyMetric(ClassificationMetric):
    name = "macro_accuracy"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        return float(balanced_accuracy_score(
            metric_input.labels,
            metric_input.predictions,
        ))
