from sklearn.metrics import precision_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class MacroPrecisionMetric(ClassificationMetric):
    name = "macro_precision"

    def calculate(self, metric_input: ClassificationMetricInput) -> float:
        return float(precision_score(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
            average="macro",
            zero_division=0,
        ))
