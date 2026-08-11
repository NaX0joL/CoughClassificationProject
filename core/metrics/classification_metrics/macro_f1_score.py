from sklearn.metrics import f1_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class MacroF1ScoreMetric(ClassificationMetric):
    name = "macro_f1_score"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        return float(f1_score(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
            average="macro",
            zero_division=0,
        ))
