import numpy as np
from sklearn.metrics import f1_score

from .abstract import ClassificationMetric, ClassificationMetricInput


BINARY_CLASS_LABELS = np.asarray([0, 1])


class F1ScoreMetric(ClassificationMetric):
    name = "f1_score"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        if np.array_equal(metric_input.class_labels, BINARY_CLASS_LABELS):
            return float(f1_score(
                metric_input.labels,
                metric_input.predictions,
                pos_label=1,
                average="binary",
                zero_division=0,
            ))
        return float(f1_score(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
            average="weighted",
            zero_division=0,
        ))
