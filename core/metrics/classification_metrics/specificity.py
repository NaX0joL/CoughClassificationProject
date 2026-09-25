import numpy as np
from sklearn.metrics import confusion_matrix

from .abstract import ClassificationMetric, ClassificationMetricInput


class SpecificityMetric(ClassificationMetric):
    name = "specificity"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        matrix = confusion_matrix(
            metric_input.labels,
            metric_input.predictions,
            labels=metric_input.class_labels,
        )
        if np.array_equal(metric_input.class_labels, np.asarray([0, 1])):
            true_negatives = matrix[0, 0]
            false_positives = matrix[0, 1]
            denominator = true_negatives + false_positives
            return float(true_negatives / denominator) if denominator else 0.0

        total_examples = matrix.sum()
        true_negatives = total_examples - (
            matrix.sum(axis=0) + matrix.sum(axis=1) - np.diag(matrix)
        )
        false_positives = matrix.sum(axis=0) - np.diag(matrix)
        specificities = np.divide(
            true_negatives,
            true_negatives + false_positives,
            out=np.zeros_like(true_negatives, dtype=float),
            where=(true_negatives + false_positives) != 0,
        )
        return float(specificities.mean())
