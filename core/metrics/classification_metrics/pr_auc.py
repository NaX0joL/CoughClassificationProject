import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from .abstract import ClassificationMetric, ClassificationMetricInput


class PRAucMetric(ClassificationMetric):
    name = "pr_auc"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        if len(metric_input.class_labels) == 2:
            binary_labels = metric_input.labels == metric_input.class_labels[1]
            return float(average_precision_score(
                binary_labels,
                metric_input.probabilities[:, 1],
            ))

        binarized_labels = label_binarize(
            metric_input.labels,
            classes=metric_input.class_labels,
        )
        return float(average_precision_score(
            np.asarray(binarized_labels),
            metric_input.probabilities,
            average="macro",
        ))
