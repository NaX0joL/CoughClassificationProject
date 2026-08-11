from sklearn.metrics import roc_auc_score

from .abstract import ClassificationMetric, ClassificationMetricInput


class RocAucMetric(ClassificationMetric):
    name = "roc_auc"

    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        if len(metric_input.class_labels) == 2:
            return float(roc_auc_score(
                metric_input.labels,
                metric_input.probabilities[:, 1],
                labels=metric_input.class_labels,
            ))

        return float(roc_auc_score(
            metric_input.labels,
            metric_input.probabilities,
            labels=metric_input.class_labels,
            multi_class="ovr",
            average="macro",
        ))
