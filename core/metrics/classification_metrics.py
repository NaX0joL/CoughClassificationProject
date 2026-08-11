from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize



@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate metrics for a binary or multiclass classification result."""

    roc_auc:float
    pr_auc:float
    precision:float
    recall:float
    specificity:float
    f1_score:float
    accuracy:float
    macro_accuracy:float
    macro_f1_score:float

    def to_dict(self) -> dict[str, float]:
        return cast(dict[str, float], asdict(self))



def calculate_classification_metrics(
    labels:ArrayLike,
    predictions:ArrayLike,
    probabilities:ArrayLike,
    class_labels:ArrayLike|None=None,
) -> ClassificationMetrics:
    """Calculate classification metrics from labels, predicted classes, and probabilities.

    Precision, recall, and F1-score use weighted averages. Specificity and both
    AUC metrics use macro averaging. Macro accuracy is balanced accuracy.
    """
    label_array = np.asarray(labels)
    prediction_array = np.asarray(predictions)
    probability_array = np.asarray(probabilities)
    resolved_class_labels = _resolve_class_labels(
        label_array,
        probability_array,
        class_labels,
    )
    _validate_inputs(
        label_array,
        prediction_array,
        probability_array,
        resolved_class_labels,
    )

    metrics = ClassificationMetrics(
        roc_auc=_calculate_roc_auc(
            label_array,
            probability_array,
            resolved_class_labels,
        ),
        pr_auc=_calculate_pr_auc(
            label_array,
            probability_array,
            resolved_class_labels,
        ),
        precision=_calculate_weighted_precision(
            label_array,
            prediction_array,
            resolved_class_labels,
        ),
        recall=_calculate_weighted_recall(
            label_array,
            prediction_array,
            resolved_class_labels,
        ),
        specificity=_calculate_macro_specificity(
            label_array,
            prediction_array,
            resolved_class_labels,
        ),
        f1_score=_calculate_weighted_f1_score(
            label_array,
            prediction_array,
            resolved_class_labels,
        ),
        accuracy=_calculate_accuracy(label_array, prediction_array),
        macro_accuracy=_calculate_macro_accuracy(label_array, prediction_array),
        macro_f1_score=_calculate_macro_f1_score(
            label_array,
            prediction_array,
            resolved_class_labels,
        ),
    )
    return metrics


def _resolve_class_labels(
    labels:np.ndarray,
    probabilities:np.ndarray,
    class_labels:ArrayLike|None,
) -> np.ndarray:
    if class_labels is not None:
        return np.asarray(class_labels)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")

    inferred_class_labels = np.unique(labels)
    if len(inferred_class_labels) != probabilities.shape[1]:
        raise ValueError(
            "class_labels is required when probabilities include classes absent from labels"
        )

    return inferred_class_labels


def _calculate_weighted_precision(
    labels:np.ndarray,
    predictions:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    precision = precision_score(
        cast(ArrayLike, labels),
        cast(ArrayLike, predictions),
        labels=cast(ArrayLike, class_labels),
        average="weighted",
        zero_division=0,
    )
    return cast(float, precision)


def _calculate_accuracy(labels:np.ndarray, predictions:np.ndarray) -> float:
    accuracy = accuracy_score(cast(ArrayLike, labels), cast(ArrayLike, predictions))
    return cast(float, accuracy)


def _calculate_macro_accuracy(labels:np.ndarray, predictions:np.ndarray) -> float:
    macro_accuracy = balanced_accuracy_score(
        cast(ArrayLike, labels),
        cast(ArrayLike, predictions),
    )
    return cast(float, macro_accuracy)


def _calculate_weighted_recall(
    labels:np.ndarray,
    predictions:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    recall = recall_score(
        cast(ArrayLike, labels),
        cast(ArrayLike, predictions),
        labels=cast(ArrayLike, class_labels),
        average="weighted",
        zero_division=0,
    )
    return cast(float, recall)


def _calculate_weighted_f1_score(
    labels:np.ndarray,
    predictions:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    weighted_f1_score = f1_score(
        cast(ArrayLike, labels),
        cast(ArrayLike, predictions),
        labels=cast(ArrayLike, class_labels),
        average="weighted",
        zero_division=0,
    )
    return cast(float, weighted_f1_score)


def _calculate_macro_f1_score(
    labels:np.ndarray,
    predictions:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    macro_f1_score = f1_score(
        cast(ArrayLike, labels),
        cast(ArrayLike, predictions),
        labels=cast(ArrayLike, class_labels),
        average="macro",
        zero_division=0,
    )
    return cast(float, macro_f1_score)


def _validate_inputs(
    labels:np.ndarray,
    predictions:np.ndarray,
    probabilities:np.ndarray,
    class_labels:np.ndarray,
) -> None:
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional arrays")

    if len(labels) == 0:
        raise ValueError("labels must contain at least one example")

    if len(labels) != len(predictions) or len(labels) != len(probabilities):
        raise ValueError("labels, predictions, and probabilities must have equal lengths")

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")

    if len(class_labels) < 2:
        raise ValueError("at least two class labels are required")

    if probabilities.shape[1] != len(class_labels):
        raise ValueError("probabilities columns must match the number of class_labels")

    if len(np.unique(class_labels)) != len(class_labels):
        raise ValueError("class_labels must not contain duplicates")

    if not np.isin(labels, class_labels).all():
        raise ValueError("labels must be present in class_labels")

    if not np.isin(predictions, class_labels).all():
        raise ValueError("predictions must be present in class_labels")

    return


def _calculate_roc_auc(
    labels:np.ndarray,
    probabilities:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    if len(class_labels) == 2:
        roc_auc = roc_auc_score(
            cast(ArrayLike, labels),
            cast(ArrayLike, probabilities[:, 1]),
            labels=cast(ArrayLike, class_labels),
        )
        return cast(float, roc_auc)

    roc_auc = roc_auc_score(
        cast(ArrayLike, labels),
        cast(ArrayLike, probabilities),
        labels=cast(ArrayLike, class_labels),
        multi_class="ovr",
        average="macro",
    )
    return cast(float, roc_auc)


def _calculate_pr_auc(
    labels:np.ndarray,
    probabilities:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    if len(class_labels) == 2:
        binary_labels = labels == class_labels[1]
        pr_auc = average_precision_score(
            cast(ArrayLike, binary_labels),
            cast(ArrayLike, probabilities[:, 1]),
        )
        return cast(float, pr_auc)

    binarized_labels = label_binarize(
        cast(ArrayLike, labels),
        classes=cast(ArrayLike, class_labels),
    )
    pr_auc = average_precision_score(
        cast(ArrayLike, binarized_labels),
        cast(ArrayLike, probabilities),
        average="macro",
    )
    return cast(float, pr_auc)


def _calculate_macro_specificity(
    labels:np.ndarray,
    predictions:np.ndarray,
    class_labels:np.ndarray,
) -> float:
    class_confusion_matrix = confusion_matrix(
        cast(ArrayLike, labels),
        cast(ArrayLike, predictions),
        labels=cast(ArrayLike, class_labels),
    )
    total_examples = class_confusion_matrix.sum()
    true_negatives = total_examples - (
        class_confusion_matrix.sum(axis=0)
        + class_confusion_matrix.sum(axis=1)
        - np.diag(class_confusion_matrix)
    )
    false_positives = class_confusion_matrix.sum(axis=0) - np.diag(class_confusion_matrix)
    class_specificities = np.divide(
        true_negatives,
        true_negatives + false_positives,
        out=np.zeros_like(true_negatives, dtype=float),
        where=(true_negatives + false_positives) != 0,
    )
    return float(class_specificities.mean())
