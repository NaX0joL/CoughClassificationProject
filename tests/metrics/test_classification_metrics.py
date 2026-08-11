import numpy as np
import pytest

from core.metrics import calculate_classification_metrics


def test_calculate_classification_metrics_for_binary_classification() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.1, 0.9],
    ])

    metrics = calculate_classification_metrics(labels, predictions, probabilities)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(0.8333333333)
    assert metrics.recall == pytest.approx(0.75)
    assert metrics.specificity == pytest.approx(0.75)
    assert metrics.f1_score == pytest.approx(0.7333333333)
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.macro_accuracy == pytest.approx(0.75)
    assert metrics.macro_f1_score == pytest.approx(0.7333333333)
    assert metrics.to_dict() == {
        "roc_auc": pytest.approx(1.0),
        "pr_auc": pytest.approx(1.0),
        "precision": pytest.approx(0.8333333333),
        "recall": pytest.approx(0.75),
        "specificity": pytest.approx(0.75),
        "f1_score": pytest.approx(0.7333333333),
        "accuracy": pytest.approx(0.75),
        "macro_accuracy": pytest.approx(0.75),
        "macro_f1_score": pytest.approx(0.7333333333),
    }


def test_calculate_classification_metrics_for_multiclass_classification() -> None:
    labels = np.array(["cold", "cold", "healthy", "healthy", "other", "other"])
    predictions = np.array(["cold", "healthy", "healthy", "healthy", "other", "other"])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.1, 0.8],
    ])
    class_labels = np.array(["cold", "healthy", "other"])

    metrics = calculate_classification_metrics(
        labels,
        predictions,
        probabilities,
        class_labels,
    )

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(5 / 6)
    assert metrics.macro_accuracy == pytest.approx(5 / 6)


def test_calculate_classification_metrics_requires_labels_for_missing_classes() -> None:
    labels = np.array([0, 1])
    predictions = np.array([0, 1])
    probabilities = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])

    with pytest.raises(ValueError, match="class_labels is required"):
        calculate_classification_metrics(labels, predictions, probabilities)
