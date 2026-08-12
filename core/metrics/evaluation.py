from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_model_device

from ..data_pipeline.dataset import ExampleDataset
from ..data_pipeline.intermediary import ORIGINAL_LABEL_METADATA_KEY
from ..model import FullModel
from .classification_metrics import ClassificationMetrics, ClassificationMetricsCalculator
from .metrics_config import MetricsConfig


BINARY_CLASS_NAMES = {
    0: "non-infectious",
    1: "infectious",
}


@dataclass(frozen=True)
class ModelEvaluation:
    metrics:ClassificationMetrics
    labels:np.ndarray
    predictions:np.ndarray
    class_names:dict[int, str]



class ModelEvaluator:
    """Evaluate a model using the metrics selected by its configuration."""

    def __init__(self, config:MetricsConfig) -> None:
        self.metrics_calculator = ClassificationMetricsCalculator(config.metrics)
        return

    def evaluate(
        self,
        model:FullModel,
        dataset:ExampleDataset,
        batch_size:int=32,
    ) -> ModelEvaluation:
        labels, predictions, probabilities, original_labels = self._collect_outputs(
            model,
            dataset,
            batch_size,
        )
        metrics = self.metrics_calculator.calculate(labels, predictions, probabilities)
        return ModelEvaluation(
            metrics=metrics,
            labels=labels,
            predictions=predictions,
            class_names=_resolve_class_names(labels, original_labels),
        )

    def _collect_outputs(
        self,
        model:FullModel,
        dataset:ExampleDataset,
        batch_size:int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        all_labels:list[np.ndarray] = []
        all_predictions:list[np.ndarray] = []
        all_probabilities:list[np.ndarray] = []
        all_original_labels:list[str] = []
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        device = get_model_device(model)
        was_training = model.training
        model.eval()

        try:
            with torch.inference_mode():
                for batch in loader:
                    values:Tensor = batch["value"].to(device)
                    labels:Tensor = batch["label"]
                    probabilities = model.predict_probabilities(values)
                    predictions = probabilities.argmax(dim=1)

                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
                    all_probabilities.append(probabilities.cpu().numpy())
                    all_original_labels.extend(_get_original_labels(batch))
        finally:
            model.train(was_training)

        if not all_labels:
            raise ValueError("dataset must contain at least one example")

        return (
            np.concatenate(all_labels),
            np.concatenate(all_predictions),
            np.concatenate(all_probabilities),
            all_original_labels,
        )


def _get_original_labels(batch:dict[str, object]) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("evaluation batch must include metadata")

    original_labels = metadata.get(ORIGINAL_LABEL_METADATA_KEY)
    if not isinstance(original_labels, (list, tuple)):
        raise ValueError("evaluation metadata must include original labels")

    return [str(label) for label in original_labels]


def _resolve_class_names(labels:np.ndarray, original_labels:list[str]) -> dict[int, str]:
    if len(labels) != len(original_labels):
        raise ValueError("labels and original labels must have equal lengths")

    class_names = dict(BINARY_CLASS_NAMES)
    for label, original_label in zip(labels, original_labels):
        numeric_label = int(label)
        class_names.setdefault(
            numeric_label,
            BINARY_CLASS_NAMES.get(numeric_label, original_label),
        )
    return class_names
