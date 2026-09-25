from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_model_device

from .classification_metrics import ClassificationMetrics, ClassificationMetricsCalculator
from .label_mapping import BinaryInfectionLabelMapping
from .metrics_config import MetricsConfig

from core.model.full_model import FullModel



@dataclass(frozen=True)
class ModelEvaluation:
    metrics:ClassificationMetrics
    labels:np.ndarray
    predictions:np.ndarray
    probabilities:np.ndarray
    reported_labels:np.ndarray
    reported_predictions:np.ndarray
    reported_probabilities:np.ndarray
    reported_class_names:tuple[str, str]



class ModelEvaluator:

    def __init__(
        self,
        config:MetricsConfig,
        label_mapping:BinaryInfectionLabelMapping|None=None,
    ) -> None:
        self.metrics_calculator = ClassificationMetricsCalculator(config.metrics)
        self.label_mapping = label_mapping or BinaryInfectionLabelMapping()
        return

    def evaluate(
        self,
        model:FullModel,
        data_loader:DataLoader,
    ) -> ModelEvaluation:
        device = get_model_device(model)
        was_training = model.training
        all_labels:list[np.ndarray] = []
        all_predictions:list[np.ndarray] = []
        all_probabilities:list[np.ndarray] = []
        all_reported_predictions:list[np.ndarray] = []
        all_reported_probabilities:list[np.ndarray] = []
        probability_width:int|None = None
        model.eval()

        try:
            with torch.inference_mode():
                for batch in data_loader:
                    values:Tensor = batch["value"].to(device)
                    labels:Tensor = batch["label"]
                    probabilities = model.predict_probabilities(values)
                    if probabilities.ndim != 2:
                        raise ValueError(
                            "model probabilities must be a two-dimensional tensor",
                        )

                    current_width = probabilities.shape[1]
                    if probability_width is None:
                        probability_width = current_width
                    elif current_width != probability_width:
                        raise ValueError(
                            "model probability width must be consistent across batches",
                        )

                    raw_probability_array = probabilities.cpu().numpy()
                    reported_probability_array = self.label_mapping.aggregate_probabilities(
                        raw_probability_array,
                    )
                    predictions = probabilities.argmax(dim=1)
                    reported_predictions = reported_probability_array.argmax(axis=1)
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
                    all_probabilities.append(raw_probability_array)
                    all_reported_predictions.append(reported_predictions)
                    all_reported_probabilities.append(reported_probability_array)
        finally:
            model.train(was_training)

        if not all_labels or probability_width is None:
            raise ValueError("data loader must contain at least one batch")

        labels = np.concatenate(all_labels)
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        reported_labels = self.label_mapping.remap_labels(labels)
        reported_predictions = np.concatenate(all_reported_predictions)
        reported_probabilities = np.concatenate(all_reported_probabilities)
        class_labels = np.arange(len(self.label_mapping.report_class_names))
        metrics = self.metrics_calculator.calculate(
            reported_labels,
            reported_predictions,
            reported_probabilities,
            class_labels,
        )
        return ModelEvaluation(
            metrics=metrics,
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            reported_labels=reported_labels,
            reported_predictions=reported_predictions,
            reported_probabilities=reported_probabilities,
            reported_class_names=self.label_mapping.report_class_names,
        )
