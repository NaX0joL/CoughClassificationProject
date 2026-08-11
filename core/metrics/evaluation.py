import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_model_device

from ..data_pipeline.dataset import ExampleDataset
from ..model import FullModel
from .classification_metrics import ClassificationMetrics, ClassificationMetricsCalculator
from .metrics_config import MetricsConfig



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
    ) -> ClassificationMetrics:
        labels, predictions, probabilities = self._collect_outputs(
            model,
            dataset,
            batch_size,
        )
        return self.metrics_calculator.calculate(labels, predictions, probabilities)

    def _collect_outputs(
        self,
        model:FullModel,
        dataset:ExampleDataset,
        batch_size:int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_labels:list[np.ndarray] = []
        all_predictions:list[np.ndarray] = []
        all_probabilities:list[np.ndarray] = []
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
        finally:
            model.train(was_training)

        if not all_labels:
            raise ValueError("dataset must contain at least one example")

        return (
            np.concatenate(all_labels),
            np.concatenate(all_predictions),
            np.concatenate(all_probabilities),
        )
