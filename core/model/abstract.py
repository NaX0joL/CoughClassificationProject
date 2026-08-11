from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class StepResult:
    loss: Tensor
    logits: Tensor
    predictions: Tensor
    labels: Tensor


class ModelArchitecture(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Transform inputs into classification logits."""


class ModelBehavior(nn.Module, ABC):

    @abstractmethod
    def training_step(
        self,
        logits: Tensor,
        labels: Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        """Compute the task loss and outputs for a training batch."""

    @abstractmethod
    def validation_step(
        self,
        logits: Tensor,
        labels: Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        """Compute the task loss and outputs for a validation batch."""

    @abstractmethod
    def predict_probabilities(self, logits: Tensor) -> Tensor:
        """Convert logits into class probabilities."""

    @abstractmethod
    def predict_classes(self, logits: Tensor) -> Tensor:
        """Convert logits into predicted class indices."""
