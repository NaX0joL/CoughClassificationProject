from torch import Tensor, nn

from ..abstract import ModelBehavior, StepResult


class ClassificationBehavior(ModelBehavior):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        return

    def training_step(
        self,
        logits: Tensor,
        labels: Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        return self._step(logits, labels, criterion)

    def validation_step(
        self,
        logits: Tensor,
        labels: Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        return self._step(logits, labels, criterion)

    def predict_probabilities(self, logits:Tensor) -> Tensor:
        return logits.softmax(dim=1)

    def predict_classes(self, logits:Tensor) -> Tensor:
        return logits.argmax(dim=1)

    def _step(
        self,
        logits: Tensor,
        labels: Tensor,
        criterion: nn.Module,
    ) -> StepResult:
        return StepResult(
            loss=criterion(logits, labels),
            logits=logits,
            predictions=self.predict_classes(logits),
            labels=labels,
        )
