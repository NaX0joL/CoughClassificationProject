from torch import Tensor, nn

from .abstract import ModelArchitecture, ModelBehavior, StepResult



class FullModel(nn.Module):
    
    def __init__(
        self,
        architecture: ModelArchitecture,
        behavior: ModelBehavior,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.behavior = behavior
        return

    def forward(self, values:Tensor) -> Tensor:
        return self.architecture(values)

    def training_step(
        self,
        batch:dict[str, Tensor],
        criterion:nn.Module,
    ) -> StepResult:
        values = batch["value"]
        labels = batch["label"]
        
        logits = self(values)
        step_result = self.behavior.training_step(logits, labels, criterion)
        return step_result

    def validation_step(
        self,
        batch:dict[str, Tensor],
        criterion:nn.Module,
    ) -> StepResult:
        values = batch["value"]
        labels = batch["label"]
        
        logits = self(values)
        step_result = self.behavior.validation_step(logits, labels, criterion)
        return step_result

    def predict_probabilities(self, values:Tensor) -> Tensor:
        logits = self(values)
        return self.behavior.predict_probabilities(logits)

    def predict_classes(self, values:Tensor) -> Tensor:
        logits = self(values)
        return self.behavior.predict_classes(logits)
