from copy import deepcopy

from torch import Tensor, nn

from .abstract import ModelArchitecture, ModelBehavior, StepResult
from .model_config import ModelConfig



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

    @classmethod
    def create(cls, config:ModelConfig) -> "FullModel":
        model = cls(
            architecture=deepcopy(config.architecture),
            behavior=deepcopy(config.behavior),
        )
        return model

    @classmethod
    def create_from_state_dict(
        cls,
        config:ModelConfig,
        state_dict:dict[str, Tensor],
    ) -> "FullModel":
        model = cls.create(config)
        model.load_state_dict(state_dict)
        return model

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
