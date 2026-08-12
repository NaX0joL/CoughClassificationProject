from dataclasses import asdict, dataclass

from .abstract import ModelArchitecture, ModelBehavior
from .architectures.MLP import MLP, MLPConfig
from .behavior.classification_behavior import ClassificationBehavior



@dataclass
class ModelConfig:
    architecture:ModelArchitecture
    behavior:ModelBehavior

    @classmethod
    def default(cls) -> "ModelConfig":
        model_config = cls(
            architecture=MLP(**asdict(MLPConfig.default())),
            behavior=ClassificationBehavior(),
        )
        return model_config
