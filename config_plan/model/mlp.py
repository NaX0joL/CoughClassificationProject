from core.model import ModelConfig
from core.model.architectures.MLP import MLP
from core.model.behavior.classification_behavior import ClassificationBehavior



mlp_config = ModelConfig(
    architecture=MLP(
        linear_dims=[512, 512, 512],
        dropout=0.3,
        output_dim=2,
    ),
    behavior=ClassificationBehavior(),
)
