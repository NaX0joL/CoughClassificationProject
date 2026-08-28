from core.model import ModelConfig
from core.model.architectures.LeNet import LeNet
from core.model.behavior.classification_behavior import ClassificationBehavior



lenet_config = ModelConfig(
    architecture=LeNet(
        conv_channels=[32, 64, 128],
        linear_dims=[256, 256],
        dropout=0.3,
        output_dim=2,
    ),
    behavior=ClassificationBehavior(),
)
