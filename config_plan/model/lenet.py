from core.model import ModelConfig
from core.model.architectures.LeNet import LeNet
from core.model.behavior.classification_behavior import ClassificationBehavior



lenet_config = ModelConfig(
    architecture=LeNet(
        conv_channels=[8, 16, 128],
        linear_dims=[640],
        output_dim=2,
    ),
    behavior=ClassificationBehavior(),
)
