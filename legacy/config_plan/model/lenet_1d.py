from core.model import ModelConfig
from core.model.architectures.LeNet1D import LeNet1D
from core.model.behavior.classification_behavior import ClassificationBehavior



lenet_1d_config = ModelConfig(
    architecture=LeNet1D(
        conv_channels=[32, 64, 128],
        linear_dims=[256, 256],
        dropout=0.3,
        output_dim=2,
    ),
    behavior=ClassificationBehavior(),
)
