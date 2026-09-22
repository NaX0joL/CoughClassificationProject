from core.model import ModelConfig
from core.model.architectures.ResNet import ResNet
from core.model.behavior.classification_behavior import ClassificationBehavior



resnet_config = ModelConfig(
    architecture=ResNet(
        block_channels=[128, 256, 512],
        blocks_per_stage=2,
        output_dim=2,
    ),
    behavior=ClassificationBehavior(),
)
