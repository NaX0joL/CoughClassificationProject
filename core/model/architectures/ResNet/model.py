from torch import Tensor, nn

from ...abstract import ModelArchitecture



class ResNet(ModelArchitecture):

    def __init__(
        self,
        block_channels: list[int],
        blocks_per_stage: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        if not block_channels:
            raise ValueError("block_channels must contain at least one value")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1")

        stages: list[nn.Module] = []
        current_channels = block_channels[0]

        for stage_index, output_channels in enumerate(block_channels):
            for block_index in range(blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(
                    ResidualBlock(current_channels, output_channels, stride)
                )
                current_channels = output_channels

        self.stem = nn.Sequential(
            nn.LazyConv1d(
                out_channels=block_channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(block_channels[0]),
            nn.ReLU(),
        )
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_channels, output_dim)
        return

    def forward(self, x:Tensor) -> Tensor:  # [batch, seq_len, channels]
        x = x.permute(0, 2, 1)              # [batch, channels, seq_len]

        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x




class ResidualBlock(nn.Module):

    def __init__(self, input_channels:int, output_channels:int, stride:int=1) -> None:
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
        )

        if input_channels != output_channels or stride != 1:
            self.skip_path = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
            )
        else:
            self.skip_path = nn.Identity()

        self.activation = nn.ReLU()
        return

    def forward(self, x:Tensor) -> Tensor:
        main = self.main_path(x)
        skip = self.skip_path(x)
        x = main + skip
        
        x = self.activation(x)
        return x
