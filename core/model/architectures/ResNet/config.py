from dataclasses import dataclass



@dataclass
class ResNetConfig:
    block_channels: list[int]
    blocks_per_stage: int
    output_dim: int
    
    @classmethod
    def default(cls) -> "ResNetConfig":
        res_net_config = cls(
            block_channels = [128, 256, 512],
            blocks_per_stage = 2,
            output_dim = 2,
        )
        return res_net_config
