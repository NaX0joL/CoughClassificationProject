from dataclasses import dataclass



@dataclass
class LeNetConfig:
    conv_channels: list[int]
    linear_dims: list[int]
    output_dim: int
    
    @classmethod
    def default(cls) -> "LeNetConfig":
        le_net_config = cls(
            conv_channels = [8, 16, 128],
            linear_dims = [640],
            output_dim = 2,
        )
        return le_net_config
