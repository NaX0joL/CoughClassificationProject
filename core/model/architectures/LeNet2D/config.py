from dataclasses import dataclass



@dataclass
class LeNet2DConfig:
    conv_channels:list[int]
    linear_dims:list[int]
    dropout:float
    output_dim:int

    @classmethod
    def default(cls) -> "LeNet2DConfig":
        le_net_2d_config = cls(
            conv_channels=[8, 16, 128],
            linear_dims=[640],
            dropout=0.3,
            output_dim=2,
        )
        return le_net_2d_config
