from dataclasses import dataclass



@dataclass
class TemporalStatisticsMLPConfig:
    linear_dims: list[int]
    dropout: float
    output_dim: int

    @classmethod
    def default(cls) -> "TemporalStatisticsMLPConfig":
        temporal_statistics_mlp_config = cls(
            linear_dims = [256, 256, 256],
            dropout = 0.3,
            output_dim = 2,
        )
        return temporal_statistics_mlp_config
