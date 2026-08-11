from dataclasses import dataclass



@dataclass
class MLPConfig:
    linear_dims: list[int]
    output_dim: int
    
    @classmethod
    def default(cls) -> "MLPConfig":
        mlp_config = cls(
            linear_dims = [256, 256, 256],
            output_dim = 2,
        )
        return mlp_config