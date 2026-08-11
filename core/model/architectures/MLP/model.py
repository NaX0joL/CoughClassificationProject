
from torch import nn, Tensor

from ...abstract import ModelArchitecture



class MLP(ModelArchitecture):
    
    def __init__(
        self, 
        linear_dims: list[int],
        output_dim:int,
    ) -> None:
        super().__init__()
        self.linear_dims = linear_dims
        self.output_dim = output_dim
        
        layers = [
            nn.Flatten(start_dim=1),
            nn.LazyLinear(out_features=linear_dims[0]),
            nn.ReLU(),
        ]
        
        for index in range(len(linear_dims) - 1):
            layers.extend([
                nn.Linear(
                    in_features=linear_dims[index], 
                    out_features=linear_dims[index + 1],
                ),
                nn.ReLU(),
            ])
            
        layers.append(nn.Linear(in_features=linear_dims[-1], out_features=output_dim))
        
        self.layers = nn.Sequential(*layers)
        return
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.layers(x)
        return x
