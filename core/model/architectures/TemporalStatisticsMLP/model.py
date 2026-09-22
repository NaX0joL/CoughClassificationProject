import torch
from torch import nn, Tensor

from ...abstract import ModelArchitecture



class TemporalStatisticsMLP(ModelArchitecture):

    def __init__(
        self,
        linear_dims:list[int],
        dropout:float,
        output_dim:int,
    ) -> None:
        super().__init__()
        if not linear_dims:
            raise ValueError("linear_dims must contain at least one dimension")

        self.linear_dims = linear_dims
        self.dropout = dropout
        self.output_dim = output_dim

        layers = [
            nn.LazyLinear(out_features=linear_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]

        for index in range(len(linear_dims) - 1):
            layers.extend([
                nn.Linear(
                    in_features=linear_dims[index],
                    out_features=linear_dims[index + 1],
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])

        layers.append(nn.Linear(
            in_features=linear_dims[-1],
            out_features=output_dim,
        ))

        self.layers = nn.Sequential(*layers)
        return

    def forward(self, x:Tensor) -> Tensor:
        temporal_mean = x.mean(dim=1)
        temporal_std = x.std(dim=1, unbiased=False)
        x = torch.cat((temporal_mean, temporal_std), dim=1)
        x = self.layers(x)
        return x
