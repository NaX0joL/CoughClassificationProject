
from torch import nn, Tensor

from ...abstract import ModelArchitecture



class LeNet1D(ModelArchitecture):
    
    def __init__(
        self, 
        conv_channels:list[int],
        linear_dims:list[int],
        dropout:float,
        output_dim:int,
    ) -> None:
        super().__init__()
        self.conv_channels = conv_channels
        self.linear_dims = linear_dims
        self.dropout = dropout
        self.output_dim = output_dim
        
        conv_layers = [
            nn.LazyConv1d(
                out_channels=conv_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout1d(p=dropout),
        ]
        
        for index in range(len(conv_channels) - 1):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=conv_channels[index],
                    out_channels=conv_channels[index + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Dropout1d(p=dropout),
            ])
            
        linear_layers = [
            nn.Flatten(),
            nn.LazyLinear(out_features=linear_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        
        for index in range(len(linear_dims) - 1):
            linear_layers.extend([
                nn.Linear(
                    in_features=linear_dims[index], 
                    out_features=linear_dims[index + 1],
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            
        linear_layers.append(nn.Linear(
            in_features=linear_dims[-1], 
            out_features=output_dim,
        ))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear_layers = nn.Sequential(*linear_layers)
        return
    
    def forward(self, x:Tensor) -> Tensor:  # [batch, seq_len, channels]
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)              # [batch, channels, seq_len]
        
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x
