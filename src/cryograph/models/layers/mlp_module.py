import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor

class MLPModule(nn.Module): 
    """Multilayer perceptron that accepts a leading batch dimension, which
    torch_geometric's MLP does not.
    """
    def __init__(
            self, 
            in_channels: int, 
            hidden_channels: int, 
            out_channels: int, 
            num_layers: int, 
            act: nn.Module,
        ): 
        super().__init__()
        self.in_channels = in_channels 
        self.act = act
        self.first_layer = pyg.nn.Linear(in_channels, hidden_channels)
        self.hidden_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.hidden_layers.append(pyg.nn.Linear(hidden_channels, hidden_channels))
        self.final_layer = pyg.nn.Linear(hidden_channels, out_channels)
    
    def forward(
            self, 
            x: Float[Tensor, "*batch in_channels"],
        ) -> Float[Tensor, "*batch out_channels"]:
        x = self.first_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        x = self.final_layer(x)
        return x

    def reset_parameters(self):
        self.first_layer.reset_parameters()
        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.final_layer.reset_parameters()
