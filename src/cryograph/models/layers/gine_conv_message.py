import torch
import torch_geometric as pyg
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.nn import MessagePassing, GINEConv
from cryograph.models.layers import MLPModule

class GINEConvMessage(MessagePassing):
    """GINEConv wrapper carrying a leading batch dimension."""
    def __init__(
            self, 
            node_in_channels: int, 
            node_out_channels: int,
            edge_attribute_channels: int,
            update_layers: int,
            update_hidden_channels: int,
            act: nn.Module,
        ):
        super().__init__(aggr='add')
        self.act = act
        update_nn = MLPModule(
            in_channels=node_in_channels,
            hidden_channels=update_hidden_channels,
            out_channels=node_out_channels,
            num_layers=update_layers,
            act=act,
        )
        self.gine_conv = GINEConv(
            nn=update_nn, 
            edge_dim=edge_attribute_channels,
        )
        
    def forward(
            self,
            edge_index: Int[Tensor, "2 num_edges"], 
            h: Float[Tensor, "B num_nodes node_in_channels"],
            a: Float[Tensor, "B num_edges edge_attribute_channels"]
        ) -> Float[Tensor, "B num_nodes node_features_out"]:
        out = self.gine_conv(x=h, edge_index=edge_index, edge_attr=a)
        return out 
    
    def reset_parameters(self):
        self.gine_conv.reset_parameters()
