import torch
import torch_geometric as pyg
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.nn import MessagePassing
from cryograph.models.layers import MLPModule

class MLPEdgeMessage(MessagePassing):
    """Message passing as in MLPMessage, with a per-edge attribute concatenated onto
    each message.
    """
    def __init__(
            self, 
            node_in_channels: int, 
            node_out_channels: int,
            edge_attribute_channels: int,
            message_layers: int,
            message_hidden_channels: int,
            message_channels: int, 
            update_layers: int,
            update_hidden_channels: int,
            act: nn.Module,
        ):
        super().__init__(aggr='add')
        self.act = act
        self.message_nn = MLPModule(
            in_channels=2*node_in_channels+edge_attribute_channels,
            hidden_channels=message_hidden_channels,
            out_channels=message_channels,
            num_layers=message_layers,
            act=act,
        )
        self.update_nn = MLPModule(
            in_channels=node_in_channels+message_channels,
            hidden_channels=update_hidden_channels,
            out_channels=node_out_channels,
            num_layers=update_layers,
            act=act,
        )
    
        
    def forward(
            self,
            edge_index: Int[Tensor, "2 num_edges"], 
            h: Float[Tensor, "B num_nodes node_in_channels"],
            a: Float[Tensor, "B num_edges edge_attribute_channels"]
        ) -> Float[Tensor, "B num_nodes node_features_out"]:

        out = self.propagate(h=h, a=a, edge_index=edge_index) # normalize node degree?
        out = torch.cat((h, out), dim=-1)
        out = self.update_nn(out)

        return out # apply final linear?
    
    def message(
            self, 
            h_i: Float[Tensor, "B num_edges node_features_in"], 
            h_j: Float[Tensor, "B num_edges node_features_in"],
            a: Float[Tensor, "B num_edges edge_attribute_channels"]
        ):
        """Return the message of each edge as an MLP of its endpoint features and
        attribute.
        """
        m = torch.cat((h_i, h_j, a), dim=-1)
        m = self.message_nn(m)
        return m
    
    def reset_parameters(self):
        self.message_nn.reset_parameters()
        self.update_nn.reset_parameters()
