import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.nn import MessagePassing
from cryograph.models.layers import MLPModule

class EGCLMessage(MessagePassing):
    """E(n)-equivariant graph convolution layer.

    Messages depend on the endpoint features and their squared distance, and
    coordinates are updated along the edge difference vectors, so the layer is
    equivariant to rotation and translation of the input coordinates.
    """
    def __init__(
        self, 
        node_in_channels: int, 
        node_out_channels: int, 
        edge_embed_layers: int, 
        edge_embed_hidden_channels: int, 
        edge_embed_channels: int, 
        message_layers: int, 
        message_hidden_channels: int, 
        update_layers: int, 
        update_hidden_channels: int, 
        act: nn.Module,
    ):
        super().__init__(aggr='add')

        self.edge_embed_nn = MLPModule(
            in_channels=2*node_in_channels+1, 
            hidden_channels=edge_embed_hidden_channels, 
            out_channels=edge_embed_channels, 
            num_layers=edge_embed_layers, 
            act=act,
        )
        self.message_nn = MLPModule(
            in_channels=edge_embed_channels, 
            hidden_channels=message_hidden_channels, 
            out_channels=1, 
            num_layers=message_layers, 
            act=act,
        )
        self.update_nn = MLPModule(
            in_channels=node_in_channels+edge_embed_channels, 
            hidden_channels=update_hidden_channels, 
            out_channels=node_out_channels, 
            num_layers=update_layers, 
            act=act
        )

    def forward(
            self, 
            h: Float[Tensor, "B num_nodes node_in_channels"], 
            x: Float[Tensor, "B num_nodes 3"], 
            edge_index: Int[Tensor, "2 num_edges"],
        ):
        norm = 1


        out = self.propagate(edge_index, h=h, x=x)
        xshift = out[:,:,-3:] 
        message = out[:,:,:-3]
        outx = x + norm * xshift
        outh = self.update_nn(torch.cat((h, message), dim=-1))
        return outh, outx

    def message(
            self, 
            h_i: Float[Tensor, "B num_edges node_in_channels"], 
            h_j: Float[Tensor, "B num_edges node_in_channels"], 
            x_i: Float[Tensor, "B num_edges 3"], 
            x_j: Float[Tensor, "B num_edges 3"],
        ):
        """Return the edge embedding with the coordinate update it induces."""
        dist = torch.sum(torch.square(x_i - x_j), dim=-1, keepdim=True)
        edge_embed = self.edge_embed_nn(torch.cat((h_i, h_j, dist), dim=-1))
        xmess = (x_i - x_j) * self.message_nn(edge_embed) 
        return torch.cat((edge_embed, xmess), dim=-1) 
    
    def reset_parameters(self):
        self.edge_embed_nn.reset_parameters()
        self.message_nn.reset_parameters()
        self.update_nn.reset_parameters()
