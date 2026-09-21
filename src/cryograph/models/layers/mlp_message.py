import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.nn import MessagePassing

class MLPMessage(MessagePassing):
    """Message passing in which the message is an MLP of the two endpoint features
    and the node update is an MLP of the node and its summed messages.
    """
    def __init__(
        self, 
        node_features_in: int, 
        node_features_out: int, 
        num_layers_aggregate: int, 
        num_layers_message: int, 
        act: nn.Module,
    ):
        super().__init__(aggr='add')
        self.node_features_in = node_features_in
        self.node_features_out = node_features_out
        self.num_layers_aggregate = num_layers_aggregate
        self.num_layers_message = num_layers_message
        self.act = act

        self.message_layers = nn.ModuleList([])
        self.aggregate_layers = nn.ModuleList([])

        self.message_layers.append(
            nn.Linear(2*node_features_in, node_features_in),
        )
        for i in range(num_layers_message):
            self.message_layers.append(
                nn.Linear(node_features_in, node_features_in),
            ) 
        
        self.aggregate_layers.append(
            nn.Linear(2*node_features_in, node_features_out),
        )
        for i in range(num_layers_aggregate):
            self.aggregate_layers.append(
                nn.Linear(node_features_out, node_features_out),
            )
        
    def forward(
            self,
            edge_index: Int[Tensor, "2 num_edges"], 
            h: Float[Tensor, "B num_nodes node_features_in"],
        ) -> Float[Tensor, "B num_nodes node_features_out"]:
        out = self.propagate(h=h, edge_index=edge_index) 
        out = torch.cat((h, out), dim=-1)
        
        for layer in self.aggregate_layers:
            out = layer(out)
            out = self.act(out)
        
        return out 
    
    def message(
            self, 
            h_i: Float[Tensor, "B num_edges node_features_in"], 
            h_j: Float[Tensor, "B num_edges node_features_in"],
        ):
        """Return the message of each edge as an MLP of its endpoint features."""
        m = torch.cat((h_i, h_j), dim=-1)

        for layer in self.message_layers:
            m = layer(m) 
            m = self.act(m)
        return m
    
    def reset_parameters(self):
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.aggregate_layers:
            layer.reset_parameters()
