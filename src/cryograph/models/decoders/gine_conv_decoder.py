import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor
from cryograph.models.layers import GINEConvMessage
from cryograph.models.decoders import Decoder

class GINEConvDecoder(Decoder):
    """Decoder passing messages through GINEConv layers, with a learned attribute
    per edge.
    """
    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        feature_dim: int,
        layer_specs: list,
        edge_attribute_channels: int,
        embed_layers: int,
        act_name: str,
        dropout: float,
    ):
        super().__init__(protein_graph, latent_dim, act_name, dropout)
        self.register_buffer(
            "edge_index", 
            protein_graph['edge_index'],
        )
        
        self.embed_list = nn.ModuleList([])
        for n in range(embed_layers):
            self.embed_list.append(nn.Linear(latent_dim, latent_dim))

        self.embed_final = nn.Linear(latent_dim, self.num_residues*feature_dim)
        self.unflatten = nn.Unflatten(1, (self.num_residues, feature_dim))

        self.edge_attributes = torch.nn.parameter.Parameter(
            torch.zeros(self.num_edges, edge_attribute_channels),
        )
        
        self.messpass_list = nn.ModuleList([])
        node_channels = feature_dim
        for i in range(len(layer_specs)):
            spec = layer_specs[i]
            self.messpass_list.append(
                GINEConvMessage(
                    node_in_channels=node_channels,
                    node_out_channels=spec['node_out_channels'],
                    edge_attribute_channels=edge_attribute_channels,
                    update_layers=spec['update_layers'],
                    update_hidden_channels=spec['update_hidden_channels'],
                    act=self.act,
                ),
            )
            node_channels = spec['node_out_channels']
        self.unbed = nn.Linear(node_channels, 3)

        self.reset_parameters()
    
    def forward(
            self,
            latents: Float[Tensor, "B L"]
        ) -> Float[Tensor, "B N 3"]:
        
        z = latents
        a = self.edge_attributes.expand(
            z.shape[0], 
            self.edge_attributes.shape[0], 
            self.edge_attributes.shape[1],
        ) 
        for layer in self.embed_list:
            z = layer(z)
            z = self.act(z)
        
        h = self.embed_final(z)
        h = self.unflatten(h)

        for n in range(len(self.messpass_list)):
            x = self.messpass_list[n](h=h, a=a, edge_index=self.edge_index)
            h = self.act(x) + h

        output = self.unbed(h) + self.base_coords
        return output

    def reset_parameters(self):
        for layer in self.embed_list:
            layer.reset_parameters()
        self.embed_final.reset_parameters()
        for layer in self.messpass_list:
            layer.reset_parameters()
        self.unbed.reset_parameters()
