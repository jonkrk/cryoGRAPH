import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor
from cryograph.models.layers import MLPMessage
from cryograph.models.decoders import Decoder

class GNNDecoder(Decoder):
    """Decoder passing messages through MLPMessage layers."""
    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        feature_dim: int,
        embed_layers: int,
        messpass_layers: int,
        aggreg_sublayers: int, 
        message_sublayers: int,
        act_name: str,
        dropout: float,
    ):
        super().__init__(protein_graph, latent_dim, act_name, dropout)

        self.register_buffer(
            "edge_index", 
            protein_graph['edge_index']
        )
        self.embed_list = nn.ModuleList([])
        for n in range(embed_layers):
            self.embed_list.append(nn.Linear(latent_dim, latent_dim))

        self.embed_final = nn.Linear(latent_dim, self.num_residues*feature_dim)
        self.unflatten = nn.Unflatten(1, (self.num_residues, feature_dim))

        self.messpass_list = nn.ModuleList([])
        for n in range(messpass_layers): 
            self.messpass_list.append(
                MLPMessage(
                    node_features_in=feature_dim, 
                    node_features_out=feature_dim, 
                    num_layers_aggregate=aggreg_sublayers, 
                    num_layers_message=message_sublayers, 
                    act=self.act,
                )
            )
        self.unbed = nn.Linear(feature_dim, 3)

        self.reset_parameters()
    
    def forward(
            self,
            latents: Float[Tensor, "B latent_dim"]
        ) -> Float[Tensor, "B N 3"]:
        
        z = latents

        for layer in self.embed_list:
            z = layer(z)
            z = self.act(z)
        
        h = self.embed_final(z)
        h = self.unflatten(h)

        for n in range(len(self.messpass_list)):
            x = self.messpass_list[n](h=h, edge_index=self.edge_index)
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
