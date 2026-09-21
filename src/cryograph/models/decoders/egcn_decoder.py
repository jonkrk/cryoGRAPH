import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor
from cryograph.models.layers import EGCLMessage
from cryograph.models.decoders import Decoder

class EGCNDecoder(Decoder):
    """Decoder passing messages through E(n)-equivariant layers, which update the
    coordinates directly rather than through a final linear layer.
    """
    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        feature_dim: int,
        embed_layers: int,
        layer_specs: list,
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
        
        self.messpass_list = nn.ModuleList([])
        node_in_channels = feature_dim
        for i in range(len(layer_specs)):
            spec = layer_specs[i]
            self.messpass_list.append(
                EGCLMessage(
                    node_in_channels=node_in_channels, 
                    node_out_channels=spec['node_out_channels'], 
                    edge_embed_layers=spec['edge_embed_layers'], 
                    edge_embed_hidden_channels=spec['edge_embed_hidden_channels'],
                    edge_embed_channels=spec['edge_embed_channels'], 
                    message_layers=spec['message_layers'], 
                    message_hidden_channels=spec['message_hidden_channels'], 
                    update_layers=spec['update_layers'], 
                    update_hidden_channels=spec['update_hidden_channels'], 
                    act=self.act,
                )
            )
            node_in_channels = spec['node_out_channels']

        self.reset_parameters()
    
    def forward(
            self,
            latents: Float[Tensor, "B L"]
        ) -> Float[Tensor, "B N 3"]:
        x = self.base_coords.repeat(latents.shape[0], 1, 1) 
        z = latents

        for layer in self.embed_list:
            z = layer(z)
            z = self.act(z)
        
        h = self.embed_final(z)
        h = self.unflatten(h)

        for n in range(len(self.messpass_list)):
            ht, x = self.messpass_list[n](h, x, self.edge_index)
            h = self.act(ht) + h

        output = x
        return output
    
    def reset_parameters(self):
        for layer in self.embed_list:
            layer.reset_parameters()
        self.embed_final.reset_parameters()
        for layer in self.messpass_list:
            layer.reset_parameters()
