import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn.conv import GCNConv
from jaxtyping import Float
from torch import Tensor
from cryograph.models.decoders import Decoder

class GCNConvDecoder(Decoder): 
    """GCNConv decoder that additionally regresses a global rotation from the latent
    and applies it to the output.

    Not registered in build_decoder and not exported by the package: unused.
    """
    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        feature_dim: int,
        layer_specs: list,
        embed_layers: int,
        act_name: str,
        dropout: float,
    ):
        super().__init__(protein_graph, latent_dim, act_name, dropout)
        self.register_buffer(
            "edge_index", 
            protein_graph['edge_index'],
        )
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.embed_list = nn.ModuleList([])
        for n in range(embed_layers):
            self.embed_list.append(nn.Linear(latent_dim, latent_dim))

        self.embed_final = nn.Linear(latent_dim, self.num_residues*feature_dim)
        self.unflatten = nn.Unflatten(1, (self.num_residues, feature_dim))
        
        self.messpass_list = nn.ModuleList([])
        node_channels = feature_dim
        for i in range(len(layer_specs)):
            spec = layer_specs[i]
            self.messpass_list.append(
                GCNConv(in_channels=node_channels,
                        out_channels=spec['node_out_channels'],
                        improved=True)
            )
            node_channels = spec['node_out_channels']
        self.unbed = nn.Linear(node_channels, 3)

        self.rot_list = nn.ModuleList([])
        for n in range(embed_layers):
            self.rot_list.append(nn.Linear(latent_dim, latent_dim))
        self.rot_list.append(nn.Linear(latent_dim, 6))
        

        self.reset_parameters()
    
    def forward(
            self,
            latents: Float[Tensor, "B L"]
        ) -> Float[Tensor, "B N 3"]:
        
        z = latents
            
        for layer in self.embed_list:
            z = layer(z)
            z = self.act(z)
        
        h = self.embed_final(z)
        h = self.unflatten(h)

        for n in range(len(self.messpass_list)):
            x = self.messpass_list[n](h, edge_index=self.edge_index)
            if n > 0 or self.latent_dim == self.feature_dim:
                h = self.act(x) + h
            else:
                h = self.act(x)

        output = self.unbed(h) + self.base_coords

        r = latents
        for layer in self.rot_list:
            r = layer(r)
            r = self.act(r)
        v1 = nn.functional.normalize(r[:,:,:3], dim=-1)
        v2 = r[:,:,3:]
        v2 = v2 - torch.linalg.vecdot(v1, v2) * v1
        v2 = nn.functional.normalize(v2, dim=-1)
        v3 = torch.linalg.cross(v1, v2)
        rot = torch.cat((v1.unsqueeze(-1), v2.unsqueeze(-1), v3.unsqueeze(-1)), dim=-1)
        output = torch.einsum('bij,bnj->bni', rot, output)
        return output

    def reset_parameters(self):
        for layer in self.embed_list:
            layer.reset_parameters()
        self.embed_final.reset_parameters()
        for layer in self.messpass_list:
            layer.reset_parameters()
        self.unbed.reset_parameters()
        for layer in self.rot_list:
            layer.reset_parameters()
