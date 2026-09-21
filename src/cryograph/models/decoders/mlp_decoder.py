import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor
from cryograph.models.decoders import Decoder

class MLPDecoder(Decoder):
    """Decoder applying a multilayer perceptron to the latent, ignoring the residue
    graph.
    """
    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        num_layers: int,
        layer_width: int,
        act_name: str,
        dropout: float,
    ):
        super().__init__(protein_graph, latent_dim, act_name, dropout)

        num_residues = protein_graph['num_nodes']
        
        self.embed = nn.Linear(latent_dim, layer_width)

        self.linear_list = nn.ModuleList([])
        for n in range(num_layers):
            self.linear_list.append(nn.Linear(layer_width, layer_width))

        self.unbed = nn.Sequential(
            nn.Linear(layer_width, num_residues*3),
            nn.Unflatten(1, (num_residues, 3)),
        )

    def forward(
            self,
            latents: Float[Tensor, "B L"]
        ) -> Float[Tensor, "B N 3"]:

        h = self.embed(latents)
        h = self.act(h)

        for layer in self.linear_list:
            h = layer(h)
            h = self.act(h)

        output = self.unbed(h) + self.base_coords
        return output
