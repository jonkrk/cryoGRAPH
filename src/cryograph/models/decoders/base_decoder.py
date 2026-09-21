import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor

class Decoder(nn.Module):
    """Base class for conformation decoders. Maps a latent code to residue
    coordinates, expressed as a displacement of the mean-centred reference
    conformation taken from the protein graph.
    """

    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        act_name: str,
        dropout: float,
        # use_pose_decoder: bool
    ):
        super().__init__()

        self.num_residues = protein_graph['num_nodes'] 
        self.num_edges = protein_graph.num_edges
        self.latent_dim = latent_dim
        # self.use_pose_decoder = use_pose_decoder

        if act_name == "relu":
            self.act = nn.ReLU()
        elif act_name == "tanh":
            self.act = nn.Tanh()
        elif act_name == "silu":
            self.act = nn.SiLU()
        else:
            raise NotImplementedError(
                f"The config lists an unimplemented decoder activation fct {act_name}",
            )
        
        if dropout >= 1.0 or dropout < 0.0:
            raise ValueError(
                f"The config lists an unreasonable dropout frequency: {dropout}"
            )
        elif dropout > 0.0:
            self.act = nn.Sequential(
                self.act,
                nn.Dropout(dropout),
            )
        
        self.register_buffer(
            "base_coords", 
            protein_graph['coords']-protein_graph['coords'].mean(dim=0)
        )
    
    def forward(
            self,
            latents: Float[Tensor, "B L"]
        ) -> Float[Tensor, "B N 3"]:
        """Return residue coordinates. Concrete decoders override this."""
        return torch.zeros(size=(latents.shape[0], self.num_residues, 3))