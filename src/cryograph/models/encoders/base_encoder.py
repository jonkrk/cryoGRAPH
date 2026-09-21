import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

class Encoder(nn.Module):
    """Base class for encoders. Maps a batch of micrographs to latent conformation
    codes.
    """
    def __init__(
            self,
            latent_dim: int,
            act_name = "",
        ):
        super().__init__()
        self.latent_dim = latent_dim

        if act_name == "":
            self.act = None
        elif act_name == "relu":
            self.act = nn.ReLU()
        elif act_name == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError(
                f"The config lists an unimplemented encoder activation fct: {act_name}"
            )

    def forward(
            self, 
            idxs: Int[Tensor, "B"], 
            mrcs: Float[Tensor, "B 1 R R"],
        ) -> Float[Tensor, "B L"]:
        """Return zero latents. Concrete encoders override this."""
        return torch.zeros((idxs.shape[0], self.latent_dim))