import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.models.encoders import Encoder

class LookupEncoder(Encoder):
    """Encoder holding one free latent vector per micrograph, indexed by position in
    the dataset.

    The latents are parameters rather than a function of the image, so there is no
    way to embed a micrograph the model was not trained on.
    """
    def __init__(
            self,
            latent_dim: int,
            num_mrcs: int,
        ):
        super().__init__(latent_dim)

        self.table = nn.Embedding(
            num_embeddings=num_mrcs,
            embedding_dim=latent_dim,
        )
    
    def forward(
            self, 
            idxs: Int[Tensor, "B"], 
            mrcs: Float[Tensor, "B 1 R R"],
        ) -> Float[Tensor, "B L"]:
        return self.table(idxs)