import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.models.encoders import Encoder

class CNNEncoder(Encoder):
    """Encoder mapping a micrograph to a latent vector through convolution and
    pooling blocks, halving the resolution until it is small enough to flatten.
    """
    def __init__(
            self,
            latent_dim: int,
            sidelength: int, 
            conv_channels: int,
            act_name: str,
        ):
        super().__init__(latent_dim, act_name)

        self.convpool_init = nn.Sequential(
            nn.Conv2d(1, conv_channels, 3, padding=1),
            self.act,
            # nn.Conv2d(conv_channels, conv_channels, 3, padding=1),
            # self.act,
            nn.MaxPool2d(2),
        )

        self.convpool_list = nn.ModuleList([])
        N = sidelength // 2
        while (N//2)**2 >= latent_dim and (N//2)**2 >= 2**8:
            self.convpool_list.append(
                nn.Sequential(
                    nn.Conv2d(conv_channels, conv_channels, 3, padding=1),
                    self.act,
                    # nn.Conv2d(conv_channels, conv_channels, 3, padding=1),
                    # self.act,
                    nn.MaxPool2d(2),
                )
            )
            N = N // 2
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(conv_channels, 1, 3, padding=1),
            self.act,
            nn.Flatten(1, -1),
            nn.Linear(N**2, latent_dim)
        )

    def forward(
            self, 
            idxs: Int[Tensor, "B"], 
            mrcs: Float[Tensor, "B 1 R R"],
        ) -> Float[Tensor, "B L"]:
        h = self.convpool_init(mrcs)
        for layer in self.convpool_list:
            h = layer(h)
        out = self.conv_final(h)
        return out
