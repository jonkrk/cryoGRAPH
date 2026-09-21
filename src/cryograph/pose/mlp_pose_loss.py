import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.renderer import Renderer
from cryograph.pose import PoseLoss
from cryograph.models.layers import MLPModule

class MLPPoseLoss(PoseLoss):
    """Reconstruction loss with the pose regressed from the latent by an MLP."""
    def __init__(
            self,
            num_layers_mlppose: int,
            num_hidden_mlppose: int,
            latent_dim: int,
            act_name: str,
            renderer: Renderer,
        ):
        super().__init__(renderer)

        if act_name == "relu":
            self.act = nn.ReLU()
        elif act_name == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError(
                f"The config lists an unimplemented decoder activation fct {act_name}",
            )

        self.pose_decoder = MLPModule(
            in_channels=latent_dim,
            hidden_channels=num_hidden_mlppose,
            out_channels=6,
            num_layers=num_layers_mlppose,
            act=self.act,
        )

    def forward(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            pred_confs: Float[Tensor, "B N 3"],
            latents: Float[Tensor, "B L"],
            idxs: Int[Tensor, "B"],
            rots: Float[Tensor, "B 1 3 3"] | None,
            shifts: Float[Tensor, "B 1 2"] | None,
        ) -> Float[Tensor, "B 1"]:
        v = self.pose_decoder(latents)
        pred_rots = self.rotmats_from_twovecs(v)
        if rots is None:
            tot_rots = pred_rots
        else:
            tot_rots = torch.einsum('bqij, bqjk -> bqik', pred_rots, rots)
        if shifts is not None:
            raise NotImplementedError("Unknown shift estimation not implemented yet")
        pred_mrcs = self.renderer(pred_confs, tot_rots, shifts)
        return self.mse_mrc(mrcs, pred_mrcs)
    
    def rotmats_from_twovecs(self, v): 
        """Convert 6-dimensional vectors to rotation matrices by Gram-Schmidt on the
        first two columns.
        """
        v1 = v[:,:3]
        v2 = v[:,3:]
        v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
        v2 = v2 - torch.einsum('bi,bi->b', v1, v2).unsqueeze(1) * v1
        v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
        v3 = torch.linalg.cross(v1, v2, dim=-1)
        rotmats = torch.cat((v1.unsqueeze(2), v2.unsqueeze(2), v3.unsqueeze(2)), dim=2)
        return rotmats.unsqueeze(1)

