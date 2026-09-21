import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.renderer import Renderer
from cryograph.pose import PoseLoss

class KnownPoseLoss(PoseLoss):
    """Reconstruction loss evaluated at supplied ground-truth poses.

    Shadowed by the identical class in base_pose_loss, which is what the package
    exports; this module is unused.
    """

    def __init__(
            self,
            renderer: Renderer,
        ):
        super().__init__(renderer)

    def forward(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            pred_confs: Float[Tensor, "B N 3"],
            latents: Float[Tensor, "B L"],
            idxs: Int[Tensor, "B"],
            rots: Float[Tensor, "B 1 3 3"],
            shifts: Float[Tensor, "B 1 2"] | None,
        ) -> Float[Tensor, "B 1"]:
        pred_mrcs = self.renderer(pred_confs, rots, shifts)
        return self.mse_mrc(mrcs, pred_mrcs)
