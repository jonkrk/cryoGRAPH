import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.renderer import Renderer
from cryograph.utils import quaternions_to_rotmats

class PoseLoss(nn.Module):
    """Base class for reconstruction losses. Returns the image mean squared error per
    sample and per pose.
    """

    def __init__(
            self, 
            renderer: Renderer,
        ):
        super().__init__()
        self.renderer = renderer 

    def forward(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            pred_confs: Float[Tensor, "B N 3"],
            latents: Float[Tensor, "B L"],
            idxs: Int[Tensor, "B"],
            rots: Float[Tensor, "B 1 3 3"] | None,
            shifts: Float[Tensor, "B 1 2"] | None,
        ) -> Float[Tensor, "B 1"]:
        """Return the image error of unposed conformations. Overridden by subclasses."""
        return self.mse_mrc(mrcs, self.renderer(pred_confs, None, None))

    def mse_mrc(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            pred_mrcs: Float[Tensor, "B Q R R"],
        ) -> Float[Tensor, "B Q"]:
        """Return the mean squared error between observed and rendered images."""
        return (pred_mrcs - mrcs).square().mean(dim=(2, 3))

class KnownPoseLoss(PoseLoss):
    """Reconstruction loss evaluated at supplied ground-truth poses."""

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

class UnknownPoseLoss(PoseLoss):
    """Base class for reconstruction losses that search over poses.

    Loads a fixed grid covering SO(3), given either as rotation matrices or as
    quaternions, which are converted on load.
    """

    def __init__(
            self, 
            rot_grid_path: str,
            renderer: Renderer,
        ):
        super().__init__(renderer)

        rot_grid = torch.load(rot_grid_path, weights_only=True)
        if rot_grid.ndim == 2 and rot_grid.shape[1] == 4:
            rot_grid = quaternions_to_rotmats(rot_grid)
        if rot_grid.ndim != 3 or rot_grid.shape[1:] != (3, 3):
            raise ValueError(f"UnknownPoseLoss found rotation grid at {rot_grid_path = } with invalid shape {rot_grid.shape = }")
        self.register_buffer(
            "rot_grid",
            rot_grid,
        )
    
    # def loss_grid(
    #     self,
    #     mrcs: Float[Tensor, "B 1 R R"],
    #     confs: Float[Tensor, "B N 3"],
    #     ) -> Float[Tensor, "B Q"]:
    #     shifts = None
    #     pred_mrcs = self.renderer(confs, self.rot_grid.unsqueeze(0), shifts)
    #     return self.mse_mrc(mrcs, pred_mrcs.expand(mrcs.shape[0], -1, -1, -1))
    
    # def best_rotmats(
    #     self,
    #     mrcs: Float[Tensor, "B 1 R R"],
    #     confs: Float[Tensor, "B N 3"],
    #     ) -> Float[Tensor, "B 1 3 3"]:
    #     loss_grid = self.loss_grid(mrcs, confs)
    #     best_idxs = torch.argmin(loss_grid, dim=1)
    #     return self.rot_grid[best_idxs]

# if __name__ == "__main__":
#     batchdim = 32
#     griddim = 137

#     loss_grid = torch.randn((batchdim, griddim))
#     rot_grid = torch.randn((griddim, 1, 3, 3))
#     best_idxs = torch.argmin(loss_grid, dim=1)
#     output = rot_grid[best_idxs]

#     print(output.shape)
