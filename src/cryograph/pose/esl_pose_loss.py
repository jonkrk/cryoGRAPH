import torch
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.pose import UnknownPoseLoss
from cryograph.renderer import Renderer
from cryograph.utils import axes_angles_to_rotmats, rotmats_to_axes_angles

class ESLPoseLoss(UnknownPoseLoss):
    """Reconstruction loss marginalised over poses by Ellipsoidal Support Lifting.

    The conformation is rendered at every point of a fixed SO(3) grid and the
    resulting image errors are reduced by a smoothed minimum, obtained by
    projecting them onto the simplex.
    """

    def __init__(
            self,
            eta: float,
            num_pts_base: int,
            num_iters: int,
            rot_grid_path: str,
            renderer: Renderer,
        ):
        if eta < 1./4. or eta > 2./3.: 
            raise ValueError(f"ESLPoseLoss needs eta within (1/4, 2/3), got {eta = }")

        super().__init__(rot_grid_path, renderer)

        self.register_buffer("ind", torch.arange(1, self.rot_grid.shape[0]+1))

        self.eta = eta
        self.J0 = num_pts_base
        self.J = min(int(self.J0 * self.ind[-1] ** ((2. - 3. * self.eta) / 5.)), self.ind[-1] - 1)
        self.num_iters = num_iters

    def forward(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            pred_confs: Float[Tensor, "B N 3"],
            latents: Float[Tensor, "B L"],
            idxs: Int[Tensor, "B"],
            rots: Float[Tensor, "B 1 3 3"] | None,
            shifts: Float[Tensor, "B 1 2"] | None,
        ) -> Float[Tensor, "B 1"]:
        """Return the pose-marginalised image error of each micrograph."""
        pred_mrcs = self.renderer(pred_confs, self.rot_grid.unsqueeze(0), shifts)
        loss_grid = self.mse_mrc(mrcs, pred_mrcs).unsqueeze(1)
        loss_vec, _ = self.compute_esl(loss_grid, skip_estimate=True)
        return loss_vec
    
    def multi_solve(
            self,
            mrcs: Float[Tensor, "B M R R"],
            confs: Float[Tensor, "B N 3"],
            rots: Float[Tensor, "B M 3 3"] | None,
            shifts: Float[Tensor, "B M 2"] | None,
            grid_size: int,
        ) -> tuple[Float[Tensor, "B M"], Float[Tensor, "B M 3 3"]]:
        """Score micrographs against the full rotation grid in chunks of grid_size, and
        return the reduced losses with their barycentric rotations.
        """
        pred_mrcs = self.renderer(confs, self.rot_grid.unsqueeze(0), shifts)
        loss_grid_list = []
        num_grids = mrcs.shape[1]//grid_size
        remainder = mrcs.shape[1]%grid_size
        for idx in range(num_grids):
            if idx%100 == 0:
                print(f"Loss grid computation at: {idx:>6}/{num_grids}")
            loss_grid_list.append((
                pred_mrcs[:, None, :, :, :] - mrcs[:, idx:idx+grid_size, None, :, :].to(pred_mrcs.device)
            ).square().mean(dim=(-2, -1)))
        if remainder > 0:
            loss_grid_list.append((
                pred_mrcs[:, None, :, :, :] - mrcs[:, -remainder:, None, :, :].to(pred_mrcs.device)
            ).square().mean(dim=(-2, -1)))
        loss_grid = torch.cat(loss_grid_list, dim=1)
        losses, bary_rots = self.compute_esl(loss_grid)
        # bary_rots = self.rot_grid[torch.argmin(loss_grid, dim=-1, keepdim=True).squeeze(-1), :, :]
        return losses, bary_rots
    
    def compute_esl(
            self, 
            loss_grid: Float[Tensor, "B M Q"], 
            skip_estimate=False,
        ) -> tuple[Float[Tensor, "B M"], Float[Tensor, "B M 3 3"] | None]:
        """Reduce a batch of per-grid-point losses to one loss per micrograph.

        With skip_estimate=False the barycentre of the resulting measure on SO(3) is
        also returned, refined by at most num_iters Riemannian averaging steps.
        """

        if loss_grid.shape[-1] != self.rot_grid.shape[0]:
            raise ValueError(
                "Shape of loss grid must match rot_grid, got ",
                f"{loss_grid.shape[-1] = }",
                f"{self.rot_grid.shape[0] = }",
            )

        # loss_grid_sorted, _ = torch.sort(loss_grid, dim=-1)
        loss_grid_bottomJ, _ = torch.topk(loss_grid, self.J, dim=-1, largest=False)
        # summed_losses = torch.sum(loss_grid_sorted[:, :, :self.J], dim=-1)
        summed_losses = torch.sum(loss_grid_bottomJ, dim=-1)

        gamma = 0.5 * self.J0 * loss_grid.shape[-1] ** ((2 + 2 * self.eta) / 5)
        # gamma *= loss_grid_sorted[:, :, self.J] - 1 / self.J * summed_losses
        gamma *= loss_grid_bottomJ[:, :, self.J-1] - 1 / self.J * summed_losses
        gamma += 1e-9
        # CONSIDER SPARSE REPRESENTATION FOR NONZEROS
        tmp = -1. * ((loss_grid.shape[-1] ** self.eta) / gamma[:, :, None]) * loss_grid 
        beta = self._project_to_simplex(tmp) 
        if skip_estimate:
            return torch.einsum('bmq, bmq -> bm', beta, loss_grid), None
        else:
            bary_rots = self.rot_grid[torch.argmax(beta, dim=-1), :, :]
            bary_rots_rel = torch.zeros_like(bary_rots)
            iter_esl = 0
            while torch.any(3. - torch.einsum("bmii", bary_rots_rel) > 1e-9) and iter_esl < self.num_iters:
                log_rots = rotmats_to_axes_angles(torch.einsum('bmji, qjk -> bmqik', bary_rots, self.rot_grid))
                bary_rots_rel = axes_angles_to_rotmats(torch.einsum('bmq, bmqi -> bmi', beta, log_rots))
                bary_rots = torch.einsum('bmij, bmjk -> bmik', bary_rots, bary_rots_rel)
                iter_esl += 1
            return torch.einsum('bmq, bmq -> bm', beta, loss_grid), bary_rots

    def _project_to_simplex(
            self, 
            V: Float[Tensor, "B M Q"],
            z: float = 1.,
        ) -> Float[Tensor, "B M Q"]:
        """Project onto the simplex scaled by z:
            P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
        """
        if V.shape[-1] != self.rot_grid.shape[0]:
            raise ValueError("Erroneus shape of input grid for projection to simplex.")
        
        # Sort V in descending order
        U, _ = torch.sort(V, descending=True, dim=-1)
        
        # Compute the cumulative sum of U and subtract z
        cssv = torch.cumsum(U, dim=-1) - z
        
        # Find the rho index
        cond = U - (cssv / self.ind) > 0
        _, rho_idx_tmp = cond.flip(dims=(-1,)).max(dim=-1, keepdim=True)
        rho_idx = self.rot_grid.shape[0] - rho_idx_tmp - 1

        # Compute the theta value
        theta = torch.take_along_dim(cssv, rho_idx, dim=-1)
        theta /= (rho_idx + 1)
        
        # Compute and return the projection
        return torch.clamp(V - theta, min=0., max=1.)
