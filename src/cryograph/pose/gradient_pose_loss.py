import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float, Int
from torch import Tensor
from cryograph.pose import UnknownPoseLoss
from cryograph.renderer import Renderer
from cryograph.dataset import CryoDataset
from cryograph.utils import rotmats_to_quaternions

class GradientPoseLoss(UnknownPoseLoss):
    """Reconstruction loss with one free quaternion per micrograph, optimised jointly
    with the rest of the model.
    """

    def __init__(
            self, 
            num_mrcs: int,
            rot_grid_path: str,
            renderer: Renderer,
        ):
        super().__init__(rot_grid_path, renderer)

        self.register_parameter(
            "quats",
            nn.Parameter(torch.ones((num_mrcs, 4))/2.),
        )

        # self.quats = nn.Embedding.from_pretrained(init_quats)

    def forward(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            pred_confs: Float[Tensor, "B N 3"],
            latents: Float[Tensor, "B L"],
            idxs: Int[Tensor, "B"],
            rots: Float[Tensor, "B 1 3 3"] | None,
            shifts: Float[Tensor, "B 1 2"] | None,
        ) -> Float[Tensor, "B 1"]:
        return self.mse_mrc(mrcs, self.renderer(pred_confs, self.quats[idxs].unsqueeze(1), shifts))
    
    def opt_quats(
            self,
            mrcs: Float[Tensor, "B 1 R R"],
            base_mrc_grid: Float[Tensor, "1 Q R R"],
        ) -> Float[Tensor, "B 4"]:
        """Return the grid quaternion minimising image error for each micrograph."""
        loss_grid = self.mse_mrc(mrcs, base_mrc_grid.expand(mrcs.shape[0], -1, -1, -1))
        best_idxs = torch.argmin(loss_grid, dim=1)
        return rotmats_to_quaternions(self.rot_grid[best_idxs, :, :])
    
    @torch.no_grad()
    def init_quats(
            self,
            dataset: CryoDataset,
            protein_graph: pyg.data.Data,
            batch_size: int,
        ):
        """Initialise the per-micrograph quaternions by exhaustive search over the
        rotation grid, rendering the reference conformation once per grid point.
        """
        new_quats = torch.zeros_like(self.quats)
        base_conf = (
            protein_graph['coords'] - protein_graph['coords'].mean(dim=0)
        ).to(device=new_quats.device, dtype=new_quats.dtype)
        shifts = None
        base_mrc_grid = self.renderer(base_conf.unsqueeze(0), self.rot_grid.unsqueeze(0), shifts)

        num_chunks = len(dataset) // batch_size
        if len(dataset) % batch_size > 0:
            num_chunks += 1
        mrc_chunks = torch.chunk(dataset[:][1].unsqueeze(1), num_chunks)
        perc = 0
        for idx, mrc_chunk in enumerate(mrc_chunks):
            tmp = self.opt_quats(
                mrcs=mrc_chunk.to(device=new_quats.device, dtype=new_quats.dtype), 
                base_mrc_grid=base_mrc_grid,
            )
            num = idx * batch_size
            if idx + 1 == num_chunks:
                new_quats[num:, :] = tmp
                print("Initial quaternion grid search is at: ",
                    f"{perc}%",
                )
            elif idx % (num_chunks // 20) == 0 and perc != 100:
                new_quats[num:num+batch_size, :] = tmp
                print("Initial quaternion grid search is at: ",
                    f"{perc}%",
                )
                perc += 5
            else:
                new_quats[num:num+batch_size, :] = tmp
        self.quats = nn.Parameter(new_quats)
