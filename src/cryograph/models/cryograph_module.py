import wandb
import torch
import torch_geometric as pyg
from pytorch_lightning import LightningModule
from jaxtyping import Float, Int
from torch import Tensor
from matplotlib import pyplot as plt
import plotly

from cryograph.models.encoders.builder import build_encoder
from cryograph.models.decoders.builder import build_decoder
from cryograph.pose.builder import build_pose_loss
from cryograph.renderer.builder import build_renderer
from cryograph.utils import build_pair_indices, build_prior_sqrlogs, kabsch_torch

class CryoGRAPH(LightningModule):
    """Autoencoder over protein conformations.

    An encoder maps each micrograph to a latent code, a graph decoder maps the code
    to residue coordinates, and a differentiable renderer projects those
    coordinates so they can be compared against the observed micrograph. Pose is
    supplied or inferred by the configured pose loss.
    """
    def __init__(
        self,
        config: dict,
        protein_graph: pyg.data.Data,
        elec_vec: Float[Tensor, "N"],
        std_vec: Float[Tensor, "N"],
        num_mrcs: int,
        mrc_sidelength: int
    ):
        super().__init__()
        # keep full config for reproducibility (ignoring huge tensors)
        self.save_hyperparameters({
            "config": config,
            "num_mrcs": num_mrcs,
            "mrc_sidelength": mrc_sidelength,
        })

        data  = config['data']
        reg   = config['regulariser']
        train = config['training']
        img   = config['rendering']['image']

        self.num_mrcs = num_mrcs
        self.mrc_sidelength = mrc_sidelength

        self.load_rots   = data['load_rots']
        self.load_shifts = data['load_shifts']
        self.load_confs  = data['load_confs']

        self.learn_proj_params = img['learn_proj_params']

        self.loss_dist_fac         = reg['dist']['loss_fac']
        self.loss_cent_fac         = reg['cent']['loss_fac']
        self.loss_geom_fac         = reg['geom']['loss_fac']
        self.loss_local_geom_fac   = reg['local_geom']['loss_fac']
        self.loss_local_geom_decay = reg['local_geom']['decay']
        self.loss_local_geom_size  = reg['local_geom']['size']
        self.true_residue_dist     = reg['dist']['ref']

        self.optimizer_name      = train['optimizer']
        self.learning_rate       = train['lr']
        self.lr_scheduler_name   = train['lr_scheduler']
        self.lr_initial_factor   = train['lr_initial_factor']
        self.lr_epochs_until_max = train['lr_epochs_until_max']

        # self.logger.log_table(
        #     key = "log/protein_graph", 
        #     columns = ["Nodes", "Edges"],
        #     data = [[int(protein_graph['num_nodes'])], [int(protein_graph['num_edges'])]],
        # )

        self.encoder = build_encoder(config, num_mrcs=num_mrcs, mrc_sidelength=mrc_sidelength)
        self.decoder = build_decoder(config, protein_graph=protein_graph)
        renderer = build_renderer(config, protein_graph=protein_graph,
                                  elec_vec=elec_vec, std_vec=std_vec,
                                  mrc_sidelength=mrc_sidelength)
        self.pose_loss = build_pose_loss(config, load_rots=self.load_rots,
                                         load_shifts=self.load_shifts,
                                         num_mrcs=num_mrcs, renderer=renderer)

        self._register_graph_buffers(protein_graph)

        self.rmsd_list = []

    def _register_graph_buffers(self, protein_graph: pyg.data.Data):
        """Register the index pairs and reference distances used by the geometric
        regularisers.
        """
        num_nodes = int(protein_graph['num_nodes'])
        coords = protein_graph['coords']  # [N,3], stays on CPU; buffers will follow module.device

        i_idx, j_idx, local_i, local_j = build_pair_indices(
            num_nodes=num_nodes, 
            local_size=self.loss_local_geom_size, 
            device=coords.device
        )
        self.register_buffer("i_idx", i_idx, persistent=False)
        self.register_buffer("j_idx", j_idx, persistent=False)
        self.register_buffer("local_i_idx", local_i, persistent=False)
        self.register_buffer("local_j_idx", local_j, persistent=False)

        prior_global = build_prior_sqrlogs(coords, i_idx, j_idx)
        prior_local  = build_prior_sqrlogs(coords, local_i, local_j)
        self.register_buffer("prior_edm_sq_log", prior_global, persistent=False)
        self.register_buffer("local_prior_edm_sq_log", prior_local, persistent=False)
    
    def forward(
            self, 
            idxs: Int[Tensor, "B"],
            mrcs: Float[Tensor, "B 1 R R"],
        ) -> tuple[Float[Tensor, "B N 3"], Float[Tensor, "B L"]]:
        """Return the predicted conformations and the latent codes they came from."""
        latents = self.encoder(idxs, mrcs) # output: [B, L]
        pred_confs = self.decoder(latents) # output: [B, N, 3]
        return pred_confs, latents

    def training_step(self, batch, batch_idx):
        """Return the reconstruction loss and every nonzero-weighted regulariser."""
        idxs, mrcs, rots, shifts, _ = self.batch_unpacker(batch)
        
        pred_confs, latents = self.forward(idxs, mrcs)

        loss_mrc = self.mrc_reconstruction_loss(mrcs, pred_confs, latents, idxs, rots, shifts)
        self.log("train/loss_mrc", loss_mrc)

        loss = loss_mrc

        for name, fac, loss_func in (
            ("dist", self.loss_dist_fac, self.residue_distance_loss),
            ("cent", self.loss_cent_fac, self.centering_loss),
            ("geom", self.loss_geom_fac, self.geometric_loss),
            ("local_geom", self.loss_local_geom_fac, self.local_geometric_loss),
        ):
            if fac == 0.:
                continue
            loss_reg = loss_func(pred_confs)
            self.log(f"train/loss_{name}", loss_reg)
            loss = loss + fac * loss_reg

        self.log("train/loss", loss)

        if self.learn_proj_params:
            self.log("projection/bias", self.pose_loss.renderer.bias)
            self.log("projection/scale", self.pose_loss.renderer.scale)
        
        return loss

    def batch_unpacker(self, batch):
        """Split a batch into indices, micrographs, rotations, shifts and conformations,
        according to which of them the dataset supplies.
        """
        idxs = batch[0]
        mrcs = batch[1].unsqueeze(1)
        next = 2
        if self.load_rots:
            rots = batch[next].unsqueeze(1)
            next += 1
        else:
            rots = None
        if self.load_shifts:
            shifts = batch[next].unsqueeze(1)
            next += 1
        else:
            shifts = None
        if self.load_confs:
            confs = batch[next]
        else:
            confs = None
        return idxs, mrcs, rots, shifts, confs

    # def KL_div_loss(
    #     self, 
    #     latents: Float[Tensor, "B L"]
    # ) -> Float[Tensor, ""]:
    #     # Assumes standard normal latent prior.
    #     mu_sqrs = latents[:, :, 0].square() # output: [B, latent_dim]
    #     sigma_sqrs = latents[:, :, 1].square() # output: [B, latent_dim]
    #     loss_vec = 0.5 * (
    #         mu_sqrs + sigma_sqrs - torch.log(sigma_sqrs) - 1.
    #     ).sum(dim=1) # output: [B], SUM OR NOT?
    #     loss_avg = loss_vec.mean()
    #     return loss_avg
    
    def mrc_reconstruction_loss(
            self, 
            mrcs: Float[Tensor, "B 1 R R"],
            pred_confs: Float[Tensor, "B N 3"],
            latents: Float[Tensor, "B L"],
            idxs: Int[Tensor, "B"],
            rots: Float[Tensor, "B 1 3 3"] | None,
            shifts: Float[Tensor, "B 1 2"] | None, 
        ) -> Float[Tensor, ""]:
        """Return the mean image error under the configured pose loss."""
        return torch.mean(self.pose_loss(mrcs, pred_confs, latents, idxs, rots, shifts))

    def residue_distance_loss(
            self, 
            pred_confs: Float[Tensor, "B N 3"],
        ) -> Float[Tensor, ""]:
        """Penalise deviation of consecutive residue distances from the reference
        distance.
        """
        residue_dists = torch.norm(
            pred_confs[:, 1:, :] - pred_confs[:, :-1, :], 
            dim=2,
        ) # output: [B, N-1]
        return (residue_dists - self.true_residue_dist).square().mean()
    
    def centering_loss(
            self,
            pred_confs: Float[Tensor, "B N 3"],
        ) -> Float[Tensor, ""]:
        """Penalise displacement of the conformation centroid from the origin."""
        return pred_confs.mean(dim=1).square().sum(dim=1).mean()

    def geometric_loss(
            self,
            pred_confs: Float[Tensor, "B N 3"],
        ) -> Float[Tensor, ""]:
        """Penalise deviation of all pairwise log squared distances from those of the
        reference conformation.
        """
        pred_edm_sq_log = (
            pred_confs[:, self.i_idx] - pred_confs[:, self.j_idx]
        ).square().sum(dim=-1).log()
        return (pred_edm_sq_log - self.prior_edm_sq_log).square().mean()
    
    def local_geometric_loss(
            self,
            pred_confs: Float[Tensor, "B N 3"],
        ) -> Float[Tensor, ""]:
        """Penalise deviation of log squared distances within a sequence window,
        weighted by exponential decay in sequence separation.
        """
        pred_edm_sq_log = (
            pred_confs[:, self.local_i_idx] - pred_confs[:, self.local_j_idx]
        ).square().sum(dim=-1).log()
        return (
            torch.exp(1 - torch.abs(self.local_i_idx - self.local_j_idx) / self.loss_local_geom_decay) 
            * (pred_edm_sq_log - self.local_prior_edm_sq_log).square()
        ).mean()
    
    def neighbourhood_loss(
            self,
            pred_confs: Float[Tensor, "B N 3"],
        ) -> Float[Tensor, ""]:
        """Unused: the neighbour index buffers it reads are never registered."""
        return (
            torch.exp(1 - torch.abs(self.neighbour_i_idx - self.neighbour_j_idx)) 
            * (
                pred_confs[:, self.neighbour_i_idx] - pred_confs[:, self.neighbour_j_idx]
            ).square().sum(dim=-1)
        ).mean()
    
    def validation_step(self, batch, batch_idx):
        """Log the Kabsch-aligned RMSD against the known conformations, and the
        conformation and image pairs of a fixed set of tracked samples.
        """
        if not self.load_confs:
            return 
        
        idxs, mrcs, rots, shifts, confs = self.batch_unpacker(batch)

        pred_confs, _ = self.forward(idxs, mrcs)
        R, t, rmsd = torch.vmap(kabsch_torch)(pred_confs, confs)
        self.rmsd_list.append(rmsd)
        rmsd_avg = rmsd.mean()
        self.log("validation/rmsd", rmsd_avg, sync_dist=True)

        if self.load_shifts:
            raise NotImplementedError("Known shifts have not yet been implemented.")
        # Only correct when rotations are known; fix.
        if self.load_rots:
            pred_mrcs = self.renderer(pred_confs, rots, shifts)
            confs_cent = confs - confs.mean(dim=1, keepdim=True)
            sim_mrcs = self.renderer(confs_cent, rots, shifts)

        # loss_mrc_true_confs = (sim_mrcs - mrcs).square().mean(dim=(2, 3)).mean()
        # self.log("validation/loss_mrc_true_confs", loss_mrc_true_confs)

        for b, idx in enumerate(idxs):
            track_idx = idx // (self.num_mrcs // 10)
            if idx % (self.num_mrcs // 10) == 0:
                pred_confs_kab = (
                    torch.einsum(
                        'ij, nj -> ni', 
                        R[b, :, :], 
                        pred_confs[b, :, :],
                    ) + t[b, :]
                ).cpu()
                figly = plotly.graph_objects.Figure()
                figly.add_trace(
                    plotly.graph_objects.Scatter3d(
                        x=pred_confs_kab[:, 0],
                        y=pred_confs_kab[:, 1],
                        z=pred_confs_kab[:, 2],
                        marker_color="red",
                        name=f"Predicted Conformation, tracked sample #{track_idx}",
                    )
                )
                confs_plot = confs[b, :, :].cpu()
                figly.add_trace(
                    plotly.graph_objects.Scatter3d(
                        x=confs_plot[:, 0],
                        y=confs_plot[:, 1],
                        z=confs_plot[:, 2],
                        marker_color="blue",
                        name=f"True Conformation, tracked sample #{track_idx}",
                    )
                )
                figly.update_scenes(
                    xaxis_visible=False, 
                    yaxis_visible=False, 
                    zaxis_visible=False,
                )
                self.logger.experiment.log(
                    {"validation/pred_confs": wandb.Plotly(figly)},
                )

                # Only correct when rotations are known; fix.
                if self.load_rots:
                    self.logger.log_image(
                        "validation/images", 
                        [
                            torch.cat(
                                [
                                    mrcs.squeeze(1)[b], 
                                    sim_mrcs.squeeze(1)[b], 
                                    pred_mrcs.squeeze(1)[b],
                                ], 
                                dim=1,
                            )
                        ], 
                        caption=[
                            f"Tracked sample #{track_idx}: "
                            + "True image (left), "
                            + "projection of true conformation (middle), "
                            + "projection of predicted conformation (right)"
                        ],
                    )

    def on_validation_epoch_start(self):
        """Discard RMSDs accumulated by any previous validation epoch."""
        self.rmsd_list = []

    def on_validation_epoch_end(self):
        """Log a histogram of the RMSDs gathered across all ranks."""
        if not self.load_confs:
            return super().on_validation_epoch_end()
        if not self.trainer.sanity_checking:
            # all_gather is collective, so every rank must reach it: under DDP
            # each rank only holds its own shard of the validation set, and
            # without this the histogram would describe rank zero's shard alone.
            rmsds = self.all_gather(torch.cat(self.rmsd_list)).flatten().cpu()
            if self.trainer.is_global_zero:
                fig, ax = plt.subplots()
                ax.hist(rmsds, bins=100, range=(0., max(10., rmsds.max().item())))
                ax.set_xlabel("RMSD [Ångström]")
                self.logger.log_image(
                    "validation/rmsd_hist", 
                    [fig], 
                    step=self.current_epoch,
                )
                plt.close(fig)
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        """Build the optimiser and learning-rate scheduler named by the config."""
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
            )
        else: 
            raise NotImplementedError(
                "The config lists an unimplemented optimizer: ",
                f"{self.optimizer_name}\n",
            )
        if self.lr_scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_initial_factor,
                total_iters=self.lr_epochs_until_max,
            )
        elif self.lr_scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                eta_min=self.lr_initial_factor*self.learning_rate,
                T_max=self.lr_epochs_until_max//2,
            )
        else:
            raise NotImplementedError(
                "The config lists an unimplemented lr scheduler: ",
                f"{self.lr_scheduler_name}\n",
            )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        return output
    
    def renderer(
            self,
            confs: Float[Tensor, "B N 3"],
            rots: Float[Tensor, "B Q 3 3"],
            shifts: Float[Tensor, "B Q 2"] | None,
        ) -> Float[Tensor, "B Q R R"]:
        """Render projections through the renderer held by the pose loss."""
        return self.pose_loss.renderer(confs, rots, shifts)
    
# class FreezeUnfreeze(BaseFinetuning):
#     def __init__(self, unfreeze_at_epoch=10):
#         super().__init__()
#         self._unfreeze_at_epoch = unfreeze_at_epoch
    
#     def freeze_before_training(self, pl_module):
#         self.freeze(pl_module.feature_extractor)

#     def finetune_function(self, pl_module, current_epoch, optimizer):
#         if current_epoch == self._unfreeze_at_epoch:
#             self.unfreeze_and_add_param_group(
#                 modules=pl_module.feature_extractor,
#                 optimizer=optimizer,
#                 train_bn=True,
#             )