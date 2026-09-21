import os
import torch
from torch.utils.data import DataLoader
import torch_geometric as pyg
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from jaxtyping import Float
from torch import Tensor
from cryograph.dataset import CryoDataset
from cryograph.pdb_to_graph import graph_construct
from cryograph.models.cryograph_module import CryoGRAPH

def define_residue_vecs(
        config: dict, 
        protein_graph: pyg.data.Data,
    ) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
    """Return the per-residue electron count and Gaussian width of a protein graph,
    looked up in the config tables by residue name.
    """
    elec_vec = torch.zeros((protein_graph['num_nodes'],), dtype=config['data']['dtype'])
    std_vec = torch.zeros((protein_graph['num_nodes'],), dtype=config['data']['dtype'])
    for idx, key in enumerate(protein_graph['residue_name']):
        elec_vec[idx] += config['rendering']['volume']['amino_electrons'][key]
        std_vec[idx] += config['rendering']['volume']['amino_std'][key]
    return elec_vec, std_vec

def train(config: dict):
    """Train a model on the dataset and with the settings given by the config."""
    pl.seed_everything(config['training']['seed'])
    torch.set_float32_matmul_precision(config['training']['precision'])
    
    structure_path = os.path.join(config['data']['dir'], config['data']['structure_file'])
    protein_graph = graph_construct(structure_path)

    logger = WandbLogger(
        project=config['wandb']['project'], 
        name=config['wandb']['run'], 
        group=config['wandb']['group'],
        save_dir=config['wandb']['dir'],
    )

    logger.log_hyperparams(config)

    elec_vec, std_vec = define_residue_vecs(config, protein_graph)

    dataset = CryoDataset(
        config=config, 
        num_residues=protein_graph['num_nodes'],
    )

    model = CryoGRAPH(
        config=config, 
        protein_graph=protein_graph, 
        elec_vec=elec_vec, 
        std_vec=std_vec,
        num_mrcs=len(dataset),
        mrc_sidelength=dataset.get_sidelength(),
    )

    callbacks = []

    if config['pose_loss']['method'] == 'gradient' and not config['data']['load_rots']:
        class CryoCallback(Callback):
            def on_fit_start(self, trainer, pl_module):
                pl_module.pose_loss.init_quats(dataset, protein_graph, batch_size=8)
        callbacks.append(CryoCallback())

    # Keep the single best model by validation RMSD. That metric only exists
    # when conformations are available, so without them fall back to Lightning's
    # default (last epoch) checkpointing rather than monitoring a missing key.
    if config['data']['load_confs']:
        callbacks.append(
            ModelCheckpoint(
                monitor="validation/rmsd",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:04d}-rmsd{validation/rmsd:.4f}",
                auto_insert_metric_name=False,
            )
        )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'], 
        logger=logger, 
        log_every_n_steps=40,
        check_val_every_n_epoch=config['validation']['check_val_every_n_epoch'],
        devices=config['training']['devices'],
        callbacks=callbacks,
    )

    dataloader_train = DataLoader(
        dataset=dataset, 
        batch_size=config['training']['batch_size_train'], 
        num_workers=config['training']['num_workers_train'],
        shuffle=True,
        pin_memory=True,
    )

    dataloader_val = DataLoader(
        dataset=dataset, 
        batch_size=config['validation']['batch_size_val'], 
        num_workers=config['validation']['num_workers_val'],
        shuffle=False,
        pin_memory=True,
    )

    logger.watch(model)

    if config['validation']['do_prevalidation']:
        devs_tmp = config['training']['devices']
        if isinstance(devs_tmp, list):
            val_device = [devs_tmp[0]] 
        else:
            val_device = [0]
        trainer_val = pl.Trainer(
            logger=logger, 
            log_every_n_steps=40,
            devices=val_device,
            num_nodes=1,
        )
        trainer_val.validate(
            model=model, 
            dataloaders=dataloader_val,
        )

    trainer.fit(
        model=model, 
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )
