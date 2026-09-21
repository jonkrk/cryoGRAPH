"""Coverage for ``cryograph.models.cryograph_module.CryoGRAPH``.

``training_step`` and ``validation_step`` need a ``Trainer`` and a dataset, and
are left to the opt-in integration tests. Everything else -- construction,
``forward``, ``batch_unpacker``, ``configure_optimizers`` and the geometric
regularisers -- is plain tensor code, exercised here on a synthetic graph.
"""

import pytest
import torch

from cryograph.models.cryograph_module import CryoGRAPH

NUM_MRCS = 16
SIDELENGTH = 16
BATCH = 4


@pytest.fixture
def model(base_config, tiny_graph):
    cfg = base_config
    cfg["network"]["encoder"]["model"] = "lookup"
    cfg["network"]["decoder"]["model"] = "gcnconv"
    cfg["pose_loss"]["method"] = "known"
    cfg["data"]["load_rots"] = True
    cfg["data"]["load_shifts"] = False
    cfg["data"]["load_confs"] = True

    n = int(tiny_graph["num_nodes"])
    return CryoGRAPH(
        config=cfg,
        protein_graph=tiny_graph,
        elec_vec=torch.full((n,), 50.0),
        std_vec=torch.full((n,), 1.5),
        num_mrcs=NUM_MRCS,
        mrc_sidelength=SIDELENGTH,
    )


def test_forward_shapes(model, tiny_graph):
    n = int(tiny_graph["num_nodes"])
    idxs = torch.arange(BATCH)
    mrcs = torch.randn(BATCH, 1, SIDELENGTH, SIDELENGTH)

    pred_confs, latents = model.forward(idxs, mrcs)

    assert pred_confs.shape == (BATCH, n, 3)
    assert latents.shape == (BATCH, model.hparams.config["network"]["latent_dim"])
    assert torch.isfinite(pred_confs).all()


def test_forward_is_differentiable(model):
    idxs = torch.arange(BATCH)
    mrcs = torch.randn(BATCH, 1, SIDELENGTH, SIDELENGTH)

    pred_confs, _ = model.forward(idxs, mrcs)
    pred_confs.pow(2).sum().backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_batch_unpacker_respects_load_flags(model):
    idxs = torch.arange(BATCH)
    mrcs = torch.randn(BATCH, SIDELENGTH, SIDELENGTH)
    rots = torch.eye(3).expand(BATCH, 3, 3)
    confs = torch.randn(BATCH, 16, 3)

    out_idxs, out_mrcs, out_rots, out_shifts, out_confs = model.batch_unpacker(
        (idxs, mrcs, rots, confs)
    )

    assert out_mrcs.shape == (BATCH, 1, SIDELENGTH, SIDELENGTH)  # channel dim added
    assert out_rots.shape == (BATCH, 1, 3, 3)
    assert out_shifts is None  # load_shifts is False
    assert out_confs.shape == (BATCH, 16, 3)
    torch.testing.assert_close(out_idxs, idxs)


def test_centering_loss_zero_iff_centred(model):
    n = int(model.decoder.base_coords.shape[0])
    centred = torch.randn(BATCH, n, 3)
    centred = centred - centred.mean(dim=1, keepdim=True)
    assert model.centering_loss(centred) < 1e-10

    offset = centred + torch.tensor([1.0, 0.0, 0.0])
    assert model.centering_loss(offset) > 0.5


def test_residue_distance_loss_zero_at_reference_spacing(model):
    ref = model.true_residue_dist
    n = int(model.decoder.base_coords.shape[0])
    # a straight chain with consecutive spacing == ref
    chain = torch.zeros(1, n, 3)
    chain[0, :, 0] = torch.arange(n, dtype=torch.float32) * ref

    assert model.residue_distance_loss(chain) < 1e-8


def test_geometric_losses_zero_at_reference_conformation(model, tiny_graph):
    # priors are built from graph coords; pairwise distances are translation
    # invariant, so the reference conformation should give ~zero geom loss.
    coords = tiny_graph["coords"].unsqueeze(0).expand(BATCH, -1, -1)

    assert model.geometric_loss(coords).abs() < 1e-4
    assert model.local_geometric_loss(coords).abs() < 1e-4

    perturbed = coords + 2.0 * torch.randn_like(coords)
    assert model.geometric_loss(perturbed) > model.geometric_loss(coords)


def test_mrc_reconstruction_loss_is_finite_scalar(model):
    n = int(model.decoder.base_coords.shape[0])
    mrcs = torch.randn(BATCH, 1, SIDELENGTH, SIDELENGTH)
    pred_confs = torch.randn(BATCH, n, 3)
    latents = torch.randn(BATCH, model.hparams.config["network"]["latent_dim"])
    idxs = torch.arange(BATCH)
    rots = torch.eye(3).expand(BATCH, 1, 3, 3)

    loss = model.mrc_reconstruction_loss(mrcs, pred_confs, latents, idxs, rots, None)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("scheduler", ["linear", "cosine"])
def test_configure_optimizers(model, scheduler):
    model.lr_scheduler_name = scheduler
    out = model.configure_optimizers()

    assert isinstance(out["optimizer"], torch.optim.Adam)
    assert "scheduler" in out["lr_scheduler"]
