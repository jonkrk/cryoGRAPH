"""Coverage for ``cryograph.pose`` — the pose-loss strategies and their builder.

* ``TestBuildPoseLoss``     - ``build_pose_loss`` dispatch and error paths.
* ``TestKnownPoseLoss``     - the ``known`` path, render + MSE + autograd.
* ``TestMLPPoseLoss``       - the 6D-vector -> rotation map.
* ``TestESLReduction``      - ESL simplex projection and loss reduction (fast,
                              data-free, uses the bundled SO(3) grid).
* ``test_esl_multi_solve_recovers_grid_poses`` - end-to-end pose recovery on a
                              real micrograph stack (opt-in: ``slow`` + ``needs_data``).
"""

import numpy as np
import pytest
import torch

from cryograph.pose import ESLPoseLoss, KnownPoseLoss, MLPPoseLoss
from cryograph.pose.builder import build_pose_loss
from cryograph.renderer import Renderer

NUM_RESIDUES = 8
SIDELENGTH = 16
ESL_ETA = 2.0 / 3.0
ESL_NUM_PTS_BASE = 15


@pytest.fixture
def renderer() -> Renderer:
    torch.manual_seed(0)
    return Renderer(
        pixel_width=1.0,
        sidelength=SIDELENGTH,
        ctf_pad=0,
        num_residues=NUM_RESIDUES,
        elec_vec=10.0 + 90.0 * torch.rand(NUM_RESIDUES),
        std_vec=1.0 + torch.rand(NUM_RESIDUES),
        spherical_aberration=2.7,
        amplitude_contrast_ratio=0.06,
        defocus=20000.0,
        electron_energy=300.0,
        init_scale=1.0,
        init_bias=0.0,
        learn_proj_params=False,
    )


class _StubRenderer(torch.nn.Module):
    """Placeholder for ESL maths tests that never call the renderer."""


@pytest.fixture
def esl(rot_grid_path) -> ESLPoseLoss:
    return ESLPoseLoss(
        eta=ESL_ETA,
        num_pts_base=ESL_NUM_PTS_BASE,
        num_iters=10,
        rot_grid_path=str(rot_grid_path),
        renderer=_StubRenderer(),
    )


# --------------------------------------------------------------------------- #
class TestBuildPoseLoss:
    @staticmethod
    def _build(cfg, renderer, *, method, load_rots=False, load_shifts=False):
        cfg["pose_loss"]["method"] = method
        return build_pose_loss(
            cfg, load_rots=load_rots, load_shifts=load_shifts, num_mrcs=16,
            renderer=renderer,
        )

    def test_known(self, base_config, renderer):
        pl = self._build(base_config, renderer, method="known", load_rots=True)
        assert isinstance(pl, KnownPoseLoss)

    def test_known_requires_rots(self, base_config, renderer):
        with pytest.raises(ValueError, match="requires load_rots"):
            self._build(base_config, renderer, method="known", load_rots=False)

    def test_esl_accepts_a_quaternion_grid(
        self, base_config, renderer, rot_grid_path, tmp_path
    ):
        """A grid may be supplied as quaternions and is converted on load."""
        import torch as _torch
        from cryograph.utils import rotmats_to_quaternions
        grid = _torch.load(rot_grid_path, weights_only=True)
        quat_path = tmp_path / "quat_grid.pt"
        _torch.save(rotmats_to_quaternions(grid), quat_path)

        base_config["pose_loss"]["rot_grid_path"] = str(quat_path)
        pl = self._build(base_config, renderer, method="esl")

        assert pl.rot_grid.shape == grid.shape
        r = pl.rot_grid
        _torch.testing.assert_close(
            r @ r.transpose(-1, -2), _torch.eye(3).expand_as(r), atol=1e-4, rtol=0
        )

    def test_esl(self, base_config, renderer, rot_grid_path):
        base_config["pose_loss"]["rot_grid_path"] = str(rot_grid_path)
        pl = self._build(base_config, renderer, method="esl")
        assert isinstance(pl, ESLPoseLoss)

    def test_mlp_est(self, base_config, renderer):
        pl = self._build(base_config, renderer, method="mlp_est")
        assert isinstance(pl, MLPPoseLoss)

    def test_shifts_not_implemented(self, base_config, renderer):
        with pytest.raises(NotImplementedError):
            self._build(base_config, renderer, method="known", load_rots=True,
                        load_shifts=True)

    def test_unknown_method(self, base_config, renderer):
        with pytest.raises(ValueError, match="Unknown pose loss"):
            self._build(base_config, renderer, method="telepathy", load_rots=True)


def test_shipped_configs_have_esl_eta_in_range():
    """ESLPoseLoss rejects eta outside (1/4, 2/3), so every shipped config must
    stay inside that interval."""
    import yaml
    from pathlib import Path

    for cfg_path in sorted((Path(__file__).parents[1] / "cfgs").glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        eta = cfg.get("pose_loss", {}).get("esl", {}).get("eta")
        if eta is not None:
            assert 0.25 < eta <= 2.0 / 3.0, f"{cfg_path.name}: eta={eta}"


# --------------------------------------------------------------------------- #
class TestKnownPoseLoss:
    def test_forward_shape_and_grad(self, renderer):
        pl = KnownPoseLoss(renderer=renderer)
        batch = 3
        mrcs = torch.randn(batch, 1, SIDELENGTH, SIDELENGTH)
        pred_confs = torch.randn(batch, NUM_RESIDUES, 3, requires_grad=True)
        rots = torch.eye(3).expand(batch, 1, 3, 3)
        latents = torch.randn(batch, 8)
        idxs = torch.arange(batch)

        loss = pl(mrcs, pred_confs, latents, idxs, rots, shifts=None)

        assert loss.shape == (batch, 1)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert pred_confs.grad is not None
        assert torch.isfinite(pred_confs.grad).all()


# --------------------------------------------------------------------------- #
class TestMLPPoseLoss:
    @pytest.mark.parametrize("batch", [2, 3, 5])
    def test_rotmats_from_twovecs_are_valid_rotations(self, renderer, batch):
        pl = MLPPoseLoss(
            num_layers_mlppose=2,
            num_hidden_mlppose=16,
            latent_dim=8,
            act_name="relu",
            renderer=renderer,
        )
        rotmats = pl.rotmats_from_twovecs(torch.randn(batch, 6))

        assert rotmats.shape == (batch, 1, 3, 3)
        r = rotmats[:, 0]
        torch.testing.assert_close(
            r @ r.transpose(-1, -2), torch.eye(3).expand(batch, 3, 3), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            torch.linalg.det(r), torch.ones(batch), atol=1e-5, rtol=0
        )


# --------------------------------------------------------------------------- #
class TestESLReduction:
    def test_project_to_simplex_returns_a_probability_vector(self, esl):
        v = torch.randn(2, 1, esl.rot_grid.shape[0])

        beta = esl._project_to_simplex(v)

        assert beta.shape == v.shape
        assert (beta >= 0).all() and (beta <= 1).all()
        torch.testing.assert_close(beta.sum(-1), torch.ones(2, 1), atol=1e-3, rtol=0)

    def test_project_to_simplex_is_near_identity_on_a_vertex(self, esl):
        v = torch.full((1, 1, esl.rot_grid.shape[0]), -10.0)
        v[..., 5] = 10.0

        beta = esl._project_to_simplex(v)

        assert beta[0, 0, 5] > 0.9
        torch.testing.assert_close(beta.sum(-1), torch.ones(1, 1), atol=1e-3, rtol=0)

    def test_compute_esl_skip_estimate(self, esl):
        loss_grid = torch.rand(4, 1, esl.rot_grid.shape[0])

        loss_vec, bary_rots = esl.compute_esl(loss_grid, skip_estimate=True)

        assert bary_rots is None
        assert loss_vec.shape == (4, 1)
        assert torch.isfinite(loss_vec).all()
        # a convex combination of grid losses -> within their range
        assert (loss_vec >= loss_grid.min()).all()
        assert (loss_vec <= loss_grid.max()).all()

    def test_compute_esl_barycentre_is_a_rotation(self, esl):
        loss_grid = torch.full((1, 1, esl.rot_grid.shape[0]), 1.0)
        loss_grid[..., 7] = 0.0  # sharp minimum at one grid point

        _, bary_rots = esl.compute_esl(loss_grid, skip_estimate=False)

        assert bary_rots.shape == (1, 1, 3, 3)
        r = bary_rots[0, 0]
        torch.testing.assert_close(r @ r.T, torch.eye(3), atol=1e-4, rtol=0)
        torch.testing.assert_close(
            torch.linalg.det(r), torch.tensor(1.0), atol=1e-4, rtol=0
        )


# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.needs_data
def test_esl_multi_solve_recovers_grid_poses(training_data_dir, rot_grid_path, structure_path):
    """End-to-end: render the base conformation over the SO(3) grid, then check
    that ESL assigns each real micrograph a barycentre close to its true rotation.
    """
    from pathlib import Path

    import yaml

    from cryograph.pdb_to_graph import graph_construct

    rots_path = training_data_dir / "rots.pt"
    if not rots_path.is_file():
        pytest.skip("rots.pt required to score pose recovery")

    cfg = yaml.safe_load((Path(__file__).parents[1] / "cfgs" / "config.yaml").read_text())
    optics = cfg["rendering"]["optics"]
    volume = cfg["rendering"]["volume"]

    graph = graph_construct(str(structure_path))
    elec_vec = torch.tensor(
        [volume["amino_electrons"][r] for r in graph["residue_name"]], dtype=torch.float32
    )
    std_vec = torch.tensor(
        [volume["amino_std"][r] for r in graph["residue_name"]], dtype=torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 64
    mrcs = torch.load(training_data_dir / "micrographs.pt", weights_only=True, mmap=True)
    sidelength = mrcs.shape[-1]
    mrcs = mrcs[:n].to(device).unsqueeze(0)
    true_rots = torch.load(rots_path, weights_only=True, mmap=True)[:n].to(device)

    renderer = Renderer(
        pixel_width=optics["pixel_width"],
        sidelength=sidelength,
        ctf_pad=cfg["rendering"]["image"]["ctf_pad"],
        num_residues=int(graph["num_nodes"]),
        elec_vec=elec_vec,
        std_vec=std_vec,
        spherical_aberration=optics["spherical_aberration"],
        amplitude_contrast_ratio=optics["amplitude_contrast_ratio"],
        defocus=optics["defocus"],
        electron_energy=optics["electron_energy"],
        init_scale=cfg["rendering"]["image"]["init_scale"],
        init_bias=cfg["rendering"]["image"]["init_bias"],
        learn_proj_params=False,
    ).to(device)

    esl = ESLPoseLoss(
        eta=ESL_ETA,
        num_pts_base=ESL_NUM_PTS_BASE,
        num_iters=100,
        rot_grid_path=str(rot_grid_path),
        renderer=renderer,
    ).to(device)

    base_conf = graph["coords"].to(device).unsqueeze(0)
    base_conf = base_conf - base_conf.mean(dim=1, keepdim=True)

    _, bary_rots = esl.multi_solve(mrcs, base_conf, rots=None, shifts=None, grid_size=1)

    rel = torch.einsum("bmij, bmkj -> bmik", bary_rots, true_rots.unsqueeze(0))
    cos_angle = (torch.einsum("bmii -> bm", rel) - 1.0) / 2.0
    assert (cos_angle.squeeze(0) > np.cos(np.deg2rad(30.0))).float().mean() > 0.5
