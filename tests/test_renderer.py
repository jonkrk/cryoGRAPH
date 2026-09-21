"""Coverage for ``cryograph.renderer`` — the differentiable forward model.

Deterministic shape and finiteness checks on the CTF kernel and on both pose
representations, plus autograd through the projection and an opt-in CPU/GPU
parity check.
"""

import pytest
import torch

from cryograph.renderer import Renderer
from cryograph.utils.rotation_funcs import rand_rotmats

NUM_RESIDUES = 8
SIDELENGTH = 16
BATCH = 2
NUM_POSES = 3

_RENDERER_KWARGS = dict(
    pixel_width=1.0,
    sidelength=SIDELENGTH,
    ctf_pad=0,
    num_residues=NUM_RESIDUES,
    spherical_aberration=2.7,
    amplitude_contrast_ratio=0.06,
    defocus=20000.0,
    electron_energy=300.0,
    init_scale=1.0,
    init_bias=0.0,
    learn_proj_params=False,
)


@pytest.fixture
def renderer() -> Renderer:
    torch.manual_seed(0)
    elec_vec = 10.0 + 90.0 * torch.rand(NUM_RESIDUES)
    std_vec = 1.0 + torch.rand(NUM_RESIDUES)
    return Renderer(elec_vec=elec_vec, std_vec=std_vec, **_RENDERER_KWARGS)


def _confs() -> torch.Tensor:
    return torch.randn(BATCH, NUM_RESIDUES, 3)


def test_ctf_kernel_shape_and_finite(renderer):
    kernel = renderer.ctf.kernel
    assert tuple(kernel.shape) == (SIDELENGTH, SIDELENGTH)
    assert torch.isfinite(kernel).all()


def test_forward_matrix_pose(renderer):
    rots = rand_rotmats((BATCH, NUM_POSES))
    out = renderer(_confs(), rots, None)
    assert tuple(out.shape) == (BATCH, NUM_POSES, SIDELENGTH, SIDELENGTH)
    assert torch.isfinite(out).all()


def test_forward_quaternion_pose(renderer):
    quats = torch.nn.functional.normalize(torch.randn(BATCH, NUM_POSES, 4), dim=-1)
    out = renderer(_confs(), quats, None)
    assert tuple(out.shape) == (BATCH, NUM_POSES, SIDELENGTH, SIDELENGTH)
    assert torch.isfinite(out).all()


def test_forward_is_differentiable_wrt_confs(renderer):
    confs = _confs().requires_grad_(True)
    rots = rand_rotmats((BATCH, NUM_POSES))
    renderer(confs, rots, None).sum().backward()
    assert confs.grad is not None
    assert torch.isfinite(confs.grad).all()


@pytest.mark.gpu
def test_forward_cuda_matches_cpu(renderer, cuda_device):
    confs = _confs()
    rots = rand_rotmats((BATCH, NUM_POSES))

    cpu_out = renderer(confs, rots, None)
    gpu_out = renderer.to(cuda_device)(
        confs.to(cuda_device), rots.to(cuda_device), None
    )
    torch.testing.assert_close(gpu_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3)
