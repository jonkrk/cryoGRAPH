"""Coverage for ``cryograph.utils.kabsch.kabsch_torch``.

The batched ``torch.vmap`` form mirrors exactly how the model calls it.
"""

import torch

from cryograph.utils.kabsch import kabsch_torch
from cryograph.utils.rotation_funcs import rand_rotmats


def test_kabsch_on_identical_point_sets():
    p = torch.randn(40, 3, dtype=torch.float64)
    R, _, rmsd = kabsch_torch(p, p.clone())

    assert rmsd < 1e-9
    torch.testing.assert_close(R, torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=0)


def test_kabsch_recovers_a_known_rotation():
    p = torch.randn(60, 3, dtype=torch.float64)
    R_true = rand_rotmats(()).double()
    q = p @ R_true.T + torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)

    R, _, rmsd = kabsch_torch(p, q)

    assert rmsd < 1e-6
    torch.testing.assert_close(R, R_true, atol=1e-5, rtol=0)


def test_kabsch_vmap_batched_matches_model_usage():
    batch, n = 4, 25
    pred = torch.randn(batch, n, 3)
    true = torch.randn(batch, n, 3)

    R, t, rmsd = torch.vmap(kabsch_torch)(pred, true)

    assert R.shape == (batch, 3, 3)
    assert t.shape == (batch, 3)
    assert rmsd.shape == (batch,)
    assert torch.isfinite(rmsd).all()
    assert (rmsd >= 0).all()
