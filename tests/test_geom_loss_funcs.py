"""Coverage for ``cryograph.utils.geom_loss_funcs`` — the index and prior builders
behind ``CryoGRAPH``'s geometric regularisers.
"""

import torch

from cryograph.utils.geom_loss_funcs import build_pair_indices, build_prior_sqrlogs


def test_build_pair_indices_global_pairs_are_all_i_lt_j():
    n = 20
    i_idx, j_idx, _, _ = build_pair_indices(n, local_size=3)

    assert i_idx.shape == j_idx.shape
    assert i_idx.numel() == n * (n - 1) // 2  # every unordered pair once
    assert (i_idx < j_idx).all()
    assert int(i_idx.min()) == 0
    assert int(j_idx.max()) == n - 1

    pairs = {(int(a), int(b)) for a, b in zip(i_idx, j_idx)}
    assert len(pairs) == i_idx.numel()  # no duplicates


def test_build_pair_indices_local_window():
    n = 20
    local_size = 3
    _, _, li, lj = build_pair_indices(n, local_size=local_size)

    offsets = (lj - li)
    assert int(offsets.min()) == 1
    assert int(offsets.max()) == local_size
    # count per offset k is (n - k), for k = 1 .. local_size
    expected = sum(n - k for k in range(1, local_size + 1))
    assert li.numel() == expected
    assert (li >= 0).all()
    assert (lj < n).all()


def test_build_prior_sqrlogs_matches_manual():
    coords = torch.randn(12, 3)
    i_idx, j_idx, _, _ = build_pair_indices(12, local_size=2)

    prior = build_prior_sqrlogs(coords, i_idx, j_idx)

    assert prior.shape == (1, i_idx.numel())
    manual = ((coords[i_idx] - coords[j_idx]) ** 2).sum(-1).log()
    torch.testing.assert_close(prior.squeeze(0), manual)
    assert torch.isfinite(prior).all()
