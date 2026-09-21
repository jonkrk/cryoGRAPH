"""Coverage for ``cryograph.dataset.CryoDataset``.

The loading contract is exercised against synthetic directories written to
``tmp_path``; the tests marked ``needs_data`` additionally run against a real
stack when ``CRYOGRAPH_TEST_DATA`` points at one.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from cryograph.dataset import CryoDataset
from cryograph.utils import axes_angles_to_rotmats, euler_angles_to_rotmats


def _num_residues(data_dir) -> int:
    confs_path = data_dir / "confs.pt"
    if confs_path.is_file():
        return int(torch.load(confs_path, weights_only=True, mmap=True).shape[1])
    pytest.skip("confs.pt not present; cannot determine num_residues")


def _config(data_dir, *, subset=False, noise=0):
    return {
        "data": {
            "dir": str(data_dir),
            "dtype": torch.float32,
            "memory_mapped": True,
            "load_rots": (data_dir / "rots.pt").is_file(),
            "load_shifts": False,
            "noise_max_angle_degrees": noise,
            "load_confs": (data_dir / "confs.pt").is_file(),
            "subset": {"use_data_subset": subset, "size": 128},
        }
    }


@pytest.mark.needs_data
def test_dataset_loads_and_yields_a_batch(training_data_dir):
    n_res = _num_residues(training_data_dir)
    dataset = CryoDataset(_config(training_data_dir), num_residues=n_res)

    assert len(dataset) > 0
    sidelength = dataset.get_sidelength()
    assert sidelength > 0

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    idxs, mrcs, *_ = next(iter(loader))

    assert idxs.shape == (8,)
    assert mrcs.shape == (8, sidelength, sidelength)
    assert mrcs.dtype == torch.float32


@pytest.mark.needs_data
def test_dataset_subset_reduces_length(training_data_dir):
    n_res = _num_residues(training_data_dir)
    full = CryoDataset(_config(training_data_dir), num_residues=n_res)
    if len(full) <= 128:
        pytest.skip("dataset too small to exercise subsetting")

    subset = CryoDataset(
        _config(training_data_dir, subset=True), num_residues=n_res
    )
    assert len(subset) == 128


@pytest.mark.needs_data
def test_rotation_noise_keeps_valid_rotations(training_data_dir):
    if not (training_data_dir / "rots.pt").is_file():
        pytest.skip("rots.pt not present")

    dataset = CryoDataset(
        _config(training_data_dir, noise=30), num_residues=_num_residues(training_data_dir)
    )
    rots = dataset.rots
    eye = torch.eye(3).expand_as(rots)
    torch.testing.assert_close(
        rots @ rots.transpose(-1, -2), eye, atol=1e-4, rtol=0
    )


# --------------------------------------------------------------------------- #
# Synthetic directories: orientation representations and normalisation.
# --------------------------------------------------------------------------- #

D, R, N = 12, 8, 5


def _write_stack(tmp_path, **extra):
    torch.save(torch.randn(D, R, R), tmp_path / "micrographs.pt")
    torch.save(torch.randn(D, N, 3), tmp_path / "confs.pt")
    for name, tensor in extra.items():
        torch.save(tensor, tmp_path / f"{name}.pt")
    return tmp_path


def _synthetic_config(data_dir, *, load_rots=True, normalize=False):
    return {
        "data": {
            "dir": str(data_dir),
            "dtype": torch.float32,
            "memory_mapped": False,
            "load_rots": load_rots,
            "load_shifts": False,
            "noise_max_angle_degrees": 0,
            "load_confs": True,
            "normalize": normalize,
            "subset": {"use_data_subset": False, "size": D},
        }
    }


def test_rots_are_read_directly_when_present(tmp_path):
    rots = torch.linalg.qr(torch.randn(D, 3, 3))[0]
    rots[torch.linalg.det(rots) < 0] *= -1
    _write_stack(tmp_path, rots=rots)

    ds = CryoDataset(_synthetic_config(tmp_path), num_residues=N)
    torch.testing.assert_close(ds.rots, rots)


def test_rots_derived_from_euler_angles(tmp_path):
    eulers = torch.rand(D, 3) * 2.0 - 1.0
    _write_stack(tmp_path, euler_angles=eulers)

    ds = CryoDataset(_synthetic_config(tmp_path), num_residues=N)

    assert ds.rots.shape == (D, 3, 3)
    torch.testing.assert_close(ds.rots, euler_angles_to_rotmats(eulers))


def test_rots_derived_from_axes_and_angles(tmp_path):
    axes = torch.nn.functional.normalize(torch.randn(D, 3), dim=-1)
    angles = torch.rand(D, 1)
    _write_stack(tmp_path, axes=axes, angles=angles)

    ds = CryoDataset(_synthetic_config(tmp_path), num_residues=N)

    assert ds.rots.shape == (D, 3, 3)
    torch.testing.assert_close(ds.rots, axes_angles_to_rotmats(axes, angles))


def test_axes_angles_take_precedence_over_euler_angles(tmp_path):
    """Ordering matters: a sign convention cannot be detected from the file, so
    the more explicit representation is preferred when a directory has both."""
    axes = torch.nn.functional.normalize(torch.randn(D, 3), dim=-1)
    angles = torch.rand(D, 1)
    _write_stack(tmp_path, axes=axes, angles=angles,
                 euler_angles=torch.rand(D, 3) * 2.0 - 1.0)

    ds = CryoDataset(_synthetic_config(tmp_path), num_residues=N)
    torch.testing.assert_close(ds.rots, axes_angles_to_rotmats(axes, angles))


def test_missing_orientations_raise(tmp_path):
    _write_stack(tmp_path)
    with pytest.raises(ValueError, match="no orientations"):
        CryoDataset(_synthetic_config(tmp_path), num_residues=N)


def test_normalize_standardises_returned_micrographs(tmp_path):
    torch.save(3.0 + 7.0 * torch.randn(512, R, R), tmp_path / "micrographs.pt")
    torch.save(torch.randn(512, N, 3), tmp_path / "confs.pt")

    plain = CryoDataset(_synthetic_config(tmp_path, load_rots=False), num_residues=N)
    normed = CryoDataset(
        _synthetic_config(tmp_path, load_rots=False, normalize=True), num_residues=N
    )

    stacked = torch.stack([normed[i][1] for i in range(len(normed))])
    assert abs(stacked.mean().item()) < 0.05
    assert abs(stacked.std().item() - 1.0) < 0.05
    # the underlying stack is untouched; only what __getitem__ returns changes
    torch.testing.assert_close(plain[0][1], normed.mrcs[0])
