"""Shared fixtures and markers for the cryoGRAPH test-suite.

Anything that needs a training dataset (micrographs) is gated behind the
``CRYOGRAPH_TEST_DATA`` environment variable and skipped when it is unset.

Run everything that does not need a dataset or GPU::

    uv run pytest -m "not needs_data and not gpu"

Run the full suite (needs a CUDA device and a data directory)::

    CRYOGRAPH_TEST_DATA=/path/to/trainingdata/adk4ake uv run pytest
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = REPO_ROOT / "cfgs" / "config.yaml"

# A small single-chain structure that ships with the repo (ADK, 214 residues).
# This is the same file the configs point at via ``data.structure_file``.
STRUCTURE_FILE = REPO_ROOT / "structs" / "adk4ake_fold_with_H_posed.pdb"

# SO(3) grid of 1821 proper rotations that ships with the repo.
ROT_GRID_FILE = REPO_ROOT / "rotation_grids" / "rot_grid_1821.pt"


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed the global RNG before every test, so that the shape and tolerance
    assertions are reproducible from run to run.
    """
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def structure_path() -> Path:
    if not STRUCTURE_FILE.is_file():
        pytest.skip(f"bundled structure not found: {STRUCTURE_FILE}")
    return STRUCTURE_FILE


@pytest.fixture(scope="session")
def rot_grid_path() -> Path:
    if not ROT_GRID_FILE.is_file():
        pytest.skip(f"bundled rotation grid not found: {ROT_GRID_FILE}")
    return ROT_GRID_FILE


@pytest.fixture(scope="session")
def training_data_dir() -> Path:
    """Directory holding ``micrographs.pt`` (+ optional ``rots.pt`` / ``confs.pt``).

    Set ``CRYOGRAPH_TEST_DATA`` to enable the tests that use it.
    """
    raw = os.environ.get("CRYOGRAPH_TEST_DATA")
    if not raw:
        pytest.skip("set CRYOGRAPH_TEST_DATA to a dir containing micrographs.pt")
    path = Path(raw).expanduser()
    if not (path / "micrographs.pt").is_file():
        pytest.skip(f"no micrographs.pt under CRYOGRAPH_TEST_DATA={path}")
    return path


@pytest.fixture(scope="session")
def cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return "cuda"


# --------------------------------------------------------------------------- #
# Synthetic model inputs (no dataset needed) for encoder / decoder / module   #
# tests.                                                                      #
# --------------------------------------------------------------------------- #

TINY_NUM_RESIDUES = 16


@pytest.fixture(scope="session")
def tiny_graph():
    """A minimal residue graph shaped like what ``graph_construct`` returns.

    16 nodes on a slightly noisy helix (so pairwise distances are all distinct
    and non-zero), backbone edges in both directions plus a couple of
    long-range contacts.  Exposes ``['num_nodes']`` / ``['coords']`` /
    ``['edge_index']`` and ``.num_edges`` like a PyG ``Data``.
    """
    from torch_geometric.data import Data

    n = TINY_NUM_RESIDUES
    t = torch.linspace(0, 4 * torch.pi, n)
    coords = torch.stack([torch.cos(t), torch.sin(t), 0.5 * t], dim=-1)
    coords = coords + 0.05 * torch.randn(n, 3, generator=torch.Generator().manual_seed(0))
    coords = (coords - coords.mean(0)) * 3.8  # ~CA spacing in Angstrom

    src = list(range(n - 1))
    dst = list(range(1, n))
    src += [0, 3]  # a couple of non-backbone contacts
    dst += [5, 9]
    ij = torch.tensor([src + dst, dst + src], dtype=torch.long)  # undirected

    return Data(edge_index=ij, coords=coords.float(), num_nodes=n)


@pytest.fixture
def base_config():
    """The repo's ``cfgs/config.yaml`` parsed to a dict, with the ``dtype``
    string resolved and the expensive depth knobs shrunk so model tests stay
    fast.  Returns a fresh deep copy each time so tests may mutate freely.
    """
    if not CONFIG_FILE.is_file():
        pytest.skip(f"config not found: {CONFIG_FILE}")
    cfg = yaml.safe_load(CONFIG_FILE.read_text())

    if cfg["data"]["dtype"] == "torch.float32":
        cfg["data"]["dtype"] = torch.float32

    dec = cfg["network"]["decoder"]
    for spec in dec.values():
        if isinstance(spec, dict) and "repeat" in spec:
            spec["repeat"] = 2
    dec["gnn"]["messpass_layers"] = 2

    return copy.deepcopy(cfg)
