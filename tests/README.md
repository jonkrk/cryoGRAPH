# Tests

`pytest` suite for the `cryograph` package.

## Layout

Tests live in this top-level `tests/` directory (outside `src/`, so they are not
packaged into the wheel). Roughly one file per source module:

| test file                 | covers                                              |
|---------------------------|-----------------------------------------------------|
| `test_rotation_funcs.py`  | `utils/rotation_funcs.py` — conversions, roundtrips, quaternion algebra |
| `test_geom_loss_funcs.py` | `utils/geom_loss_funcs.py` — pair-index / prior builders |
| `test_kabsch.py`          | `utils/kabsch.py` — `kabsch_torch` alignment / RMSD |
| `test_renderer.py`        | `renderer/renderer.py` — projection + CTF shapes, autograd, CPU/GPU parity |
| `test_pdb_to_graph.py`    | `pdb_to_graph.py` — graph shape, edge validity, backbone order |
| `test_layers.py`          | `models/layers/{mlp_message,egcl_message}` — batched message passing |
| `test_encoders.py`        | `models/encoders/` — lookup + CNN encoders, builder dispatch |
| `test_decoders.py`        | `models/decoders/` — all 9 latent→conformation decoders via `build_decoder` |
| `test_cryograph_module.py` | `models/cryograph_module.py` — `forward`, `batch_unpacker`, regularisers, optimizers |
| `test_poseloss.py`            | `pose/` — `build_pose_loss` dispatch, `KnownPoseLoss`, `MLPPoseLoss`, ESL simplex/reduction, (opt-in) end-to-end recovery |
| `test_dataset.py`         | `dataset.py` — loading contract, orientation representations, normalisation |

## Running

```bash
uv sync                       # installs pytest (dev group)

uv run pytest                 # everything runnable in this environment
uv run pytest -m "not gpu and not needs_data and not slow"   # fast, portable subset
uv run pytest tests/test_renderer.py -q
```

## Markers

- `gpu` — needs a CUDA device (auto-skips otherwise).
- `needs_data` — needs `CRYOGRAPH_TEST_DATA` set to a directory containing
  `micrographs.pt` (and `rots.pt` / `confs.pt` for the fuller checks).
- `slow` — full render-the-whole-grid / solve path.

```bash
CRYOGRAPH_TEST_DATA=/path/to/trainingdata/adk4ake uv run pytest
```

## Fixtures (`conftest.py`)

- `_deterministic` (autouse) — seeds the RNG before every test.
- `structure_path` — bundled `structs/adk4ake_fold_with_H_posed.pdb` (214 residues).
- `rot_grid_path` — bundled `rotation_grids/rot_grid_1821.pt`.
- `tiny_graph` — a synthetic 16-residue PyG graph (no data / no PDB needed).
- `base_config` — `cfgs/config.yaml` parsed to a dict, depth knobs shrunk, fresh per test.
- `training_data_dir` — resolves/validates `CRYOGRAPH_TEST_DATA`, else skips.
- `cuda_device` — `"cuda"` or skip.
