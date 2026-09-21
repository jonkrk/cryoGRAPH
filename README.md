# cryoGRAPH

Reference implementation of [*Protein Graph Neural Networks for Heterogeneous
Cryo-EM Reconstruction*](https://arxiv.org/abs/2602.21915).

## Overview

The method reconstructs a per-image atomic backbone conformation from a
single-particle cryo-EM dataset exhibiting continuous heterogeneity. It is an
*autodecoder*: each image is assigned a free latent vector through an
index-based lookup table, a graph neural network maps that latent to a set of
3D displacements of a template conformation, and a differentiable forward model
projects the resulting conformation so that it can be compared against the
observed image.

The protein backbone is represented as a graph with one node per amino acid
residue, carrying its C-alpha coordinate, and edges given by peptide bonds and
by hydrogen bonds of the secondary structure. Placing the decoder on this graph
supplies the geometric prior that distinguishes the method from the MLP
baseline.

Orientations may be supplied as ground truth or estimated during optimisation
by Ellipsoidal Support Lifting (ESL), which is run separately for each predicted
conformation. The data-discrepancy term is the expected squared error over the
measure ESL returns on SO(3), rather than the error at its barycentre.

## Installation

[uv](https://docs.astral.sh/uv/) is the supported way to run this code; it
resolves the pinned environment from `uv.lock`. Python 3.12 is required.

```bash
uv sync
```

## Running

Training is driven entirely by a YAML configuration file:

```bash
uv run main.py -c cfgs/config.yaml
```

Selected fields may be overridden on the command line, which is convenient for
seed sweeps and architecture comparisons:

| flag | overrides |
|------|-----------|
| `-c`, `--config` | path to the configuration file (required) |
| `-m`, `--model`  | `network.decoder.model` |
| `-s`, `--seed`   | `training.seed` |
| `-d`, `--devices`| `training.devices` (`auto`, `-1`, or e.g. `0,1`) |
| `-g`, `--group`  | `wandb.group` |
| `-r`, `--run`    | `wandb.run` |

`scripts/run_cfg_batch.sh cfgs/DIR/*` runs a directory of configurations in
sequence.

Metrics, reconstructed conformations and projection comparisons are logged to
Weights & Biases under `wandb.dir`; set `WANDB_MODE=disabled` to run without it.
The checkpoint minimising the validation RMSD is retained.

## Data

Datasets are not distributed with this repository. `data.dir` must contain:

| file | shape | required |
|------|-------|----------|
| `micrographs.pt` | `[D, R, R]` | yes |
| `<data.structure_file>` | template `.pdb` | yes |
| `confs.pt` | `[D, N, 3]` | if `data.load_confs` (validation) |
| `shifts.pt` | `[D, 2]` | if `data.load_shifts` (not implemented) |

With `data.load_rots`, the orientations are read from whichever representation
the directory provides, in this order:

| file | shape | convention |
|------|-------|------------|
| `rots.pt` | `[D, 3, 3]` | rotation matrices |
| `axes.pt` + `angles.pt` | `[D, 3]` + `[D, 1]` | axis and angle in radians |
| `euler_angles.pt` | `[D, 3]` | intrinsic Z, new Y, new Z, in radians |

Anything other than `rots.pt` is converted as it is loaded, so no preprocessing
step is required. Note that a sign convention cannot be detected from the file:
the opposite one yields inverse rotations silently, with no error and no visible
symptom beyond a poor reconstruction. Supply `rots.pt` when in doubt.

All tensors must share the dtype given by `data.dtype`. `D` is the number of
particle images, `R` the image side length and `N` the number of residues.
Note that the template is read from `data.dir`, not from `structs/`.

Micrographs are expected to be standardised to zero mean and unit variance. Set
`data.normalize: True` to have them standardised as they are read instead; the
statistics are estimated from a bounded sample, so the stack stays memory
mapped.

The datasets in the paper were produced with the TEM simulator Parakeet from
molecular dynamics trajectories, and the templates are independent AlphaFold 3
predictions; the ADK template and the AlphaFold output it came from are included
under `structs/`.

## Configuration

`cfgs/config.yaml` is the annotated reference configuration; the remaining files
are cluster- and dataset-specific variants of it. The blocks that correspond to
the paper are:

- `rendering` — the forward model. Per-residue electron counts and Gaussian
  widths, and the optics (defocus, spherical aberration, amplitude contrast,
  electron energy, pixel width) entering the contrast transfer function.
- `network.decoder` — `gcnconv` is the graph convolutional decoder evaluated in
  the paper and `mlp` the baseline; `feature_dim`, `repeat` and
  `node_out_channels` correspond to the node embedding dimension, layer count
  and channel count reported for it. Seven further decoders are implemented but
  were not used.
- `pose_loss.method` — `known` for supplied orientations, `esl` for estimated
  ones. `gradient` and `mlp_est` are alternatives that were not used.
  `pose_loss.rot_grid_path` selects the SO(3) grid; `rotation_grids/` provides
  grids of 1821, 14761 and 114564 points, of which the paper used 14761. A grid
  may be given as rotation matrices `[Q, 3, 3]` or as quaternions `[Q, 4]`.
- `regulariser` — `cent` penalises displacement of the conformation centroid,
  `dist` the deviation of consecutive residue distances from the template, and
  `local_geom` the deviation of log squared inter-residue distances weighted by
  exponential decay in sequence separation, with `decay` the reciprocal of the
  decay rate. Note that `local_geom` is evaluated on a sliding window of `size`
  residues rather than over all residue pairs. `geom` is an unweighted
  all-pairs variant of the same term.

## Repository layout

| path | contents |
|------|----------|
| `main.py` | entry point |
| `src/cryograph/train.py` | training loop |
| `src/cryograph/pdb_to_graph.py` | template `.pdb` to residue graph |
| `src/cryograph/dataset.py` | particle stack and optional poses/conformations |
| `src/cryograph/models/cryograph_module.py` | the autodecoder and its objective |
| `src/cryograph/models/encoders/` | lookup table and CNN latent maps |
| `src/cryograph/models/decoders/` | graph and MLP conformation decoders |
| `src/cryograph/renderer/` | differentiable forward model and CTF |
| `src/cryograph/pose/` | pose-loss strategies, including ESL |
| `src/cryograph/utils/` | rotations, Kabsch alignment, regulariser indices |
| `cfgs/` | configurations |
| `rotation_grids/` | SO(3) grids for pose search |
| `structs/` | templates, AlphaFold outputs and trajectories |
| `tests/` | test suite |

## Tests

```bash
uv run pytest
```

Tests needing a dataset or a GPU are skipped unless available; see
`tests/README.md`. Point `CRYOGRAPH_TEST_DATA` at a directory laid out as above to
enable the former.
