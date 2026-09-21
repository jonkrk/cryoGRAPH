import os
import torch
from torch.utils.data import Dataset
from jaxtyping import Float
from torch import Tensor
from cryograph.utils import (
    axes_angles_to_rotmats,
    euler_angles_to_rotmats,
    noisy_rotmats,
)

class CryoDataset(Dataset):
    """Preprocessed stack of micrographs of a conformationally heterogeneous protein.

    data.dir must contain micrographs.pt of shape [D, R, R] in the configured
    dtype. It may also contain shifts.pt [D, 2] and confs.pt [D, N, 3], loaded
    when the corresponding load_* flag is set.

    With load_rots the orientations are read from rots.pt [D, 3, 3] if present,
    and otherwise derived from axes.pt [D, 3] with angles.pt [D, 1], or from
    euler_angles.pt [D, 3].

    With use_data_subset the stack is reduced to a random subset, with
    noise_max_angle_degrees > 0 the loaded rotations are perturbed by that much,
    and with normalize the micrographs are standardised as they are read.
    """
    def __init__(
            self,
            config: dict,
            num_residues: int,
        ):
        super().__init__()

        if not os.path.isdir(config['data']['dir']): 
            raise ValueError(f"CryoDataset cannot find data directory: {config['data']['dir']}")

        self.data_dir = config['data']['dir']
        self.dtype = config['data']['dtype']
        self.num_residues = num_residues
        self.load_rots = config['data']['load_rots']
        self.load_shifts = config['data']['load_shifts']
        self.load_confs = config['data']['load_confs']

        self.mrcs = self._load_mrcs(
            config['data']['dir'], 
            config['data']['dtype'],
            config['data']['memory_mapped'],
        )
        if self.load_rots:
            self.rots = self._load_rots(
                config['data']['dir'], 
                config['data']['dtype'],
                config['data']['memory_mapped'],
            )
        if self.load_shifts:
            self.shifts = self._load_shifts(
                config['data']['dir'], 
                config['data']['dtype'],
                config['data']['memory_mapped'],
            )
        if self.load_confs:
            self.confs = self._load_confs(
                config['data']['dir'], 
                config['data']['dtype'],
                config['data']['memory_mapped'],
            )
        
        if config['data']['subset']['use_data_subset']:
            self._reduce_datadim(config['data']['subset']['size'])
        
        if self.load_rots and config['data']['noise_max_angle_degrees'] > 0:
            self._add_rotation_noise(config['data']['noise_max_angle_degrees'])

        self.normalize = config['data'].get('normalize', False)
        if self.normalize:
            self._measure_normalization()

    def __getitem__(self, idx):
        mrc = self.mrcs[idx]
        if self.normalize:
            mrc = (mrc - self.mrc_mean) / self.mrc_std
        out = [idx, mrc]
        if self.load_rots:
            out.append(self.rots[idx])
        if self.load_shifts:
            out.append(self.shifts[idx])
        if self.load_confs:
            out.append(self.confs[idx])
        return tuple(out)

    def __len__(self):
        return self.mrcs.shape[0]

    def get_sidelength(self):
        """Return the micrograph side length in pixels."""
        return self.mrcs.shape[1]
    
    def _load_mrcs(
            self, 
            data_dir: str, 
            dtype: torch.dtype, 
            memory_mapped: bool,
        ) -> Float[Tensor, "D R R"]:
        """Load micrographs.pt and check its dtype and shape."""
        mrcs_path = os.path.join(data_dir, "micrographs.pt")

        if not os.path.exists(mrcs_path): 
            raise ValueError(f"CryoDataset cannot find micrographs.pt in {data_dir}")

        mrcs = torch.load(mrcs_path, weights_only=True, mmap=memory_mapped)

        if mrcs.dtype != dtype: 
            raise ValueError(f"CryoDataset found micrographs.pt with dtype = {mrcs.dtype} != {dtype}")
        if mrcs.ndim != 3:
            raise ValueError(f"CryoDataset found micrographs.pt with ndim = {mrcs.ndim} != 3")
        if mrcs.shape[1] != mrcs.shape[2]: 
            raise ValueError(f"CryoDataset found micrographs.pt with XY dims {mrcs.shape[1]} != {mrcs.shape[2]}")
        return mrcs
    
    def _load_rots(
            self, 
            data_dir: str, 
            dtype: torch.dtype, 
            memory_mapped: bool,
        ) -> Float[Tensor, "D 3 3"]:
        """Load the particle orientations as rotation matrices.

        rots.pt is used when present; otherwise the orientations are derived
        from whichever representation the directory provides.
        """
        rots_path = os.path.join(data_dir, "rots.pt")

        if os.path.exists(rots_path):
            rots = torch.load(rots_path, weights_only=True, mmap=memory_mapped)
        else:
            rots = self._derive_rots(data_dir, memory_mapped)

        if rots.dtype != dtype: 
            raise ValueError(f"CryoDataset found rots.pt with dtype = {rots.dtype} != {dtype}")
        if tuple(rots.shape) != (len(self), 3, 3): 
            raise ValueError(f"CryoDataset found rots.pt with shape {rots.shape} != ({len(self)}, 3, 3)")
        return rots

    def _derive_rots(
            self,
            data_dir: str,
            memory_mapped: bool,
        ) -> Float[Tensor, "D 3 3"]:
        """Build rotation matrices from an alternative orientation representation.

        Accepts axes.pt with angles.pt, or euler_angles.pt, so that a directory
        need not carry a precomputed rots.pt.

        The sign convention of a stored representation is not detectable, and
        the wrong one yields inverse rotations without any error. Euler angles
        are therefore read last, and are taken in the convention of
        euler_angles_to_rotmats: intrinsic Z, new Y, new Z, in radians.
        """
        def load(name):
            return torch.load(
                os.path.join(data_dir, name), weights_only=True, mmap=memory_mapped
            )

        def present(*names):
            return all(os.path.exists(os.path.join(data_dir, n)) for n in names)

        if present("axes.pt", "angles.pt"):
            return axes_angles_to_rotmats(load("axes.pt"), load("angles.pt"))
        if present("euler_angles.pt"):
            return euler_angles_to_rotmats(load("euler_angles.pt"))
        raise ValueError(
            "CryoDataset has load_rots = True but found no orientations in "
            f"{data_dir}: expected rots.pt, axes.pt with angles.pt, or euler_angles.pt"
        )

    def _load_shifts(
            self, 
            data_dir: str, 
            dtype: torch.dtype, 
            memory_mapped: bool,
        ) -> Float[Tensor, "D 2"]:
        """Load shifts.pt and check its shape."""
        shifts_path = os.path.join(data_dir, "shifts.pt")

        if not os.path.exists(shifts_path):
            raise ValueError(f"CryoDataset has load_shifts = True, cannot find shifts.pt in {data_dir}")

        shifts = torch.load(shifts_path, weights_only=True, mmap=memory_mapped)

        # assert shifts.dtype == dtype, (
        #     "CryoDataset found that shifts.pt has ",
        #     f"dtype = {shifts.dtype} != {dtype}",
        # )
        if shifts.shape[0] != len(self) or shifts.shape[1] != 2: 
            raise ValueError(f"CryoDataset found shifts.pt with shape {shifts.shape} != ({len(self)}, 2)")
        return shifts
    
    def _load_confs(
            self, 
            data_dir: str, 
            dtype: torch.dtype, 
            memory_mapped: bool,
        ) -> Float[Tensor, "D N 3"]:
        """Load confs.pt and check its dtype and shape."""
        confs_path = os.path.join(data_dir, "confs.pt")
        if not os.path.exists(confs_path): 
            raise ValueError(f"CryoDataset has load_confs = True, cannot find confs.pt in {data_dir}")

        confs = torch.load(confs_path, weights_only=True, mmap=memory_mapped)

        if confs.dtype != dtype: 
            raise ValueError(f"CryoDataset found confs.pt with torch.dtype != {dtype}")
        if tuple(confs.shape) != (len(self), self.num_residues, 3): 
            raise ValueError(f"CryoDataset found confs.pt with shape: {confs.shape} != ({len(self)}, {self.num_residues}, 3)")
        return confs
    
    def _reduce_datadim(
            self, 
            num_subset: int, 
            shuffle=True,
        ):
        """Restrict the stack to a random subset of num_subset samples."""
        if num_subset > len(self):
            raise ValueError(f"CryoDataset attempted to reduce datadim ({len(self)}) to {num_subset} > {len(self)}")

        if shuffle: 
            indices = torch.randperm(len(self))[0:num_subset]
            self.mrcs = self.mrcs[indices, ...]
            if self.load_rots:
                self.rots = self.rots[indices, ...]
            if self.load_shifts:
                self.shifts = self.shifts[indices, ...]
            if self.load_confs:
                self.confs = self.confs[indices, ...]
        else: 
            raise NotImplementedError(
                "Data subset without shuffle has not been implemented in CryoDataset."
            )
    
    def _measure_normalization(self, max_samples: int = 4096):
        """Estimate the mean and standard deviation used to standardise micrographs.

        Statistics are taken from at most max_samples images so that the estimate
        costs bounded reads on a memory-mapped stack. Standardisation is applied
        per item in __getitem__, which leaves the stack memory-mapped.
        """
        if len(self) > max_samples:
            sample = self.mrcs[torch.randperm(len(self))[:max_samples]]
        else:
            sample = self.mrcs

        self.mrc_mean = sample.mean().item()
        self.mrc_std = sample.std().item()

        if self.mrc_std == 0.:
            raise ValueError(
                "CryoDataset cannot normalise micrographs of zero variance"
            )

    def _add_rotation_noise(
            self,
            max_angle_degrees: float,
        ):
        """Left-multiply the loaded rotations by random rotations of at most
        max_angle_degrees.
        """
        self.rots = torch.einsum(
            'dij, djk -> dik', 
            noisy_rotmats(
                size=(self.rots.shape[0],), 
                max_angle_degrees=max_angle_degrees,
                device=self.rots.device,
                # dtype=self.rots.dtype,
            ),
            self.rots, 
        )
