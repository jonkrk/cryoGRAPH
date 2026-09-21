import torch
from torch import nn, fft
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float
from torch import Tensor
from cryograph.utils import quaternion_rotate
#OPTIMISE BY PROJECTING OUT THE Z AXIS AT THE START

class CTF(nn.Module):
    """Contrast transfer function of a weak-phase object, applied as a
    multiplication in Fourier space.
    """
    def __init__(
            self,
            pixel_width: float,
            sidelength: int,
            ctf_pad: int,
            spherical_aberration: float,
            amplitude_contrast_ratio: float,
            defocus: float,
            electron_energy: float,
        ):
        super().__init__()
        
        self.register_buffer(
            "ctf_pad",
            torch.tensor(ctf_pad),
        )
        self.register_buffer(
            "ACR", 
            torch.tensor(amplitude_contrast_ratio),
        )
        self.register_buffer(
            "defocus", 
            torch.tensor(defocus),
        )
        self.register_buffer(
            "aberr", 
            torch.tensor(spherical_aberration*1e7),
        ) # mm -> Ångström
        E = electron_energy * 1e3
        self.register_buffer(
            "wavelen", 
            torch.tensor(12.2639/np.sqrt(E+0.97845e-6*E**2)),
        ) # (keV -> eV, m -> Ångström), https://www.jeol.com/words/emterms/20121023.071258.php
        self.register_buffer(
            "kernel", 
            self.create_kernel(pixel_width, sidelength, ctf_pad),
        )
        
    def compute_ctf(self, normsqr):
        """Evaluate the transfer function at the given squared spatial frequencies."""
        term = 0.25 * self.aberr * (self.wavelen ** 3) * (normsqr ** 2)
        angle = (0.5 * self.defocus * self.wavelen * normsqr - term) * 2. * np.pi
        out = -1. * (
            (1. - self.ACR ** 2).sqrt() * torch.sin(angle) 
            + self.ACR * torch.cos(angle)
        )
        return out
    
    def create_kernel(
            self,
            pixel_width: float,
            sidelength: int,
            ctf_pad: int,
        ) -> Float[Tensor, "R_CTF R_CTF"]:
        """Build the transfer function on the padded image grid, in fftfreq layout."""
        its = fft.fftfreq(sidelength+2*ctf_pad, d=pixel_width)
        x = its.unsqueeze(1)
        y = its.unsqueeze(0)
        normsqr = x ** 2 + y ** 2
        return self.compute_ctf(normsqr)
    
    def forward(
            self, 
            projs: Float[Tensor, "*batch R R"]
        ) -> Float[Tensor, "*batch R R"]:
        """Apply the transfer function, padding and cropping by ctf_pad."""
        projs_pad = F.pad(projs, [self.ctf_pad]*4)
        projs_fourier = fft.fft2(projs_pad) * self.kernel
        projs_ctf = fft.ifft2(projs_fourier)
        if self.ctf_pad > 0:
            return projs_ctf[
                ..., 
                self.ctf_pad:-self.ctf_pad, 
                self.ctf_pad:-self.ctf_pad
            ].real
        else:
            return projs_ctf.real

class Renderer(nn.Module):
    """Differentiable cryo-EM image formation. Poses a conformation, projects its
    residues as Gaussian blobs, applies the contrast transfer function and
    rescales the result to match the observed micrographs.
    """
    def __init__(
            self, 
            pixel_width: float,                   # pixel width (Ångström)
            sidelength: int,                      # pixels in image side
            ctf_pad: int,                         # padding for CTF oversample
            num_residues: int,                    # number of amino residues 
            elec_vec: Float[Tensor, "num_residues"], # electrons per residue 
            std_vec: Float[Tensor, "num_residues"],  # std dev per residue
            spherical_aberration: float,          # spherical aberration (mm)
            amplitude_contrast_ratio: float,      # amplitude contrast ratio 
            defocus: float,                       # negative defocus (Ångström)
            electron_energy: float,               # electron energy (keV)
            init_scale: float,                    # initial scale of projections
            init_bias: float,                     # initial bias of projections
            learn_proj_params: bool,              # learn scale and bias
        ):
        super().__init__()

        self.ctf = CTF(
            pixel_width=pixel_width,
            sidelength=sidelength,
            ctf_pad=ctf_pad,
            spherical_aberration=spherical_aberration,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            defocus=defocus,
            electron_energy=electron_energy,
        )

        # self.pad_fac = 2
        # self.register_buffer("zero_image", torch.zeros((1, 1, sidelength, sidelength)))
        # t = torch.linspace(
        #     -pixel_width*(sidelength-1)/2, 
        #     pixel_width*(sidelength-1)/2, 
        #     sidelength//self.pad_fac,
        # )

        t = torch.linspace(
            -pixel_width*(sidelength-1)/2, 
            pixel_width*(sidelength-1)/2, 
            sidelength,
        )
        self.register_buffer("plane", t.repeat(2, 1).T.repeat(num_residues, 1, 1))
        self.register_buffer("elec_vec_sqrt", torch.sqrt(elec_vec))
        self.register_buffer("std_fac", 1./(2.*(std_vec**2)))
        self.register_buffer("std_fac_div_sqrt", (self.std_fac/np.pi).sqrt())
        self.register_buffer("init_scale", torch.tensor(init_scale))
        self.register_buffer("init_bias", torch.tensor(init_bias))
        if learn_proj_params:
            self.register_parameter("scale", nn.Parameter(torch.tensor(1.)))
            self.register_parameter("bias", nn.Parameter(torch.tensor(0.)))  
        else:
            self.register_buffer("scale", torch.tensor(1.))
            self.register_buffer("bias", torch.tensor(0.)) 

    def forward(
            self,
            confs: Float[Tensor, "B N 3"],
            rots: Float[Tensor, "B Q 3 3"] | Float[Tensor, "B Q 4"] | None,
            shifts: Float[Tensor, "B Q 2"] | None, # None PLACEHOLDER
        ) -> Float[Tensor, "B Q R R"]:
        """Render one image per pose. Rotations are read as rotation matrices when
        4-dimensional and as quaternions when 3-dimensional; None leaves the
        conformation unposed.
        """
        if rots is None:
            posed_confs = confs
        elif rots.ndim == 4:
            posed_confs = self._pose_mat(confs, rots, shifts)
        elif rots.ndim == 3:
            posed_confs = self._pose_quat(confs, rots, shifts)
        else:
            raise ValueError(f"Renderer got rots with {rots.ndim = }")
        return self.project(posed_confs)

    def project(
            self,
            posed_confs: Float[Tensor, "B Q N 3"],
        ) -> Float[Tensor, "B Q R R"]:
        """Project posed residues onto the image plane and apply the transfer function.

        Each residue is an isotropic Gaussian normalised to its electron count, so
        integrating along z leaves a separable 2D Gaussian and the image is formed as
        an outer product of per-axis marginals rather than through a dense volume.
        """
        res = torch.square(self.plane - posed_confs[..., None, :2])
        res = torch.einsum('bqnki, n -> bqnki', res, self.std_fac)
        res = torch.exp(-res)
        res = torch.einsum('bqnki, n -> bqnki', res, self.elec_vec_sqrt)
        res = torch.einsum('bqnki, n -> bqnki', res, self.std_fac_div_sqrt)
        out = torch.einsum('bqnk, bqnl -> bqlk', res[..., 0], res[..., 1]) # IMG XY FLIP
        # tmp = torch.einsum('bqnk, bqnl -> bqlk', res[..., 0], res[..., 1]) # IMG XY FLIP
        # out = self.zero_image.expand(tmp.shape[0], tmp.shape[1], -1, -1).clone()
        # out[..., out.shape[2]//2-out.shape[2]//(2*self.pad_fac):out.shape[2]//2+out.shape[2]//(2*self.pad_fac), out.shape[3]//2-out.shape[3]//(2*self.pad_fac):out.shape[3]//2+out.shape[3]//(2*self.pad_fac)] += tmp
        out = self.ctf(out)
        out = (out - self.init_bias) / self.init_scale
        out = (out - self.bias) / self.scale
        return out
    
    def _pose_mat(
            self,
            confs: Float[Tensor, "B N 3"],
            rotmats: Float[Tensor, "B Q 3 3"],
            shifts: Float[Tensor, "B Q 2"] | None, # None PLACEHOLDER
        ) -> Float[Tensor, "B Q N 3"]:
        """Rotate a conformation by rotation matrices, then shift in the image plane."""
        posed_confs = torch.einsum('bqij, bnj -> bqni', rotmats, confs)
        if shifts is not None:
            posed_confs[..., :2] += shifts
        return posed_confs

    def _pose_quat(
            self,
            confs: Float[Tensor, "B N 3"],
            quats: Float[Tensor, "B Q 4"],
            shifts: Float[Tensor, "B Q 2"] | None, # None PLACEHOLDER
        ) -> Float[Tensor, "B Q N 3"]:
        """Rotate a conformation by quaternions, then shift in the image plane."""
        posed_confs = quaternion_rotate(quats, confs.unsqueeze(1))
        if shifts is not None:
            posed_confs[..., :2] += shifts
        return posed_confs
