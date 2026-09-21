from typing import Dict, Any
from . import Renderer
import torch_geometric as pyg
from jaxtyping import Float
from torch import Tensor

def build_renderer(
    cfg: Dict[str, Any],
    protein_graph: pyg.data.Data,
    elec_vec: Float[Tensor, "N"],
    std_vec: Float[Tensor, "N"],
    mrc_sidelength: int
) -> Renderer:
    """Construct the renderer described by the rendering section of the config."""
    return Renderer(
        pixel_width=cfg['rendering']['optics']['pixel_width'],
        sidelength=mrc_sidelength,
        ctf_pad=cfg['rendering']['image']['ctf_pad'],
        num_residues=protein_graph['num_nodes'],
        elec_vec=elec_vec,
        std_vec=std_vec,
        spherical_aberration=cfg['rendering']['optics']['spherical_aberration'],
        amplitude_contrast_ratio=cfg['rendering']['optics']['amplitude_contrast_ratio'],
        defocus=cfg['rendering']['optics']['defocus'],
        electron_energy=cfg['rendering']['optics']['electron_energy'],
        init_scale=cfg['rendering']['image']['init_scale'],
        init_bias=cfg['rendering']['image']['init_bias'],
        learn_proj_params=cfg['rendering']['image']['learn_proj_params'],
    )
