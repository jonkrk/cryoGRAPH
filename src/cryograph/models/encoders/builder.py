# cryograph/models/encoders/builder.py
from typing import Dict, Any
from . import LookupEncoder, CNNEncoder  # re-exported in package __init__.py

def build_encoder(
    cfg: Dict[str, Any], 
    num_mrcs: int, 
    mrc_sidelength: int
):
    """Construct the encoder named by network.encoder.model."""
    enc = cfg['network']['encoder']
    latent_dim = cfg['network']['latent_dim']
    model = enc['model']
    if model == "lookup":
        return LookupEncoder(latent_dim=latent_dim, num_mrcs=num_mrcs)
    if model == "cnn":
        return CNNEncoder(
            latent_dim=latent_dim,
            sidelength=mrc_sidelength,
            conv_channels=enc['cnn']['channels'],
            act_name=enc['act_name'],
        )
    raise ValueError(f"Unknown encoder model: {model!r}")
