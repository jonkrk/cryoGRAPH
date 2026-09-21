"""Coverage for ``cryograph.models.decoders`` — the latent -> conformation maps.

Every decoder shares one contract: ``forward(latents[B, L]) -> confs[B, N, 3]``.
This builds each variant through ``build_decoder`` (so the config wiring is
exercised too) and checks the contract.
"""

import pytest
import torch

from cryograph.models.decoders import Decoder
from cryograph.models.decoders.builder import build_decoder

ALL_DECODERS = [
    "mlp",
    "gnn",
    "egcn",
    "gcnconv",
    "sageconv",
    "graphconv",
    "uniconv",
    "gineconv",
    "edge_gnn",
]

BATCH = 3


def _build(base_config, tiny_graph, model):
    base_config["network"]["decoder"]["model"] = model
    return build_decoder(base_config, protein_graph=tiny_graph)


@pytest.mark.parametrize("model", ALL_DECODERS)
def test_decoder_maps_latent_to_conformation(base_config, tiny_graph, model):
    decoder = _build(base_config, tiny_graph, model)
    assert isinstance(decoder, Decoder)

    latent_dim = base_config["network"]["latent_dim"]
    n = int(tiny_graph["num_nodes"])
    latents = torch.randn(BATCH, latent_dim)

    confs = decoder(latents)

    assert confs.shape == (BATCH, n, 3)
    assert torch.isfinite(confs).all()


@pytest.mark.parametrize("model", ALL_DECODERS)
def test_decoder_is_differentiable(base_config, tiny_graph, model):
    decoder = _build(base_config, tiny_graph, model)
    latents = torch.randn(BATCH, base_config["network"]["latent_dim"], requires_grad=True)

    decoder(latents).pow(2).sum().backward()

    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


@pytest.mark.parametrize("model", ALL_DECODERS)
def test_decoder_output_is_structure_scale(base_config, tiny_graph, model):
    """Output should be on the same length scale as the reference conformation
    (Angstrom), i.e. neither collapsed to a point nor exploded."""
    decoder = _build(base_config, tiny_graph, model)
    decoder.eval()

    ref_scale = decoder.base_coords.std().item()
    confs = decoder(torch.zeros(2, base_config["network"]["latent_dim"]))
    out_scale = confs.std().item()

    assert 0.1 * ref_scale < out_scale < 10.0 * ref_scale


def test_build_decoder_rejects_unknown_model(base_config, tiny_graph):
    base_config["network"]["decoder"]["model"] = "nope"
    with pytest.raises(ValueError, match="Unknown decoder"):
        build_decoder(base_config, protein_graph=tiny_graph)
