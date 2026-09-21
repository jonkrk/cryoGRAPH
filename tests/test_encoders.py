"""Coverage for ``cryograph.models.encoders`` — the micrograph -> latent maps."""

import pytest
import torch

from cryograph.models.encoders import CNNEncoder, LookupEncoder
from cryograph.models.encoders.builder import build_encoder

LATENT_DIM = 8
NUM_MRCS = 32
SIDELENGTH = 64
BATCH = 5


def test_lookup_encoder_returns_one_latent_per_index():
    enc = LookupEncoder(latent_dim=LATENT_DIM, num_mrcs=NUM_MRCS)
    idxs = torch.tensor([0, 3, 3, 31, 7])
    mrcs = torch.randn(len(idxs), 1, SIDELENGTH, SIDELENGTH)  # ignored by lookup

    out = enc(idxs, mrcs)

    assert out.shape == (5, LATENT_DIM)
    # same index -> same latent
    torch.testing.assert_close(out[1], out[2])


def test_lookup_encoder_is_trainable():
    enc = LookupEncoder(latent_dim=LATENT_DIM, num_mrcs=NUM_MRCS)
    out = enc(torch.arange(4), torch.randn(4, 1, SIDELENGTH, SIDELENGTH))
    out.pow(2).sum().backward()
    assert enc.table.weight.grad is not None


def test_cnn_encoder_maps_image_batch_to_latent():
    enc = CNNEncoder(
        latent_dim=LATENT_DIM,
        sidelength=SIDELENGTH,
        conv_channels=8,
        act_name="relu",
    )
    mrcs = torch.randn(BATCH, 1, SIDELENGTH, SIDELENGTH)

    out = enc(idxs=torch.arange(BATCH), mrcs=mrcs)

    assert out.shape == (BATCH, LATENT_DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model", ["lookup", "cnn"])
def test_build_encoder_dispatch(model):
    cfg = {
        "network": {
            "latent_dim": LATENT_DIM,
            "encoder": {"model": model, "act_name": "relu", "cnn": {"channels": 8}},
        }
    }
    enc = build_encoder(cfg, num_mrcs=NUM_MRCS, mrc_sidelength=SIDELENGTH)
    expected = {"lookup": LookupEncoder, "cnn": CNNEncoder}[model]
    assert isinstance(enc, expected)


def test_build_encoder_rejects_unknown_model():
    cfg = {"network": {"latent_dim": LATENT_DIM, "encoder": {"model": "nope"}}}
    with pytest.raises(ValueError, match="Unknown encoder"):
        build_encoder(cfg, num_mrcs=NUM_MRCS, mrc_sidelength=SIDELENGTH)
