"""Coverage for ``cryograph.models.layers`` — the hand-rolled message passers.

Each layer is constructed and run forward, and its output shapes and finiteness
are checked. The leading batch dimension is what makes these layers non-standard
for PyG, so it is exercised explicitly.
"""

import torch
import torch.nn as nn

from cryograph.models.layers import EGCLMessage, MLPMessage

# 3-node path graph, bidirectional edges: 0 <-> 1 <-> 2
EDGE_INDEX = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]]).t().contiguous()


def test_mlp_message_forward_shape():
    h = torch.tensor([[[1.0, 0, 0, 0], [2, 1, 0, 0], [3, 0, 1, 0]]])  # [1, 3, 4]

    layer = MLPMessage(
        node_features_in=4,
        node_features_out=5,
        num_layers_aggregate=3,
        num_layers_message=2,
        act=nn.ReLU(),
    )
    out = layer(EDGE_INDEX, h)

    assert tuple(out.shape) == (1, 3, 5)
    assert torch.isfinite(out).all()


def test_egcl_message_forward_shapes():
    h = torch.tensor([[[1.0, 0, 0, 0, 7], [2, 1, 0, 0, 8], [3, 0, 1, 0, 4]]])  # [1, 3, 5]
    x = torch.tensor([[[9.0, 7, 0], [4, 7, 3], [1, 2, 4]]])  # [1, 3, 3]

    layer = EGCLMessage(
        node_in_channels=5,
        node_out_channels=5,
        edge_embed_layers=2,
        edge_embed_hidden_channels=5,
        edge_embed_channels=5,
        message_layers=2,
        message_hidden_channels=5,
        update_layers=2,
        update_hidden_channels=5,
        act=nn.ReLU(),
    )
    out_h, out_x = layer(h, x, EDGE_INDEX)

    assert tuple(out_h.shape) == (1, 3, 5)
    assert tuple(out_x.shape) == (1, 3, 3)
    assert torch.isfinite(out_h).all()
    assert torch.isfinite(out_x).all()
