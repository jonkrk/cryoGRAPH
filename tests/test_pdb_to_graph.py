"""Coverage for ``cryograph.pdb_to_graph`` — the template ``.pdb`` to residue graph.

Asserts the properties the rest of the pipeline depends on: one node per residue,
coordinates in Angstrom, a valid undirected ``edge_index``, and a peptide-bonded
backbone whose node order matches residue order (``coords[i]`` <-> residue ``i``).
"""

import torch

from cryograph.pdb_to_graph import graph_construct

# ADK monomer: 214 residues, hence 214 CA nodes.
EXPECTED_NODES = 214


def test_graph_construct_node_count_and_features(structure_path):
    g = graph_construct(str(structure_path))
    n = int(g["num_nodes"])

    assert n == EXPECTED_NODES
    assert tuple(g["coords"].shape) == (n, 3)
    assert len(g["residue_name"]) == n
    assert torch.isfinite(g["coords"]).all()


def test_edge_index_is_valid_and_undirected(structure_path):
    g = graph_construct(str(structure_path))
    n = int(g["num_nodes"])
    edge_index = g["edge_index"]

    assert edge_index.shape[0] == 2
    assert int(edge_index.min()) >= 0
    assert int(edge_index.max()) < n

    edges = {(int(a), int(b)) for a, b in edge_index.t()}
    assert all((b, a) in edges for a, b in edges), "edge_index is not symmetric"
    assert all(a != b for a, b in edges), "unexpected self-loop in edge_index"


def test_backbone_follows_residue_order(structure_path):
    """Every pair of consecutive residues must be peptide-bonded.

    This guards the fragile assumption in ``graph_construct`` that
    ``from_networkx`` preserves residue insertion order.
    """
    g = graph_construct(str(structure_path))
    n = int(g["num_nodes"])
    edges = {(int(a), int(b)) for a, b in g["edge_index"].t()}

    missing = [i for i in range(n - 1) if (i, i + 1) not in edges]
    assert not missing, f"backbone bond missing between residues {missing}"


def test_coords_are_in_angstrom(structure_path):
    g = graph_construct(str(structure_path))
    span = g["coords"].max(dim=0).values - g["coords"].min(dim=0).values
    # A 214-residue domain spans tens of Angstrom - not nm (~a few) and not
    # some absurdly large unit.
    assert (span > 10.0).all()
    assert (span < 1000.0).all()
