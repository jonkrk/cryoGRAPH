from torch_geometric.utils import from_networkx
import warnings
import networkx as nx
import mdtraj as md

warnings.filterwarnings("ignore", module="torch_geometric")

def get_pbonds(struct):
    """Return peptide-bond edges between consecutive residues."""
    t = struct.topology
    edges = []
    a1 = t.residue(0)
    for i in range(1, t.n_residues):
        a2 = t.residue(i)
        pbond = (a1, a2, dict(kind='pbond'))
        edges.append(pbond)
        a1 = a2
    return edges

def get_hbonds(struct):
    """Return hydrogen-bond edges between non-adjacent residues, as identified
    by the Baker-Hubbard criterion.
    """
    t = struct.topology
    edges = []
    hbonds = md.baker_hubbard(struct)
    for hbond in hbonds:
        a1 = t.atom(hbond[0]).residue
        a2 = t.atom(hbond[2]).residue
        if abs(a2.index-a1.index) > 1:
            hbond = (a1, a2, dict(kind='hbond'))
            edges.append(hbond)
    return edges

def get_secstructure(struct):
    """Return residue nodes annotated with their DSSP secondary-structure code."""
    dssp = md.compute_dssp(struct)[0]
    t = struct.topology
    nodes = []
    for i in range(t.n_residues):
        node = (t.residue(i), dict(kind=dssp[i]))
        nodes.append(node)
    return nodes

def get_xyz(struct):
    """Return residue nodes annotated with the C-alpha coordinate in Angstrom
    and the residue name.
    """
    t = struct.topology
    nodes = []
    for atom in t.atoms_by_name('CA'):
        xyz = 10 * struct.xyz[0, atom.index, :] # make sure unit is converted to Å
        node = (atom.residue, dict(coords=xyz, residue_name=atom.residue.name))
        nodes.append(node)
    return nodes

def graph_construct(path: str):
    """Build the residue graph of a structure file: one node per residue carrying
    its C-alpha coordinate, with peptide- and hydrogen-bond edges.

    Node order follows residue order, so coords[i] belongs to residue i.
    """
    struct = md.load(path)
    g = nx.Graph()
    g.add_edges_from(get_pbonds(struct))
    g.add_edges_from(get_hbonds(struct))
    g.add_nodes_from(get_secstructure(struct))
    g.add_nodes_from(get_xyz(struct))
    pyg_graph = from_networkx(g)
    return pyg_graph
