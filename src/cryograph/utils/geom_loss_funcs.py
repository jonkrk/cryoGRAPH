import torch

def build_pair_indices(
    num_nodes: int, 
    local_size: int, 
    device=None
):
    """Return index pairs for the geometric regularisers: every i < j pair, and
    separately the pairs within a sliding sequence window of local_size.
    """
    device = device or torch.device("cpu")
    # global i<j pairs
    pairs = torch.combinations(torch.arange(num_nodes, device=device), r=2)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    # local consecutive windows up to distance local_size
    i_local = torch.arange(num_nodes-1, device=device)
    j_local = torch.arange(1, num_nodes, device=device)
    for k in range(1, local_size):
        i_local = torch.cat((i_local, torch.arange(0, num_nodes-1-k, device=device)))
        j_local = torch.cat((j_local, torch.arange(1+k, num_nodes, device=device)))
    return i_idx, j_idx, i_local, j_local

@torch.no_grad()
def build_prior_sqrlogs(coords, i_idx, j_idx):
    """Return the log squared distances of the given coordinate pairs, used as the
    reference the geometric regularisers are measured against.
    """
    # coords: [N,3]
    diffsqr = (coords[i_idx] - coords[j_idx]) ** 2
    return diffsqr.sum(dim=-1).log().unsqueeze(0)
