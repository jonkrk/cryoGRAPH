import torch
from jaxtyping import Float
from torch import Tensor

# Taken from https://hunterheidenreich.com/posts/kabsch_algorithm/#pytorch
def kabsch_torch(
        P: Float[Tensor, "N 3"], 
        Q: Float[Tensor, "N 3"], 
        align=False,
    ) -> tuple[Float[Tensor, "3 3"], Float[Tensor, "3"], Float[Tensor, ""]]:
    """Return the rotation aligning P onto Q, the difference of their centroids,
    and the resulting RMSD. With align=True the aligned copy of P is appended.

    The returned translation is the centroid difference, not the rigid-body
    translation accompanying the rotation.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    Vt[:, -1] *= (torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1)))).sign()

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    rmsd = (torch.matmul(p, R.transpose(0, 1)) - q).square().mean(dim=0).sum().sqrt()
    if align:
        return R, t, rmsd, torch.matmul(p, R.transpose(0, 1)) + centroid_Q
    else:
        return R, t, rmsd