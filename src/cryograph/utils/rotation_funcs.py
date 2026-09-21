import torch
import numpy as np
from scipy.optimize import fsolve
from jaxtyping import Float
from torch import Tensor

def rand_quaternions(
        size: tuple[int, ...],
        device='cpu',
        generator=None,
    )-> Float[Tensor, "*batch 4"]:
    """Return unit quaternions distributed uniformly over SO(3) with respect to the
    Haar measure.
    """
    quats = torch.randn(size+(4,), device=device, generator=generator)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)
    return quats

def rand_rotmats(
        size: tuple[int, ...],
        device='cpu',
        generator=None,
    )-> Float[Tensor, "*batch 3 3"]:
    """Return rotation matrices distributed uniformly over SO(3) with respect to the
    Haar measure.
    """
    return quaternions_to_rotmats(rand_quaternions(size, device, generator))

def noisy_rotmats(
        size: tuple[int, ...],
        max_angle_degrees: float,
        device='cpu',
        generator=None,
        verbose=False,
        precise=False,
    )-> Float[Tensor, "*batch 3 3"]:
    """Return rotation matrices with angles up to max_angle_degrees, approximately
    Haar-uniform on that ball.

    Angles are drawn by inverting the cumulative distribution: precise=True solves
    it numerically, otherwise a small-angle Taylor approximation is used.
    """
    axes = torch.randn(
        size=size+(3,), 
        device=device, 
        generator=generator,
    )
    axes = axes / torch.norm(axes, dim=-1).unsqueeze(-1)
    max_angle = 2 * np.pi * max_angle_degrees / 360.
    if verbose:
        print(
            f"Logging noisy_rotmats with {size = },",
            f"{max_angle_degrees = }",
            f"and {precise = }.",
        )
        mean_angle = (max_angle ** 2) / 2. 
        mean_angle += 1 - max_angle * np.sin(max_angle) - np.cos(max_angle)
        mean_angle /= (max_angle - np.sin(max_angle))
        print(
            "\tExpected mean angle in degrees:          ",
            f"{mean_angle * 360. / (2 * np.pi):.2f}",
        )

    if precise or max_angle_degrees > 30.:
        x = np.zeros(size).flatten() * (max_angle - np.sin(max_angle))
        angles_np = np.zeros(size).flatten()
        for idx in range(x.shape[0]): # Slow
            x_tmp = np.random.random_sample() * (max_angle - np.sin(max_angle))
            angles_np[idx] = fsolve(
                lambda theta : theta - np.sin(theta) - x_tmp, 
                x_tmp, 
                fprime=lambda theta : 1 - np.cos(theta),
            ).item()
            x[idx] = x_tmp
        # Absolute tolerance; relative may be better.
        failed_sol = np.abs(angles_np-np.sin(angles_np)-x) > 1e-4
        num_resolves = 0
        while np.any(failed_sol):
            x_f = np.zeros((x[failed_sol].shape[0],))
            angles_np_f = np.zeros(x_f.shape[0])
            for idx in range(x_f.shape[0]): # Slow
                x_f_tmp = np.random.random_sample() * (max_angle - np.sin(max_angle))
                angles_np_f[idx] = fsolve(
                    lambda theta : theta - np.sin(theta) - x_f_tmp,
                    x_f_tmp, 
                    fprime=lambda theta : 1 - np.cos(theta),
                ).item()
                x_f[idx] = x_f_tmp
            angles_np[failed_sol] = angles_np_f
            x[failed_sol] = x_f
            failed_sol[failed_sol] = np.abs(angles_np_f-np.sin(angles_np_f)-x_f) > 1e-4
            num_resolves += 1
            if num_resolves >= 100:
                raise ArithmeticError(
                    "Number of resolves performed by noisy_rotmats exceeded 100."
                )
        angles = torch.from_numpy(
            angles_np.reshape(size)
        ).to(device=device, dtype=torch.float32).unsqueeze(-1)
    else:
        # Taylor approximation of x-sin(x) with inverse of cumulative distribution.
        unif = torch.rand(
            size=size+(1,), 
            device=device, 
            generator=generator
        ) * 6. * (max_angle ** 3)
        angles = (unif / 6.) ** (1/3)
     
    if verbose:
        axes_xy = axes[..., :2]
        angles_xy = angles * torch.sum(axes_xy**2, dim=-1, keepdim=True)
        print(
            "\tSample max angle in degrees:             ",
            f"{torch.max(angles).item()*360/(2*np.pi):.2f}",
        )
        print(
            "\tSample mean angle in degrees:            ",
            f"{torch.mean(angles).item()*360/(2*np.pi):.2f}",
        )
        print(
            "\tSample stdev angle in degrees:           ",
            f"{torch.std(angles).item()*360/(2*np.pi):.2f}",
        )
        print(
            "\tSample mean projected angle in degrees:  ",
            f"{torch.mean(angles_xy)*360/(2*np.pi):.2f}",
        )
    return axes_angles_to_rotmats(axes, angles)

def axes_angles_to_rotmats(
        axes: Float[Tensor, "*batch 3"], 
        angles: Float[Tensor, "*batch 1"] | None = None,
    ) -> Float[Tensor, "*batch 3 3"]:
    """Convert an axis and angle to a rotation matrix. When angles is None the axes
    are read as axis-angle vectors, whose norm is the angle.
    """
    return quaternions_to_rotmats(axes_angles_to_quaternions(axes, angles))

def axes_angles_to_quaternions(
        axes: Float[Tensor, "*batch 3"], 
        angles: Float[Tensor, "*batch 1"] | None = None,
    ) -> Float[Tensor, "*batch 4"]:
    """Convert an axis and angle to a unit quaternion. When angles is None the axes
    are read as axis-angle vectors, whose norm is the angle.
    """
    if angles is None:
        angles = torch.norm(axes, dim=-1, keepdim=True)
        axes_angles = axes
    else:
        axes_angles = axes * angles
    # half_angles = angles * 0.5
    # eps = 1e-6
    # small = angles.abs() < eps
    # sin_half_angle_div = torch.empty_like(angles)
    # sin_half_angle_div[~small] = torch.sin(half_angles[~small]) / angles[~small]
    # sin_half_angle_div[small] = 0.5 - (angles[small] * angles[small]) / 48
    # axes_angles = axes * angles
    # quats = torch.cat(
    #     [
    #         torch.cos(half_angles), 
    #         axes_angles*sin_half_angle_div,
    #     ], 
    #     dim=-1,
    # )
    sin_half_angs_div_angs = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([torch.cos(angles*0.5), axes_angles*sin_half_angs_div_angs], dim=-1)

def quaternions_to_rotmats(
        quats: Float[Tensor, "*batch 4"]
    ) -> Float[Tensor, "*batch 3 3"]:
    """Convert quaternions to rotation matrices. The quaternions need not be
    normalised.
    """
    qr, qi, qj, qk = torch.unbind(quats, dim=-1)
    two_s = 2. / quats.square().sum(dim=-1)
    rotmats = torch.stack(
        [
            1. - two_s * (qj * qj + qk * qk),
            two_s * (qi * qj - qk * qr),
            two_s * (qi * qk + qj * qr),
            two_s * (qi * qj + qk * qr),
            1. - two_s * (qi * qi + qk * qk),
            two_s * (qj * qk - qi * qr),
            two_s * (qi * qk - qj * qr),
            two_s * (qj * qk + qi * qr),
            1. - two_s * (qi * qi + qj * qj),
        ],
        dim=-1,
    )
    return rotmats.reshape(quats.shape[:-1] + (3, 3))

def rotmats_to_quaternions(
        rotmats: Float[Tensor, "*batch 3 3"],
    ) -> Float[Tensor, "*batch 4"]:
    """Convert rotation matrices to unit quaternions in standard form."""
    return standardize_quaternions(
        axes_angles_to_quaternions(rotmats_to_axes_angles(rotmats), None)
    )

def rotmats_to_axes_angles(
        rotmats: Float[Tensor, "*batch 3 3"],
    ) -> Float[Tensor, "*batch 3"]:
    """Convert rotation matrices to axis-angle vectors, whose norm is the angle.

    At an angle of pi the axis is recovered from (R + I) / 2 and its sign is
    undetermined.
    """

    if rotmats.shape[-1] != 3 or rotmats.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix tensor shape: {rotmats.shape}")

    omgs = torch.stack(
        [
            rotmats[..., 2, 1] - rotmats[..., 1, 2],
            rotmats[..., 0, 2] - rotmats[..., 2, 0],
            rotmats[..., 1, 0] - rotmats[..., 0, 1],
        ],
        dim=-1,
    )
    traces = torch.diagonal(rotmats, dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(-1)
    angles = torch.atan2(torch.norm(omgs, dim=-1, keepdim=True), traces-1)

    zeros = torch.zeros(3, dtype=rotmats.dtype, device=rotmats.device)
    omgs = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omgs)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axes_angles = torch.empty_like(omgs)
    axes_angles[~near_pi] = 0.5 * (omgs[~near_pi] / torch.sinc(angles[~near_pi] / np.pi))

    # Near theta = pi, R is symmetric and omgs -> 0. Recover the axis from
    # n n^T = (R + I) / 2: its diagonal holds n_i^2, and the row with the
    # largest diagonal entry is the most numerically stable multiple of n
    # (that entry is >= 1/3 for any unit axis, so no division by ~0). The axis
    # sign is inherently ambiguous at theta = pi.
    eye = torch.eye(3, dtype=rotmats.dtype, device=rotmats.device)
    nnT = 0.5 * (rotmats[near_pi] + eye)
    k = torch.diagonal(nnT, dim1=-2, dim2=-1).argmax(dim=-1)
    rows = torch.arange(nnT.shape[0], device=nnT.device)
    axis = nnT[rows, k]
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    axes_angles[near_pi] = angles[near_pi] * axis

    return axes_angles

def euler_angles_to_rotmats(
        euler_angles: Float[Tensor, "*batch 3"]
    ) -> Float[Tensor, "*batch 3 3"]:
    """Convert Euler angles in radians to rotation matrices, right-handed, in the
    Relion convention.

    The rotations are applied about the Z-axis (rlnAngleRot), the new Y-axis
    (rlnAngleTilt) and the new Z-axis (rlnAnglePsi).
    """
    xyz = torch.eye(
        n=3, 
        device=euler_angles.device, 
        dtype=euler_angles.dtype
    ).expand(euler_angles.shape+(3,))
    rotmats_1 = axes_angles_to_rotmats(xyz[..., :, 2], euler_angles[..., None, 0])
    rotmats_2 = axes_angles_to_rotmats(xyz[..., :, 1], euler_angles[..., None, 1])
    rotmats_3 = axes_angles_to_rotmats(xyz[..., :, 2], euler_angles[..., None, 2])
    rotmats = torch.einsum(
        "...ik, ...kl -> ...il", 
        torch.einsum(
            "...ij, ...jk -> ...ik", 
            rotmats_3, 
            rotmats_2,
        ), 
        rotmats_1,
    )
    return rotmats

def so3_exp_map(
        log_rot: Float[Tensor, "*batch 3"], 
        # eps: float = 0.0001,
    ) -> Float[Tensor, "*batch 3 3"]:
    """Convert axis-angle vectors to rotation matrices by Rodrigues' formula.

    Singular at zero, where the axis is undefined.
    """
    if log_rot.shape[-1] != 3:
        raise ValueError("Input tensor shape[0] has to be 3.")

    # nrms = (log_rot * log_rot).sum(1)
    # rot_angles = torch.clamp(nrms, eps).sqrt()
    # skews = hat(log_rot)
    # skews_square = torch.bmm(skews, skews)

    log_rot_angles = log_rot.norm(dim=-1, keepdim=True)
    log_rot_axes = log_rot / log_rot_angles

    rotmats = axes_angles_to_rotmats(log_rot_axes, log_rot_angles)

    return rotmats

def so3_log_map(
        rotmats: Float[Tensor, "*batch 3 3"], 
    ) -> Float[Tensor, "*batch 3"]:
    """Convert rotation matrices to axis-angle vectors.

    Singular at the identity, where the axis is undefined.
    """
    return rotmats_to_axes_angles(rotmats)

def hat(
        v: Float[Tensor, "*batch 3"],
    ) -> Float[Tensor, "*batch 3 3"]:
    """Return the skew-symmetric matrix of a batch of 3D vectors, such that
    hat(a) @ b is the cross product of a and b.

    Accepts a single leading batch dimension only, despite the annotation.
    """

    if v.shape[-1] != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros(tuple(v.shape[:-1])+(3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[..., 0, 1] = -z
    h[..., 0, 2] = y
    h[..., 1, 0] = z
    h[..., 1, 2] = -x
    h[..., 2, 0] = -y
    h[..., 2, 1] = x

    return h

def multiply_quaternions(
        a: Float[Tensor, "*#batch 4"],
        b: Float[Tensor, "*#batch 4"],
    ) -> Float[Tensor, "*batch 4"]:
    """Return the Hamilton product of two quaternions, i.e. the composition of the
    rotations they represent.
    """
    ar, ai, aj, ak = torch.unbind(a, dim=-1)
    br, bi, bj, bk = torch.unbind(b, dim=-1)
    cr = ar * br - ai * bi - aj * bj - ak * bk
    ci = ar * bi + ai * br + aj * bk - ak * bj
    cj = ar * bj - ai * bk + aj * br + ak * bi
    ck = ar * bk + ai * bj - aj * bi + ak * br
    return torch.stack((cr, ci, cj, ck), dim=-1)

def invert_quaternions(quats: Float[Tensor, "*batch 4"]) -> Float[Tensor, "*batch 4"]:
    """Return the conjugate quaternions, which invert the rotation for unit input."""
    signs = torch.tensor([1, -1, -1, -1], device=quats.device)
    return torch.einsum('...i, i -> ...i', quats, signs)

def quaternion_rotate(
        quats: Float[Tensor, "*quat_batch 4"],
        pts: Float[Tensor, "*point_batch 3"],
    ) -> Float[Tensor, "*out_batch 3"]:
    """Rotate 3D points by quaternions.

    The batch axes of quats align with the leading batch axes of pts and
    broadcast against them; any remaining trailing axes of pts are carried
    through. Writing Q = quats.shape[:-1] and P = pts.shape[:-1], the result is

        out_batch = broadcast(Q, P[:len(Q)]) + P[len(Q):]

    so the output is not in general the shape of pts: quats may supply an axis
    where pts has a singleton. The three batch shapes are named separately
    because jaxtyping permits only one variadic axis per annotation.
    """
    if pts.shape[-1] != 3:
        raise ValueError(f"Pts are not in 3D: {pts.shape = }.")
    real_parts = pts.new_zeros(pts.shape[:-1] + (1,))
    pts_as_quats = torch.cat([real_parts, pts], dim=-1)
    if pts.ndim > quats.ndim:
        quats_view = quats.view(tuple(quats.shape[:-1])+(1,)*(pts.ndim-quats.ndim)+(4,))
    else:
        quats_view = quats
    out = multiply_quaternions(
        multiply_quaternions(quats_view, pts_as_quats),
        invert_quaternions(quats_view),
    )
    return out[..., 1:]

def standardize_quaternions(quats: Float[Tensor, "*batch 4"]) -> Float[Tensor, "*batch 4"]:
    """Return the representative of each quaternion whose real part is non-negative."""
    return torch.where(quats[..., 0:1] < 0, -quats, quats)
