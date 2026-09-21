"""Coverage for ``cryograph.utils.rotation_funcs`` — conversions between rotation
representations, their round trips, and the quaternion group operations.
"""

import numpy as np
import pytest
import torch

from cryograph.utils.rotation_funcs import (
    axes_angles_to_rotmats,
    euler_angles_to_rotmats,
    invert_quaternions,
    multiply_quaternions,
    quaternion_rotate,
    quaternions_to_rotmats,
    noisy_rotmats,
    rand_quaternions,
    rand_rotmats,
    rotmats_to_axes_angles,
    rotmats_to_quaternions,
    so3_exp_map,
    so3_log_map,
    standardize_quaternions,
)


def _assert_valid_rotmats(m: torch.Tensor, atol: float = 1e-5) -> None:
    eye = torch.eye(3, dtype=m.dtype).expand_as(m)
    torch.testing.assert_close(m @ m.transpose(-1, -2), eye, atol=atol, rtol=0)
    torch.testing.assert_close(
        torch.linalg.det(m), torch.ones(m.shape[:-2], dtype=m.dtype), atol=atol, rtol=0
    )


@pytest.mark.parametrize("shape", [(4,), (2, 8), (3, 5, 7)])
def test_rand_rotmats_shape(shape):
    assert tuple(rand_rotmats(shape).shape) == (*shape, 3, 3)


def test_rand_rotmats_are_proper_rotations():
    _assert_valid_rotmats(rand_rotmats((256,)))


@pytest.mark.parametrize("precise", [False, True])
def test_noisy_rotmats_shape_and_validity(precise):
    m = noisy_rotmats((2, 8), max_angle_degrees=30.0, precise=precise)
    assert tuple(m.shape) == (2, 8, 3, 3)
    _assert_valid_rotmats(m, atol=1e-4)


@pytest.mark.parametrize("precise", [False, True])
def test_noisy_rotmats_respects_max_angle(precise):
    max_deg = 20.0
    m = noisy_rotmats((512,), max_angle_degrees=max_deg, precise=precise)
    trace = torch.diagonal(m, dim1=-2, dim2=-1).sum(-1)
    angle_deg = torch.rad2deg(torch.arccos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)))
    assert angle_deg.max().item() <= max_deg + 1e-3


def test_euler_angles_90_degrees():
    rotmats_90 = torch.tensor(
        [
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            [[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]],
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]],
            [[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]],
        ]
    )
    euler_angles_90 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [np.pi / 2.0, 0.0, 0.0],
            [0.0, np.pi / 2.0, 0.0],
            [np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0],
        ]
    )
    torch.testing.assert_close(
        euler_angles_to_rotmats(euler_angles_90), rotmats_90, atol=1e-6, rtol=1e-5
    )


def test_quaternion_rotate_broadcasts_batch_dims():
    b1, b2 = (2, 3, 4), (5, 6)
    pts = torch.randn(b1 + b2 + (3,))
    quats = torch.randn(b1 + (4,))
    assert quaternion_rotate(quats, pts).shape == pts.shape


def test_quaternion_rotate_matches_matrix_rotation():
    quats = standardize_quaternions(
        torch.nn.functional.normalize(torch.randn(16, 4), dim=-1)
    )
    rotmats = quaternions_to_rotmats(quats)
    pts = torch.randn(16, 3)

    via_quat = quaternion_rotate(quats, pts)
    via_mat = torch.einsum("...ij, ...j -> ...i", rotmats, pts)
    torch.testing.assert_close(via_quat, via_mat, atol=1e-5, rtol=1e-5)


def test_quaternion_rotmat_roundtrip():
    quats = standardize_quaternions(
        torch.nn.functional.normalize(torch.randn(16, 4), dim=-1)
    )
    recovered = rotmats_to_quaternions(quaternions_to_rotmats(quats))
    torch.testing.assert_close(recovered, quats, atol=1e-5, rtol=1e-5)


def test_rand_quaternions_are_unit_norm():
    q = rand_quaternions((4, 16))
    assert tuple(q.shape) == (4, 16, 4)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(4, 16), atol=1e-6, rtol=0)


def test_quaternion_inverse_law():
    q = rand_quaternions((32,))
    identity = multiply_quaternions(q, invert_quaternions(q))
    expected = torch.zeros(32, 4)
    expected[:, 0] = 1.0
    torch.testing.assert_close(standardize_quaternions(identity), expected, atol=1e-6, rtol=0)


def test_quaternion_multiplication_is_rotation_composition():
    q1 = rand_quaternions((16,))
    q2 = rand_quaternions((16,))

    via_quat = quaternions_to_rotmats(multiply_quaternions(q1, q2))
    via_mat = quaternions_to_rotmats(q1) @ quaternions_to_rotmats(q2)
    torch.testing.assert_close(via_quat, via_mat, atol=1e-5, rtol=1e-5)


def test_rotmats_to_axes_angles_roundtrip():
    # random rotations, away from the theta = pi singularity
    rotmats = rand_rotmats((64,))
    recovered = axes_angles_to_rotmats(rotmats_to_axes_angles(rotmats))
    torch.testing.assert_close(recovered, rotmats, atol=1e-4, rtol=0)


def test_rotmats_to_axes_angles_at_180_degrees():
    # theta = pi is the singular branch: R is symmetric, so the axis cannot be
    # read off its antisymmetric part, and the axis sign is ambiguous. The three
    # coordinate axes are covered because each stresses a different entry of
    # (R + I)/2.
    rotmats = torch.stack(
        [
            torch.diag(torch.tensor([1.0, -1.0, -1.0])),
            torch.diag(torch.tensor([-1.0, 1.0, -1.0])),
            torch.diag(torch.tensor([-1.0, -1.0, 1.0])),
        ]
    )
    recovered = axes_angles_to_rotmats(rotmats_to_axes_angles(rotmats))
    torch.testing.assert_close(recovered, rotmats, atol=1e-4, rtol=0)


def test_so3_exp_log_roundtrip():
    # log-rotation vectors with angle comfortably below pi
    v = torch.randn(64, 3)
    v = v / v.norm(dim=-1, keepdim=True) * torch.rand(64, 1) * 3.0
    torch.testing.assert_close(so3_log_map(so3_exp_map(v)), v, atol=1e-4, rtol=0)
