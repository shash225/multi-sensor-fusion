"""Rigid-body geometry helpers (NumPy).

Conventions used across the whole project
-----------------------------------------
* Quaternions follow nuScenes / pyquaternion order ``[w, x, y, z]``.
* 4x4 homogeneous matrices are named ``<dst>_from_<src>`` or ``<src>2<dst>``:
  ``ego2global`` maps a point in the ego frame to the global frame.
* The **reference frame** of every sample is the ego-vehicle frame at the
  timestamp of the LIDAR_TOP key-frame (x forward, y left, z up, origin at the
  centre of the rear axle on the ground).
* 3D boxes are stored as ``[x, y, z, l, w, h, yaw, vx, vy]``. Note that nuScenes
  stores sizes as ``[w, l, h]``; conversion happens once in the info builder.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def quat_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """Convert a unit quaternion ``[w, x, y, z]`` to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion ``[w, x, y, z]`` (w >= 0)."""
    m = np.asarray(rot, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def transform_matrix(translation: Sequence[float], rotation: Sequence[float] | np.ndarray) -> np.ndarray:
    """Build a 4x4 matrix from a translation and a quaternion (or 3x3 rotation)."""
    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        rot = quat_to_rotmat(rot)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def invert_transform(mat: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""
    inv = np.eye(4)
    rot_t = mat[:3, :3].T
    inv[:3, :3] = rot_t
    inv[:3, 3] = -rot_t @ mat[:3, 3]
    return inv


def apply_transform(mat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to ``(N, 3+)`` points; extra columns are kept unchanged."""
    if points.shape[0] == 0:
        return points.copy()
    out = points.copy()
    out[:, :3] = points[:, :3] @ mat[:3, :3].T + mat[:3, 3]
    return out


def yaw_from_rotmat(rot: np.ndarray) -> float:
    """Heading of the x-axis of ``rot`` projected onto the ground plane (nuScenes ``quaternion_yaw``)."""
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def yaw_from_quat(q: Sequence[float]) -> float:
    return yaw_from_rotmat(quat_to_rotmat(q))


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle (array) to ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """BEV corners of boxes ``(N, >=7)`` ``[x, y, z, l, w, h, yaw, ...]`` -> ``(N, 4, 2)``.

    Corner order: front-left, front-right, rear-right, rear-left (counter-clockwise from front).
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 4, 2))
    x, y, l, w, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    local = np.stack(
        [
            np.stack([l / 2, w / 2], -1),
            np.stack([l / 2, -w / 2], -1),
            np.stack([-l / 2, -w / 2], -1),
            np.stack([-l / 2, w / 2], -1),
        ],
        axis=1,
    )  # (N, 4, 2)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], axis=1)  # (N, 2, 2)
    return np.einsum("nij,nkj->nki", rot, local) + np.stack([x, y], -1)[:, None, :]


def box_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """3D corners ``(N, 8, 3)``; first four on the top face, last four on the bottom face."""
    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3))
    bev = box_corners_bev(boxes)
    z, h = boxes[:, 2], boxes[:, 5]
    top = np.concatenate([bev, np.repeat((z + h / 2)[:, None, None], 4, 1)], -1)
    bottom = np.concatenate([bev, np.repeat((z - h / 2)[:, None, None], 4, 1)], -1)
    return np.concatenate([top, bottom], axis=1)


def points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Boolean mask ``(N_points, N_boxes)`` of points inside 3D yaw-rotated boxes."""
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros((points.shape[0], boxes.shape[0]), dtype=bool)
    rel = points[:, None, :3] - boxes[None, :, :3]  # (P, B, 3)
    c, s = np.cos(-boxes[:, 6]), np.sin(-boxes[:, 6])
    lx = rel[..., 0] * c - rel[..., 1] * s
    ly = rel[..., 0] * s + rel[..., 1] * c
    return (
        (np.abs(lx) <= boxes[:, 3] / 2)
        & (np.abs(ly) <= boxes[:, 4] / 2)
        & (np.abs(rel[..., 2]) <= boxes[:, 5] / 2)
    )


def transform_boxes(boxes: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Transform ``[x, y, z, l, w, h, yaw, vx, vy]`` boxes with a rigid 4x4 transform.

    Velocities are rotated (not translated); yaw is re-extracted from the rotated heading
    vector so pitch/roll in ``mat`` cannot corrupt it.
    """
    if boxes.shape[0] == 0:
        return boxes.copy()
    out = boxes.copy()
    out[:, :3] = boxes[:, :3] @ mat[:3, :3].T + mat[:3, 3]
    heading = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], -1) @ mat[:3, :3].T
    out[:, 6] = np.arctan2(heading[:, 1], heading[:, 0])
    if boxes.shape[1] >= 9:
        vel = np.concatenate([boxes[:, 7:9], np.zeros((len(boxes), 1))], -1) @ mat[:3, :3].T
        out[:, 7:9] = vel[:, :2]
    return out


def project_to_image(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of camera-frame points ``(N, 3)`` -> pixels ``(N, 2)`` and depths ``(N,)``."""
    depth = points_cam[:, 2]
    uvw = points_cam[:, :3] @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, depth


def bev_rotation_matrix(rot_angle: float, scale: float, flip_x: bool, flip_y: bool) -> np.ndarray:
    """4x4 BEV data-augmentation matrix (``bda``): flips, then rotation about z, then isotropic scale.

    ``flip_y`` mirrors the world left/right (y -> -y); ``flip_x`` mirrors forward/backward.
    """
    flip = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0, 1.0])
    rot = np.eye(3)
    rot[:2, :2] = rot2d(rot_angle)
    mat = np.eye(4)
    mat[:3, :3] = scale * rot @ flip
    return mat


def augment_boxes_bev(boxes: np.ndarray, bda: np.ndarray) -> np.ndarray:
    """Apply a BEV augmentation matrix (possibly with reflection and scale) to boxes."""
    if boxes.shape[0] == 0:
        return boxes.copy()
    out = boxes.copy()
    lin = bda[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(lin))))
    out[:, :3] = boxes[:, :3] @ lin.T + bda[:3, 3]
    out[:, 3:6] = boxes[:, 3:6] * scale
    heading = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], -1) @ lin.T
    out[:, 6] = np.arctan2(heading[:, 1], heading[:, 0])
    if boxes.shape[1] >= 9:
        vel = np.concatenate([boxes[:, 7:9], np.zeros((len(boxes), 1))], -1) @ lin.T
        out[:, 7:9] = vel[:, :2]
    return out
