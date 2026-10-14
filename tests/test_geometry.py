"""Geometry conventions (quaternions, transforms, boxes, BEV augmentation, rasterisation)."""
from __future__ import annotations

import numpy as np

from utils.bev_grid import BEVGrid
from utils.geometry import (augment_boxes_bev, bev_rotation_matrix, box_corners_bev, invert_transform, points_in_boxes,
                            quat_to_rotmat, rotmat_to_quat, transform_boxes, transform_matrix, yaw_from_quat)
from utils.raster import rasterize_boxes


def test_quaternion_roundtrip_and_yaw() -> None:
    yaw = 0.7
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # rotation about z, [w, x, y, z]
    assert np.isclose(yaw_from_quat(q), yaw)
    r = quat_to_rotmat(q)
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.allclose(quat_to_rotmat(rotmat_to_quat(r)), r)


def test_invert_transform() -> None:
    m = transform_matrix([1.0, -2.0, 0.5], [0.9, 0.1, -0.2, 0.37])
    assert np.allclose(m @ invert_transform(m), np.eye(4))


def test_transform_boxes_rotates_yaw_and_velocity() -> None:
    box = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 5.0, 0.0]])
    m = transform_matrix([0, 0, 0], quat_to_rotmat([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))  # +90 deg
    out = transform_boxes(box, m)
    assert np.allclose(out[0, :2], [0.0, 10.0], atol=1e-9)
    assert np.isclose(out[0, 6], np.pi / 2)
    assert np.allclose(out[0, 7:9], [0.0, 5.0], atol=1e-9)


def test_points_in_boxes() -> None:
    box = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 2.0, np.pi / 2]])  # rotated: long axis along y
    pts = np.array([[0.0, 1.9, 0.0], [1.9, 0.0, 0.0]])
    inside = points_in_boxes(pts, box)[:, 0]
    assert inside.tolist() == [True, False]


def test_mirror_augmentation_flips_yaw_and_vy() -> None:
    bda = bev_rotation_matrix(0.0, 1.0, flip_x=False, flip_y=True)
    box = np.array([[5.0, 3.0, 0.0, 4.0, 2.0, 1.5, 0.3, 1.0, 2.0]])
    out = augment_boxes_bev(box, bda)
    assert np.allclose(out[0, :2], [5.0, -3.0])
    assert np.isclose(out[0, 6], -0.3)
    assert np.allclose(out[0, 7:9], [1.0, -2.0])


def test_rasterize_box_area() -> None:
    grid = BEVGrid((-10, 10), (-10, 10), (-5, 5), 0.5)
    occ = rasterize_boxes(np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]]), grid)
    assert abs(occ.sum() * 0.25 - 8.0) <= 2.0  # 4 m x 2 m = 8 m^2 (cell 0.25 m^2)
    corners = box_corners_bev(np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]]))[0]
    assert np.allclose(corners[0], [2.0, 1.0])  # front-left
