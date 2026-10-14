"""Rasterisation of oriented boxes / points onto a :class:`BEVGrid` (array layout ``[iy, ix]``)."""
from __future__ import annotations

import cv2
import numpy as np

from utils.bev_grid import BEVGrid
from utils.geometry import box_corners_bev

_SHIFT = 4  # cv2 fixed-point sub-pixel bits


def rasterize_boxes(boxes: np.ndarray, grid: BEVGrid, inflate: float = 0.0, values: np.ndarray | None = None,
                    out: np.ndarray | None = None) -> np.ndarray:
    """Fill oriented BEV boxes ``[x, y, z, l, w, h, yaw, ...]`` into an ``(ny, nx)`` array.

    ``inflate`` grows length and width by ``2 * inflate`` metres (safety margin).
    """
    canvas = out if out is not None else np.zeros(grid.shape, dtype=np.uint8)
    if boxes.shape[0] == 0:
        return canvas
    b = boxes.astype(np.float64).copy()
    # cv2.fillPoly also fills the pixels on the polygon boundary, which adds ~half a cell on every side;
    # shrinking by one cell makes the filled area match the metric box area.
    b[:, 3:5] = np.maximum(b[:, 3:5] + 2.0 * inflate - grid.resolution, 0.01 * grid.resolution)
    corners = box_corners_bev(b)  # (N, 4, 2) metres
    cols = (corners[..., 0] - grid.x_range[0]) / grid.resolution
    rows = (corners[..., 1] - grid.y_range[0]) / grid.resolution
    poly = np.round(np.stack([cols, rows], -1) * (1 << _SHIFT) - 0.5 * (1 << _SHIFT)).astype(np.int32)
    for i in range(len(poly)):
        value = 1 if values is None else values[i]
        cv2.fillPoly(canvas, [poly[i]], color=value, lineType=cv2.LINE_8, shift=_SHIFT)
    return canvas


def rasterize_points(points: np.ndarray, grid: BEVGrid) -> np.ndarray:
    """Count of points per cell ``(ny, nx)`` (float32)."""
    counts = np.zeros(grid.shape, dtype=np.float32)
    if points.shape[0] == 0:
        return counts
    ix = np.floor((points[:, 0] - grid.x_range[0]) / grid.resolution).astype(np.int64)
    iy = np.floor((points[:, 1] - grid.y_range[0]) / grid.resolution).astype(np.int64)
    ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    np.add.at(counts, (iy[ok], ix[ok]), 1.0)
    return counts
