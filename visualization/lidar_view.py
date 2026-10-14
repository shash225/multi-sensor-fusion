"""LiDAR views: raw point cloud in BEV (height-coloured) and the encoder's BEV feature energy."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from visualization.common import BEVCanvas, colorize, put_label


def render_lidar_bev(points: np.ndarray, x_range: Sequence[float] = (-50, 50), y_range: Sequence[float] = (-50, 50),
                     ppm: float = 4.0, title: str = "LiDAR BEV", max_points: int = 120000) -> np.ndarray:
    """Points ``(N, >=3)`` in the ego frame, coloured by height (blue = ground, yellow = high)."""
    canvas = BEVCanvas(x_range, y_range, ppm)
    if points.shape[0] > max_points:
        points = points[np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)]
    if points.shape[0]:
        canvas.draw_points(points[:, :2], colorize(points[:, 2], "viridis", -2.0, 2.5))
    canvas.draw_ego()
    put_label(canvas.img, f"{title}  ({len(points):,} pts)", (6, 18), 0.5)
    return canvas.img


def feature_energy(feat: np.ndarray) -> np.ndarray:
    """L2 norm over channels of a ``(C, H, W)`` feature map -> ``(H, W)``."""
    return np.sqrt((feat.astype(np.float32) ** 2).sum(0))
