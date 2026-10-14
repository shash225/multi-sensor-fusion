"""Radar view: detections in BEV with ego-motion-compensated velocity arrows (colour = speed)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from visualization.common import BEVCanvas, colorize, put_label


def render_radar_bev(points: np.ndarray, x_range: Sequence[float] = (-50, 50), y_range: Sequence[float] = (-50, 50),
                     ppm: float = 4.0, title: str = "Radar BEV", arrow_seconds: float = 1.0) -> np.ndarray:
    """``points`` columns follow ``RADAR_POINT_FEATURES`` (x, y, z, rcs, vx_comp, vy_comp, ...)."""
    canvas = BEVCanvas(x_range, y_range, ppm)
    if points.shape[0]:
        speed = np.hypot(points[:, 4], points[:, 5])
        colors = colorize(speed, "plasma", -4.0, 10.0)  # offset keeps static returns visible on the dark canvas
        canvas.draw_points(points[:, :2], colors, radius=2)
        moving = speed > 0.5
        for p, v, c in zip(points[moving, :2], points[moving, 4:6], colors[moving]):
            canvas.draw_arrows(p[None], v[None], tuple(int(x) for x in c), scale=arrow_seconds)
    canvas.draw_ego()
    n_moving = int((np.hypot(points[:, 4], points[:, 5]) > 0.5).sum()) if points.shape[0] else 0
    put_label(canvas.img, f"{title}  ({len(points)} pts, {n_moving} moving)", (6, 18), 0.5)
    return canvas.img
