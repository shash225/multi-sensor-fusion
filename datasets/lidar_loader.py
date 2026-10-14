"""LiDAR loading: multi-sweep accumulation in the reference frame.

Pipeline: ``.pcd.bin`` (sensor frame) -> remove ego returns -> sensor->ref transform
(motion-compensated through the global frame, per sweep) -> time-lag channel -> range filter.

Output points: ``(N, 5)`` float32 ``[x, y, z, intensity, time_lag]`` in the reference frame.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from utils.bev_grid import BEVGrid
from utils.geometry import apply_transform
from utils.logging_utils import WarnOnce

logger = logging.getLogger(__name__)
_warn = WarnOnce(logger)

LIDAR_POINT_FEATURES = ("x", "y", "z", "intensity", "time_lag")


def read_lidar_bin(path: Path) -> np.ndarray:
    """Read a nuScenes LiDAR file -> ``(N, 5)`` ``[x, y, z, intensity, ring]`` in the sensor frame."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 5 != 0:
        raise ValueError(f"Corrupt LiDAR file (size {data.size} not divisible by 5): {path}")
    return data.reshape(-1, 5)


class LidarLoader:
    """Load and accumulate LiDAR sweeps for one sample.

    Missing or corrupt sweep files are skipped with a warning; if every sweep fails the
    loader returns an empty ``(0, 5)`` array so the model can still run on other sensors.
    """

    def __init__(self, root: Path, grid: BEVGrid, num_sweeps: int = 10, max_points: Optional[int] = None,
                 remove_close_radius: float = 1.5, range_margin: float = 0.0) -> None:
        self.root = Path(root)
        self.grid = grid
        self.num_sweeps = num_sweeps
        self.max_points = max_points
        self.remove_close_radius = remove_close_radius
        self.range_margin = range_margin

    def load_sweep(self, rec: dict[str, Any]) -> Optional[np.ndarray]:
        path = self.root / rec["path"]
        try:
            pts = read_lidar_bin(path)
        except (OSError, ValueError) as exc:
            _warn(str(path), f"Skipping LiDAR sweep {path}: {exc}")
            return None
        if self.remove_close_radius > 0:
            far = np.hypot(pts[:, 0], pts[:, 1]) > self.remove_close_radius
            pts = pts[far]
        out = np.empty((pts.shape[0], 5), dtype=np.float32)
        out[:, :3] = apply_transform(rec["sensor2ref"], pts[:, :3].astype(np.float64))[:, :3]
        out[:, 3] = pts[:, 3]
        out[:, 4] = rec["time_lag"]
        return out

    def load(self, info: dict[str, Any], num_sweeps: Optional[int] = None, rng: Optional[np.random.Generator] = None,
             filter_range: bool = True) -> np.ndarray:
        n = num_sweeps or self.num_sweeps
        chunks = [p for p in (self.load_sweep(rec) for rec in info["lidar"][:n]) if p is not None]
        if not chunks:
            _warn(info["token"], f"No LiDAR data for sample {info['token']}")
            return np.zeros((0, 5), dtype=np.float32)
        points = np.concatenate(chunks, axis=0)
        if filter_range:
            g, m = self.grid, self.range_margin
            keep = (
                (points[:, 0] >= g.x_range[0] - m) & (points[:, 0] < g.x_range[1] + m)
                & (points[:, 1] >= g.y_range[0] - m) & (points[:, 1] < g.y_range[1] + m)
                & (points[:, 2] >= g.z_range[0]) & (points[:, 2] < g.z_range[1])
            )
            points = points[keep]
        if self.max_points and points.shape[0] > self.max_points:
            rng = rng or np.random.default_rng()
            points = points[rng.choice(points.shape[0], self.max_points, replace=False)]
        return points

    def load_keyframe(self, info: dict[str, Any]) -> np.ndarray:
        """Only the key-frame sweep (used for depth supervision and calibration checks)."""
        pts = self.load_sweep(info["lidar"][0])
        return pts if pts is not None else np.zeros((0, 5), dtype=np.float32)
