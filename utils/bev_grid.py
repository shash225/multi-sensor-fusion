"""The single Bird's-Eye-View grid definition shared by every module.

Layout convention (identical for camera, LiDAR, radar, fusion, detection, planning):

* BEV tensors are ``(B, C, H, W)`` with ``H = ny`` (rows index **y**, left axis) and
  ``W = nx`` (columns index **x**, forward axis).
* Cell ``(iy, ix)`` covers ``x in [x_min + ix*r, x_min + (ix+1)*r)`` and
  ``y in [y_min + iy*r, y_min + (iy+1)*r)``.
* For display, :func:`visualization.common.bev_to_display` rotates arrays so that
  forward points up and left points left.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BEVGrid:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    resolution: float

    @property
    def nx(self) -> int:
        return int(round((self.x_range[1] - self.x_range[0]) / self.resolution))

    @property
    def ny(self) -> int:
        return int(round((self.y_range[1] - self.y_range[0]) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        """``(H, W) = (ny, nx)``."""
        return self.ny, self.nx

    def downsample(self, stride: int) -> "BEVGrid":
        """Same extent with ``stride``-times coarser cells (e.g. the detection-head grid)."""
        return BEVGrid(self.x_range, self.y_range, self.z_range, self.resolution * stride)

    def world_to_grid(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous grid coordinates (``ix``, ``iy``); cell centres are at ``k + 0.5``."""
        return (np.asarray(x) - self.x_range[0]) / self.resolution, (np.asarray(y) - self.y_range[0]) / self.resolution

    def grid_to_world(self, ix: np.ndarray, iy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(ix) * self.resolution + self.x_range[0], np.asarray(iy) * self.resolution + self.y_range[0]

    def in_range(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x_range[0]) & (x < self.x_range[1]) & (y >= self.y_range[0]) & (y < self.y_range[1])

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x_range[0] + (np.arange(self.nx) + 0.5) * self.resolution
        ys = self.y_range[0] + (np.arange(self.ny) + 0.5) * self.resolution
        return xs, ys

    @classmethod
    def from_config(cls, bev_cfg: Mapping[str, Any]) -> "BEVGrid":
        return cls(
            tuple(float(v) for v in bev_cfg["x_range"]),
            tuple(float(v) for v in bev_cfg["y_range"]),
            tuple(float(v) for v in bev_cfg["z_range"]),
            float(bev_cfg["resolution"]),
        )
