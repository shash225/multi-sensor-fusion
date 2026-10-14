"""Occupancy / cost grids for planning (ego frame, layout ``[iy, ix]``).

* **Dynamic occupancy** ``(T, ny, nx)``: detected objects propagated with their
  predicted velocity (constant-velocity motion forecast) to every planning step.
* **Static geometry**: slow / static objects at t=0 inflated by the ego half-width
  (configuration space for a point-robot A*).
* **Soft costs**: obstacle clearance (distance transform), non-ground LiDAR returns
  (covers missed detections, curbs, poles), and an optional map prior.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from utils.bev_grid import BEVGrid
from utils.raster import rasterize_boxes


def planning_grid(cfg_planning: Any) -> BEVGrid:
    g = cfg_planning.grid
    return BEVGrid(tuple(g.x_range), tuple(g.y_range), (-5.0, 5.0), float(g.resolution))


def propagate_boxes(boxes: np.ndarray, t: float) -> np.ndarray:
    """Constant-velocity forecast of ``[x, y, z, l, w, h, yaw, vx, vy]`` boxes by ``t`` seconds."""
    out = boxes.copy()
    out[:, 0] += boxes[:, 7] * t
    out[:, 1] += boxes[:, 8] * t
    return out


class OccupancyBuilder:
    def __init__(self, cfg_planning: Any) -> None:
        self.cfg = cfg_planning
        self.grid = planning_grid(cfg_planning)
        self.horizon = int(cfg_planning.horizon)
        self.dt = float(cfg_planning.dt)
        ego = cfg_planning.ego
        self.ego_box = np.array([[float(ego.rear_axle_to_center), 0.0, 0.0, float(ego.length), float(ego.width), 1.6, 0.0, 0.0, 0.0]])

    def dynamic_occupancy(self, boxes: np.ndarray, inflate: float = 0.0, times: Optional[np.ndarray] = None) -> np.ndarray:
        """``(T, ny, nx)`` uint8 occupancy at ``t = dt, 2dt, ... T·dt`` (CV motion forecast)."""
        times = times if times is not None else (np.arange(1, self.horizon + 1) * self.dt)
        occ = np.zeros((len(times),) + self.grid.shape, dtype=np.uint8)
        for k, t in enumerate(times):
            if boxes.shape[0]:
                rasterize_boxes(propagate_boxes(boxes, float(t)), self.grid, inflate=inflate, out=occ[k])
        return occ

    def static_occupancy(self, boxes: np.ndarray, inflate: float) -> np.ndarray:
        occ = np.zeros(self.grid.shape, dtype=np.uint8)
        if boxes.shape[0]:
            rasterize_boxes(boxes, self.grid, inflate=inflate, out=occ)
        return occ

    def lidar_obstacles(self, points: Optional[np.ndarray], z_band: tuple[float, float]) -> np.ndarray:
        """Cells with non-ground LiDAR returns (excluding the ego vehicle itself)."""
        occ = np.zeros(self.grid.shape, dtype=np.uint8)
        if points is None or points.shape[0] == 0:
            return occ
        p = points[(points[:, 2] > z_band[0]) & (points[:, 2] < z_band[1])]
        ego = self.ego_box[0]
        in_ego = (np.abs(p[:, 0] - ego[0]) < ego[3] / 2 + 0.5) & (np.abs(p[:, 1]) < ego[4] / 2 + 0.5)
        p = p[~in_ego]
        ix = np.floor((p[:, 0] - self.grid.x_range[0]) / self.grid.resolution).astype(np.int64)
        iy = np.floor((p[:, 1] - self.grid.y_range[0]) / self.grid.resolution).astype(np.int64)
        ok = (ix >= 0) & (ix < self.grid.nx) & (iy >= 0) & (iy < self.grid.ny)
        occ[iy[ok], ix[ok]] = 1
        return occ

    def clearance_cost(self, occ: np.ndarray, scale_m: float = 1.5) -> np.ndarray:
        """``exp(-d / scale)`` where ``d`` is the distance (m) to the nearest occupied cell."""
        if not occ.any():
            return np.zeros(occ.shape, dtype=np.float32)
        dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 5) * self.grid.resolution
        return np.exp(-dist / scale_m).astype(np.float32)

    def footprint_points(self, xy: np.ndarray, yaw: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Points sampled inside the ego footprint ``(N, 15, 2)`` for rear-axle poses ``xy (N, 2)``."""
        ego = self.ego_box[0]
        ls = np.linspace(-ego[3] / 2 - margin, ego[3] / 2 + margin, 5) + ego[0]
        ws = np.linspace(-ego[4] / 2 - margin, ego[4] / 2 + margin, 3)
        local = np.stack(np.meshgrid(ls, ws, indexing="ij"), -1).reshape(-1, 2)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        x = xy[:, 0:1] + local[None, :, 0] * c - local[None, :, 1] * s
        y = xy[:, 1:2] + local[None, :, 0] * s + local[None, :, 1] * c
        return np.stack([x, y], -1)

    def occupied_at(self, occ: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Look up ``occ (ny, nx)`` at metric points ``(..., 2)`` (outside the grid = free)."""
        ix = np.floor((pts[..., 0] - self.grid.x_range[0]) / self.grid.resolution).astype(np.int64)
        iy = np.floor((pts[..., 1] - self.grid.y_range[0]) / self.grid.resolution).astype(np.int64)
        ok = (ix >= 0) & (ix < self.grid.nx) & (iy >= 0) & (iy < self.grid.ny)
        out = np.zeros(pts.shape[:-1], dtype=bool)
        out[ok] = occ[iy[ok], ix[ok]] > 0
        return out
