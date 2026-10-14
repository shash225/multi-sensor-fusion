"""Planning metrics against the logged (human) ego trajectory.

* **ADE / FDE**: mean / final L2 displacement over valid steps.
* **L2 @ 1 s, 2 s, 3 s**: displacement error at those horizons (ST-P3 / UniAD protocol).
* **Collision rate**: fraction of samples whose ego footprint (placed at the planned
  waypoint, oriented along the trajectory) overlaps a GT box of another agent at the same
  future time (agents taken from the real future annotations, not a forecast).
* **Lateral / longitudinal error**: error decomposed in the frame of the logged trajectory.
* **Smoothness**: mean |jerk| (m/s³) and mean |curvature change|.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from utils.geometry import box_corners_bev


def _ego_box(xy: np.ndarray, yaw: float, ego_cfg: Any) -> np.ndarray:
    c = float(ego_cfg.rear_axle_to_center)
    return np.array([xy[0] + c * np.cos(yaw), xy[1] + c * np.sin(yaw), 0.0, float(ego_cfg.length),
                     float(ego_cfg.width), 1.6, yaw])


def _polygons_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals ``(4, 2)``."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def trajectory_headings(xy: np.ndarray) -> np.ndarray:
    prev = np.concatenate([np.zeros((1, 2)), xy[:-1]])
    d = xy - prev
    yaw = np.arctan2(d[:, 1], d[:, 0])
    yaw[np.linalg.norm(d, axis=1) < 0.05] = 0.0
    return yaw


def collision_flags(pred: np.ndarray, agent_future: list[dict], collision_labels: Iterable[int], ego_cfg: Any) -> np.ndarray:
    """Per-step collision flags ``(T,)`` against the GT boxes of other agents."""
    labels = set(int(l) for l in collision_labels)
    yaws = trajectory_headings(pred)
    flags = np.zeros(len(pred), dtype=bool)
    for t in range(min(len(pred), len(agent_future))):
        boxes = agent_future[t]["boxes"]
        sel = np.array([int(l) in labels for l in agent_future[t]["labels"]], dtype=bool)
        if not sel.any():
            continue
        ego_poly = box_corners_bev(_ego_box(pred[t], yaws[t], ego_cfg)[None])[0]
        near = np.hypot(boxes[sel, 0] - pred[t, 0], boxes[sel, 1] - pred[t, 1]) < 12.0
        for poly in box_corners_bev(boxes[sel][near].astype(np.float64)):
            if _polygons_intersect(ego_poly, poly):
                flags[t] = True
                break
    return flags


def sample_planning_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, dt: float,
                            agent_future: list[dict] | None = None, collision_labels: Iterable[int] = (),
                            ego_cfg: Any = None) -> dict[str, float]:
    """Metrics of one sample (only valid future steps are used)."""
    mask = mask.astype(bool)
    out: dict[str, float] = {}
    if not mask.any():
        return out
    err = np.linalg.norm(pred - gt, axis=1)
    out["ade"] = float(err[mask].mean())
    last = np.nonzero(mask)[0][-1]
    out["fde"] = float(err[last])
    for sec in (1, 2, 3):
        k = int(round(sec / dt)) - 1
        if k < len(mask) and mask[k]:
            out[f"l2_{sec}s"] = float(err[k])
    # Lateral / longitudinal errors in the logged trajectory's frame.
    gt_yaw = trajectory_headings(gt)
    d = pred - gt
    lon = d[:, 0] * np.cos(gt_yaw) + d[:, 1] * np.sin(gt_yaw)
    lat = -d[:, 0] * np.sin(gt_yaw) + d[:, 1] * np.cos(gt_yaw)
    out["lateral_error"] = float(np.abs(lat[mask]).mean())
    out["longitudinal_error"] = float(np.abs(lon[mask]).mean())
    # Smoothness of the predicted trajectory.
    path = np.concatenate([np.zeros((1, 2)), pred], 0)
    vel = np.diff(path, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    out["jerk"] = float(np.linalg.norm(jerk, axis=1).mean()) if len(jerk) else 0.0
    yaws = np.unwrap(trajectory_headings(pred))
    out["curvature_change"] = float(np.abs(np.diff(yaws, 2)).mean()) if len(yaws) > 2 else 0.0
    if agent_future is not None and ego_cfg is not None:
        flags = collision_flags(pred, agent_future, collision_labels, ego_cfg)
        flags &= mask
        out["collision"] = float(flags.any())
        for sec in (1, 2, 3):
            k = int(round(sec / dt))
            out[f"collision_{sec}s"] = float(flags[:k].any())
    return out


def aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric over the samples that define it (collision -> collision rate)."""
    keys = sorted(set().union(*metrics)) if metrics else []
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics if k in m]
        name = k.replace("collision", "collision_rate") if k.startswith("collision") else k
        out[name] = float(np.mean(vals)) if vals else float("nan")
    out["num_samples"] = len(metrics)
    return out
