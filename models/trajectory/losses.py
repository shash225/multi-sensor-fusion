"""Trajectory-planning losses.

``L_trajectory = w_reg · L1(pred, logged) + w_col · collision + w_smooth · smoothness + w_goal · goal``

* **Collision penalty**: the GT future occupancy of other agents ``(B, T, H, W)`` is
  softened (3×3 average) and sampled with ``grid_sample`` at 5 points of the ego
  footprint (rear, centre, front, left, right) placed at each predicted waypoint with the
  heading of the trajectory — differentiable w.r.t. the waypoints.
* **Smoothness**: squared second differences including the current pose (and the
  current velocity), penalising jerky trajectories.
* **Goal / progress**: Euclidean error of the final valid waypoint.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from utils.bev_grid import BEVGrid


def ego_footprint_points(waypoints: torch.Tensor, ego_cfg: Any) -> torch.Tensor:
    """Sample points on the ego body ``(B, T, 5, 2)`` for waypoints ``(B, T, 2)`` (rear-axle positions)."""
    prev = torch.cat([torch.zeros_like(waypoints[:, :1]), waypoints[:, :-1]], 1)
    delta = waypoints - prev
    heading = torch.atan2(delta[..., 1], delta[..., 0] + 1e-6)
    still = delta.norm(dim=-1) < 0.05
    heading = torch.where(still, torch.zeros_like(heading), heading)
    fwd = torch.stack([heading.cos(), heading.sin()], -1)
    left = torch.stack([-heading.sin(), heading.cos()], -1)
    l, w, c = float(ego_cfg.length), float(ego_cfg.width), float(ego_cfg.rear_axle_to_center)
    centre = waypoints + c * fwd
    offsets = [(-l / 2 + 0.3, 0.0), (0.0, 0.0), (l / 2 - 0.3, 0.0), (0.0, w / 2 - 0.2), (0.0, -w / 2 + 0.2)]
    return torch.stack([centre + a * fwd + b * left for a, b in offsets], 2)


def sample_occupancy(occ: torch.Tensor, points: torch.Tensor, grid: BEVGrid) -> torch.Tensor:
    """Bilinear sample ``occ (B, T, H, W)`` at metric ``points (B, T, P, 2)`` -> ``(B, T, P)``."""
    b, t, h, w = occ.shape
    gx = (points[..., 0] - grid.x_range[0]) / (grid.x_range[1] - grid.x_range[0]) * 2 - 1  # x -> width axis
    gy = (points[..., 1] - grid.y_range[0]) / (grid.y_range[1] - grid.y_range[0]) * 2 - 1  # y -> height axis
    coords = torch.stack([gx, gy], -1).reshape(b * t, 1, -1, 2)
    sampled = F.grid_sample(occ.reshape(b * t, 1, h, w), coords, align_corners=False, padding_mode="zeros")
    return sampled.view(b, t, -1)


def trajectory_loss(pred: torch.Tensor, batch: dict[str, Any], grid: BEVGrid, ego_cfg: Any, weights: Any,
                    dt: float) -> dict[str, torch.Tensor]:
    gt = batch["ego_future"].float()
    mask = batch["ego_future_mask"].float()
    pred = pred.float()
    valid = mask.sum().clamp(min=1.0)
    reg = ((pred - gt).abs().sum(-1) * mask).sum() / valid

    occ = F.avg_pool2d(batch["future_occ"].float(), 3, stride=1, padding=1)
    occ_at = sample_occupancy(occ, ego_footprint_points(pred, ego_cfg), grid).amax(-1)  # (B, T)
    collision = (occ_at * mask).sum() / valid

    speed = batch["ego_state"][:, 0].float()
    start = torch.stack([-speed * dt, torch.zeros_like(speed)], -1).unsqueeze(1)
    path = torch.cat([start, torch.zeros_like(start), pred], 1)
    smooth = (path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2]).pow(2).sum(-1).mean()

    last = (mask.cumsum(1) == mask.sum(1, keepdim=True)) & (mask > 0)
    has_last = last.any(1)
    if has_last.any():
        goal = ((pred - gt).norm(dim=-1) * last.float()).sum() / has_last.float().sum()
    else:
        goal = pred.sum() * 0.0
    total = (float(weights.regression) * reg + float(weights.collision) * collision
             + float(weights.smoothness) * smooth + float(weights.goal) * goal)
    return {"traj_regression": reg, "traj_collision": collision, "traj_smoothness": smooth, "traj_goal": goal,
            "trajectory": total}
