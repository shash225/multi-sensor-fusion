"""The main "autonomous driving" BEV scene.

Layers (bottom -> top): map prior, LiDAR points, radar returns + Doppler arrows,
ground-truth boxes (dashed) and their logged future paths, predicted boxes (filled,
class colour) with velocity arrows / track IDs / motion trails, logged ego trajectory
(white dashed), planned trajectory (cyan, red if a collision is predicted), ego vehicle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from visualization.common import (EGO_COLOR, GT_BOX_COLOR, GT_TRAJ_COLOR, PLAN_COLOR, RADAR_COLOR, BEVCanvas,
                                  class_color, colorize, put_label)

MAP_COLOR = np.array([38, 44, 60], np.uint8)


@dataclass
class SceneLayers:
    lidar_points: Optional[np.ndarray] = None
    radar_points: Optional[np.ndarray] = None
    map_patch: Optional[np.ndarray] = None                    # display layout, same extent/ppm as the canvas
    det_boxes: Optional[np.ndarray] = None
    det_names: Sequence[str] = field(default_factory=list)
    det_scores: Optional[np.ndarray] = None
    track_ids: Optional[np.ndarray] = None
    track_histories: Optional[list] = None
    gt_boxes: Optional[np.ndarray] = None
    gt_names: Sequence[str] = field(default_factory=list)
    gt_future: Optional[np.ndarray] = None                     # (M, T, 2)
    gt_future_mask: Optional[np.ndarray] = None
    plan: Optional[np.ndarray] = None
    plan_collision: bool = False
    ego_future: Optional[np.ndarray] = None
    ego_future_mask: Optional[np.ndarray] = None
    ego_past: Optional[np.ndarray] = None
    astar_path: Optional[np.ndarray] = None


def _legend(img: np.ndarray, items: list[tuple[str, tuple[int, int, int], str]]) -> None:
    x0, y0 = img.shape[1] - 196, img.shape[0] - 16 - 18 * len(items)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 18), (img.shape[1] - 6, img.shape[0] - 6), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    for i, (label, color, kind) in enumerate(items):
        y = y0 + i * 18
        if kind == "dot":
            cv2.circle(img, (x0 + 8, y - 4), 4, color, -1, cv2.LINE_AA)
        elif kind == "box":
            cv2.rectangle(img, (x0 + 1, y - 10), (x0 + 15, y + 1), color, 2)
        elif kind == "dash":
            for k in range(0, 18, 6):
                cv2.line(img, (x0 + k, y - 4), (x0 + k + 3, y - 4), color, 2)
        else:
            cv2.line(img, (x0, y - 4), (x0 + 18, y - 4), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 235), 1, cv2.LINE_AA)


def render_driving_scene(layers: SceneLayers, x_range: Sequence[float] = (-25, 55), y_range: Sequence[float] = (-40, 40),
                         ppm: float = 8.0, show_lidar: bool = True, show_radar: bool = True, show_gt: bool = True,
                         show_pred: bool = True, show_tracks: bool = True, title: str = "", ego_cfg=None) -> np.ndarray:
    canvas = BEVCanvas(x_range, y_range, ppm)
    if layers.map_patch is not None and layers.map_patch.shape[:2] == canvas.img.shape[:2]:
        canvas.img[layers.map_patch > 0] = MAP_COLOR
    canvas.draw_range_rings((10, 20, 30, 40, 50), (52, 58, 74))
    legend = []
    if show_lidar and layers.lidar_points is not None and len(layers.lidar_points):
        pts = layers.lidar_points
        if len(pts) > 150000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 150000, replace=False)]
        canvas.draw_points(pts[:, :2], (colorize(np.clip(pts[:, 2], -2, 3), "cividis", -2.0, 3.0) * 0.75).astype(np.uint8))
        legend.append(("LiDAR", (150, 150, 120), "dot"))
    if show_radar and layers.radar_points is not None and len(layers.radar_points):
        rp = layers.radar_points
        canvas.draw_points(rp[:, :2], RADAR_COLOR, radius=2)
        moving = np.hypot(rp[:, 4], rp[:, 5]) > 0.5
        canvas.draw_arrows(rp[moving, :2], rp[moving, 4:6], RADAR_COLOR, scale=1.0)
        legend.append(("Radar (+velocity)", RADAR_COLOR, "dot"))
    if show_gt and layers.gt_boxes is not None and len(layers.gt_boxes):
        canvas.draw_boxes(layers.gt_boxes, GT_BOX_COLOR, thickness=1, dashed=True)
        if layers.gt_future is not None:
            for fut, m in zip(layers.gt_future, layers.gt_future_mask):
                if m.any():
                    canvas.draw_polyline(fut[m], (90, 200, 120), thickness=1, markers=False, start_at_ego=False)
        legend.append(("Ground-truth box", GT_BOX_COLOR, "dash"))
    if show_pred and layers.det_boxes is not None and len(layers.det_boxes):
        colors = [class_color(n) for n in layers.det_names]
        labels = None
        if show_tracks and layers.track_ids is not None:
            labels = [f"#{int(t)}" if int(t) > 0 else "" for t in layers.track_ids]
        if show_tracks and layers.track_histories is not None:
            for hist, col in zip(layers.track_histories, colors):
                if len(hist) > 1:
                    canvas.draw_polyline(hist, tuple(int(c * 0.6) for c in col), thickness=1, markers=False, start_at_ego=False)
        canvas.draw_boxes(layers.det_boxes, colors, thickness=2, fill_alpha=0.35, labels=labels)
        canvas.draw_arrows(layers.det_boxes[:, :2], layers.det_boxes[:, 7:9], (255, 255, 255), scale=1.0, thickness=2)
        legend.append(("Detected object", (255, 158, 0), "box"))
    if layers.ego_past is not None and len(layers.ego_past):
        canvas.draw_polyline(layers.ego_past[::-1], (120, 120, 130), thickness=1, markers=False)
    if layers.ego_future is not None and layers.ego_future_mask is not None and layers.ego_future_mask.any():
        canvas.draw_polyline(layers.ego_future[layers.ego_future_mask.astype(bool)], GT_TRAJ_COLOR, thickness=2, dashed=True)
        legend.append(("Logged future", GT_TRAJ_COLOR, "dash"))
    if layers.plan is not None:
        col = (255, 70, 70) if layers.plan_collision else PLAN_COLOR
        canvas.draw_polyline(layers.plan, col, thickness=3)
        legend.append(("Planned trajectory", col, "line"))
    if ego_cfg is not None:
        canvas.draw_ego(float(ego_cfg.length), float(ego_cfg.width), float(ego_cfg.rear_axle_to_center))
    else:
        canvas.draw_ego()
    legend.append(("Ego vehicle", EGO_COLOR, "box"))
    _legend(canvas.img, legend)
    put_label(canvas.img, "FRONT", (canvas.w // 2 - 22, 18), 0.5, (200, 200, 210), None)
    if title:
        put_label(canvas.img, title, (8, canvas.h - 12), 0.5)
    return canvas.img
