"""Trajectory close-up: planned vs logged (ground-truth) ego trajectory with errors."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from visualization.common import GT_TRAJ_COLOR, PLAN_COLOR, BEVCanvas, class_color, put_label


def render_trajectory_closeup(planned: Optional[np.ndarray], gt: Optional[np.ndarray], gt_mask: Optional[np.ndarray],
                              boxes: Optional[np.ndarray] = None, names: Optional[list[str]] = None,
                              past: Optional[np.ndarray] = None, size: int = 420, collision: bool = False,
                              ego_cfg=None) -> np.ndarray:
    """Zoomed BEV (-8..40 m forward, ±24 m lateral) of the planned and logged ego paths."""
    canvas = BEVCanvas((-8, 40), (-24, 24), ppm=size / 48)
    for r in (10, 20, 30):
        cx, cy = canvas.to_px(0, 0)
        cv2.circle(canvas.img, (int(cx), int(cy)), int(r * canvas.ppm), (45, 48, 60), 1, cv2.LINE_AA)
    if boxes is not None and len(boxes):
        canvas.draw_boxes(boxes, [class_color(n) for n in names], thickness=1, fill_alpha=0.25)
    if past is not None and len(past):
        canvas.draw_polyline(past[::-1], (110, 110, 120), thickness=1, markers=False, start_at_ego=False)
    if gt is not None and gt_mask is not None and gt_mask.any():
        canvas.draw_polyline(gt[gt_mask.astype(bool)], GT_TRAJ_COLOR, thickness=2, dashed=True)
    if planned is not None:
        canvas.draw_polyline(planned, (255, 80, 80) if collision else PLAN_COLOR, thickness=2)
        if gt is not None and gt_mask is not None:
            for p, g, ok in zip(planned, gt, gt_mask.astype(bool)):
                if ok:
                    a = canvas._pts(p[None])[0].astype(int)
                    b = canvas._pts(g[None])[0].astype(int)
                    cv2.line(canvas.img, tuple(a), tuple(b), (255, 200, 0), 1, cv2.LINE_AA)
    if ego_cfg is not None:
        canvas.draw_ego(float(ego_cfg.length), float(ego_cfg.width), float(ego_cfg.rear_axle_to_center))
    else:
        canvas.draw_ego()
    put_label(canvas.img, "planned (cyan)  vs  logged (white dashed)", (6, 18), 0.45)
    if collision:
        put_label(canvas.img, "COLLISION PREDICTED", (6, 40), 0.55, (255, 255, 255), (200, 40, 40))
    return canvas.img
