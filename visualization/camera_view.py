"""Camera view: original images with projected 3D boxes, labels and LiDAR points.

Boxes live in the reference frame; they are mapped into each camera with
``inv(cam2ref)`` (which includes the camera's own ego pose, i.e. time-synchronised)
and projected with the camera intrinsics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from utils.geometry import box_corners_3d, invert_transform
from visualization.common import class_color, colorize, put_label, stack_images

# Order of the 2x3 surround mosaic (front row, back row).
MOSAIC_ORDER = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def load_image(root: Path, cam_rec: dict[str, Any]) -> Optional[np.ndarray]:
    try:
        with Image.open(Path(root) / cam_rec["path"]) as im:
            return np.asarray(im.convert("RGB")).copy()
    except OSError:
        return None


def project_points(points_ref: np.ndarray, cam_rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reference-frame points ``(N, 3)`` -> pixel coordinates ``(N, 2)`` and depth ``(N,)``."""
    ref2cam = invert_transform(np.asarray(cam_rec["sensor2ref"], dtype=np.float64))
    pc = points_ref[:, :3].astype(np.float64) @ ref2cam[:3, :3].T + ref2cam[:3, 3]
    depth = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc @ np.asarray(cam_rec["intrinsic"]).T)[:, :2] / depth[:, None]
    return uv, depth


def draw_lidar_on_image(img: np.ndarray, points_ref: np.ndarray, cam_rec: dict[str, Any], max_depth: float = 60.0,
                        radius: int = 2) -> np.ndarray:
    out = img.copy()
    uv, d = project_points(points_ref, cam_rec)
    h, w = img.shape[:2]
    ok = (d > 1.0) & (d < max_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    colors = colorize(d[ok], "turbo", 1.0, 40.0)
    for (u, v), c in zip(uv[ok].astype(int), colors):
        cv2.circle(out, (int(u), int(v)), radius, tuple(int(x) for x in c), -1)
    return out


def draw_boxes_on_image(img: np.ndarray, boxes: np.ndarray, cam_rec: dict[str, Any], names: Sequence[str],
                        scores: Optional[Sequence[float]] = None, color: Optional[tuple[int, int, int]] = None,
                        thickness: int = 2, show_labels: bool = True) -> np.ndarray:
    """Draw 3D boxes (all 8 corners in front of the camera) with class / confidence labels."""
    out = img.copy()
    if boxes.shape[0] == 0:
        return out
    h, w = img.shape[:2]
    corners = box_corners_3d(boxes.astype(np.float64))  # (N, 8, 3)
    for i in range(len(boxes)):
        uv, depth = project_points(corners[i], cam_rec)
        if np.any(depth < 0.5):
            continue
        if uv[:, 0].max() < 0 or uv[:, 0].min() >= w or uv[:, 1].max() < 0 or uv[:, 1].min() >= h:
            continue
        col = color or class_color(names[i])
        pts = np.round(uv).astype(np.int32)
        for a, b in _EDGES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), col, thickness, cv2.LINE_AA)
        # Front face cross for orientation.
        cv2.line(out, tuple(pts[0]), tuple(pts[5]), col, 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[1]), tuple(pts[4]), col, 1, cv2.LINE_AA)
        if show_labels:
            label = names[i].replace("_", " ").upper()
            if scores is not None:
                label += f" {scores[i]:.2f}"
            x0, y0 = int(np.clip(pts[:, 0].min(), 0, w - 1)), int(np.clip(pts[:, 1].min(), 14, h - 1))
            put_label(out, label, (x0, y0 - 4), 0.8, (0, 0, 0), col)
    return out


def camera_ipm_bev(images: dict[str, np.ndarray], cam_recs: dict[str, dict[str, Any]], x_range: Sequence[float] = (-40, 40),
                   y_range: Sequence[float] = (-40, 40), ppm: float = 4.0, ground_z: float = 0.0) -> np.ndarray:
    """Geometric camera BEV (inverse perspective mapping) using only the calibration.

    Every ground-plane cell (``z = ground_z`` in the ego frame) is projected into each camera
    with ``K · inv(cam2ref)`` and coloured by the pixel it lands on. This is *not* learned and
    assumes a flat ground — objects above the ground are smeared — which is exactly the
    limitation the learned Lift-Splat depth distribution removes. Display layout (forward up).
    """
    h = int(round((x_range[1] - x_range[0]) * ppm))
    w = int(round((y_range[1] - y_range[0]) * ppm))
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    x = x_range[1] - (rows + 0.5) / ppm
    y = y_range[1] - (cols + 0.5) / ppm
    pts = np.stack([x.ravel(), y.ravel(), np.full(x.size, ground_z)], -1)
    out = np.zeros((h * w, 3), np.uint8)
    best_depth = np.full(h * w, np.inf)
    for cam, img in images.items():
        rec = cam_recs.get(cam)
        if img is None or rec is None:
            continue
        uv, d = project_points(pts, rec)
        ih, iw = img.shape[:2]
        ok = (d > 1.0) & (d < best_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < iw - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < ih - 1)
        if not ok.any():
            continue
        # cv2.remap limits map sides to < 32767, so lay the lookup out as a (rows, 1024) block.
        n, width = int(ok.sum()), 1024
        rows_n = (n + width - 1) // width
        map_x = np.zeros(rows_n * width, np.float32)
        map_y = np.zeros(rows_n * width, np.float32)
        map_x[:n], map_y[:n] = uv[ok, 0], uv[ok, 1]
        sampled = cv2.remap(img, map_x.reshape(rows_n, width), map_y.reshape(rows_n, width), cv2.INTER_LINEAR)
        out[ok] = sampled.reshape(-1, 3)[:n]
        best_depth[ok] = d[ok]  # closest camera wins (least perspective stretch)
    return out.reshape(h, w, 3)


def camera_mosaic(images: dict[str, np.ndarray], width: int = 1200, order: Sequence[str] = MOSAIC_ORDER,
                  label: bool = True) -> np.ndarray:
    """2x3 surround-view mosaic; each tile is labelled with its camera name."""
    tile_w = width // 3
    tiles = []
    for cam in order:
        img = images.get(cam)
        if img is None:
            img = np.zeros((900, 1600, 3), np.uint8)
            put_label(img, "MISSING", (700, 450), 2.0, (255, 80, 80), None)
        tile = cv2.resize(img, (tile_w, int(tile_w * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        if label:
            put_label(tile, cam.replace("CAM_", ""), (6, 18), 0.5)
        tiles.append(tile)
    return stack_images(tiles, cols=3, pad=3)
