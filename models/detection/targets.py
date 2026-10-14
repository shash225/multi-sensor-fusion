"""CenterPoint training targets (NumPy, generated inside the dataset workers).

For each GT box the class heatmap receives a 2D Gaussian at the box centre; regression
targets are stored at the integer centre cell:
``[dx, dy, z, log l, log w, log h, sin yaw, cos yaw, vx, vy]``.
"""
from __future__ import annotations

import numpy as np

from utils.bev_grid import BEVGrid


def gaussian_radius(height: float, width: float, min_overlap: float = 0.1) -> float:
    """CornerNet/CenterPoint Gaussian radius such that a shifted box keeps ``min_overlap`` IoU."""
    a1, b1 = 1, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return float(min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Draw (max-merge) a Gaussian of ``radius`` cells at ``center = (ix, iy)`` into ``heatmap[iy, ix]``."""
    diameter = 2 * radius + 1
    sigma = diameter / 6
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    gaussian = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    x, y = center
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    region = heatmap[y - top:y + bottom, x - left:x + right]
    g = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(region, g, out=region)


def build_center_targets(boxes: np.ndarray, labels: np.ndarray, velocity_valid: np.ndarray, grid: BEVGrid,
                         num_classes: int, max_objects: int = 500, min_radius: int = 2,
                         gaussian_overlap: float = 0.1) -> dict[str, np.ndarray]:
    """Targets on the detection-head grid.

    Args:
        boxes: ``(M, 9)`` reference-frame boxes ``[x, y, z, l, w, h, yaw, vx, vy]``.
        labels: ``(M,)`` class indices.
        velocity_valid: ``(M,)`` whether the GT velocity is defined (single-frame tracks are not).
        grid: the head grid (BEV grid downsampled by the head stride).
    """
    ny, nx = grid.shape
    heatmap = np.zeros((num_classes, ny, nx), dtype=np.float32)
    ind = np.zeros(max_objects, dtype=np.int64)
    mask = np.zeros(max_objects, dtype=np.float32)
    vel_mask = np.zeros(max_objects, dtype=np.float32)
    reg = np.zeros((max_objects, 10), dtype=np.float32)
    k = 0
    for box, label, vel_ok in zip(boxes, labels, velocity_valid):
        if k >= max_objects:
            break
        cx, cy = grid.world_to_grid(box[0], box[1])
        ix, iy = int(np.floor(cx)), int(np.floor(cy))
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        length, width = box[3] / grid.resolution, box[4] / grid.resolution
        if length <= 0 or width <= 0:
            continue
        radius = max(min_radius, int(gaussian_radius(length, width, gaussian_overlap)))
        draw_gaussian(heatmap[label], (ix, iy), radius)
        ind[k] = iy * nx + ix
        mask[k] = 1.0
        vel_mask[k] = float(vel_ok)
        reg[k] = [cx - ix, cy - iy, box[2], np.log(max(box[3], 1e-3)), np.log(max(box[4], 1e-3)),
                  np.log(max(box[5], 1e-3)), np.sin(box[6]), np.cos(box[6]), box[7], box[8]]
        k += 1
    return {"heatmap": heatmap, "ind": ind, "mask": mask, "vel_mask": vel_mask, "reg_target": reg}


def foreground_heatmap(boxes: np.ndarray, grid: BEVGrid, min_radius: int = 2, gaussian_overlap: float = 0.1) -> np.ndarray:
    """Class-agnostic centre heatmap ``(1, ny, nx)`` at BEV resolution (feature-selection guidance)."""
    ny, nx = grid.shape
    heatmap = np.zeros((ny, nx), dtype=np.float32)
    for box in boxes:
        cx, cy = grid.world_to_grid(box[0], box[1])
        ix, iy = int(np.floor(cx)), int(np.floor(cy))
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        radius = max(min_radius, int(gaussian_radius(box[3] / grid.resolution, box[4] / grid.resolution, gaussian_overlap)))
        draw_gaussian(heatmap, (ix, iy), radius)
    return heatmap[None]
