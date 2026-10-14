"""Rendering primitives shared by every view (OpenCV, returns RGB ``uint8`` arrays).

Display convention for every BEV image: **forward (+x) points up, left (+y) points left**,
ego vehicle at the centre unless the extent is asymmetric.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from utils.bev_grid import BEVGrid
from utils.geometry import box_corners_bev

# Bright palette that stays readable on the dark BEV background.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 215, 0),
    "trailer": (210, 105, 30),
    "barrier": (170, 170, 190),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 160),
    "pedestrian": (80, 170, 255),
    "traffic_cone": (0, 210, 180),
}
EGO_COLOR = (0, 230, 118)
PLAN_COLOR = (0, 229, 255)
GT_TRAJ_COLOR = (255, 255, 255)
GT_BOX_COLOR = (120, 255, 120)
LIDAR_COLOR = (150, 150, 165)
RADAR_COLOR = (255, 80, 200)
BG_COLOR = (16, 18, 26)


@lru_cache(maxsize=16)
def _lut(cmap: str) -> np.ndarray:
    import matplotlib

    return (matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def colorize(values: np.ndarray, cmap: str = "inferno", vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Map an array of scalars to RGB with a matplotlib colormap."""
    v = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(v)) if vmin is None else vmin
    hi = float(np.nanmax(v)) if vmax is None else vmax
    norm = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return _lut(cmap)[(np.nan_to_num(norm) * 255).astype(np.uint8)]


def bev_to_display(arr: np.ndarray) -> np.ndarray:
    """Convert a model-layout BEV array ``[iy, ix, ...]`` to display layout (forward up, left left)."""
    flipped = arr[::-1, ::-1]
    return np.ascontiguousarray(np.swapaxes(flipped, 0, 1))


def heatmap_image(arr: np.ndarray, cmap: str = "inferno", size: Optional[int] = None, vmin: Optional[float] = None,
                  vmax: Optional[float] = None, percentile: Optional[float] = 99.0) -> np.ndarray:
    """Model-layout ``(ny, nx)`` map -> display-oriented RGB heatmap."""
    a = np.asarray(arr, dtype=np.float32)
    if vmax is None and percentile is not None and np.isfinite(a).any():
        vmax = float(np.percentile(a, percentile))
    rgb = colorize(bev_to_display(a), cmap, vmin, vmax)
    if size is not None:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    return rgb


def put_label(img: np.ndarray, text: str, org: tuple[int, int] = (8, 22), scale: float = 0.55,
              color: tuple[int, int, int] = (255, 255, 255), bg: Optional[tuple[int, int, int]] = (0, 0, 0)) -> np.ndarray:
    """Draw text with an optional background box (in place) and return the image."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 1), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


class BEVCanvas:
    """Top-down drawing surface in metric ego coordinates.

    Args:
        x_range, y_range: visible extent in metres (ego frame).
        ppm: pixels per metre.
    """

    def __init__(self, x_range: Sequence[float] = (-50, 50), y_range: Sequence[float] = (-50, 50), ppm: float = 6.0,
                 bg: tuple[int, int, int] = BG_COLOR) -> None:
        self.x_range, self.y_range, self.ppm = tuple(x_range), tuple(y_range), float(ppm)
        self.h = int(round((self.x_range[1] - self.x_range[0]) * ppm))
        self.w = int(round((self.y_range[1] - self.y_range[0]) * ppm))
        self.img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        self.img[:] = bg

    # ---------------------------------------------------------------- coords
    def to_px(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Metric ego coordinates -> (column, row) pixel coordinates (float)."""
        return (self.y_range[1] - np.asarray(y)) * self.ppm, (self.x_range[1] - np.asarray(x)) * self.ppm

    def _pts(self, xy: np.ndarray) -> np.ndarray:
        c, r = self.to_px(xy[..., 0], xy[..., 1])
        return np.stack([c, r], -1)

    # ---------------------------------------------------------------- layers
    def draw_grid_overlay(self, arr: np.ndarray, grid: BEVGrid, cmap: str = "inferno", alpha: float = 0.6,
                          vmin: Optional[float] = None, vmax: Optional[float] = None, threshold: Optional[float] = None) -> None:
        """Blend a model-layout ``(ny, nx)`` map defined on ``grid`` under the current drawing."""
        a = np.asarray(arr, dtype=np.float32)
        rgb = colorize(bev_to_display(a), cmap, vmin, vmax)
        disp_mask = bev_to_display(a > (threshold if threshold is not None else -np.inf))
        # Position of the grid in canvas pixels.
        c0, r0 = self.to_px(grid.x_range[1], grid.y_range[1])
        c1, r1 = self.to_px(grid.x_range[0], grid.y_range[0])
        c0, r0, c1, r1 = int(round(c0)), int(round(r0)), int(round(c1)), int(round(r1))
        if c1 <= c0 or r1 <= r0:
            return
        rgb = cv2.resize(rgb, (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(disp_mask.astype(np.uint8), (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST).astype(bool)
        # Clip to canvas.
        cc0, rr0, cc1, rr1 = max(c0, 0), max(r0, 0), min(c1, self.w), min(r1, self.h)
        if cc1 <= cc0 or rr1 <= rr0:
            return
        sub_rgb = rgb[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        sub_mask = mask[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        region = self.img[rr0:rr1, cc0:cc1]
        blended = (region * (1 - alpha) + sub_rgb * alpha).astype(np.uint8)
        region[sub_mask] = blended[sub_mask]

    def draw_range_rings(self, radii: Iterable[float] = (10, 20, 30, 40, 50), color=(55, 60, 75)) -> None:
        cx, cy = self.to_px(0.0, 0.0)
        for r in radii:
            cv2.circle(self.img, (int(cx), int(cy)), int(r * self.ppm), color, 1, cv2.LINE_AA)
            put_label(self.img, f"{int(r)} m", (int(cx) + 4, int(cy - r * self.ppm) + 12), 0.35, color, None)

    def draw_points(self, xy: np.ndarray, color: tuple[int, int, int] | np.ndarray = LIDAR_COLOR, radius: int = 1) -> None:
        if xy.shape[0] == 0:
            return
        c, r = self.to_px(xy[:, 0], xy[:, 1])
        ci, ri = c.astype(np.int32), r.astype(np.int32)
        ok = (ci >= 0) & (ci < self.w) & (ri >= 0) & (ri < self.h)
        colors = np.asarray(color, dtype=np.uint8)
        if radius <= 1:
            if colors.ndim == 2:
                self.img[ri[ok], ci[ok]] = colors[ok]
            else:
                self.img[ri[ok], ci[ok]] = colors
            return
        for k in np.nonzero(ok)[0]:
            col = tuple(int(v) for v in (colors[k] if colors.ndim == 2 else colors))
            cv2.circle(self.img, (int(ci[k]), int(ri[k])), radius, col, -1, cv2.LINE_AA)

    def draw_boxes(self, boxes: np.ndarray, colors: Sequence[tuple[int, int, int]] | tuple[int, int, int],
                   thickness: int = 2, fill_alpha: float = 0.0, labels: Optional[Sequence[str]] = None,
                   dashed: bool = False) -> None:
        """Oriented boxes ``[x, y, z, l, w, h, yaw, ...]``; a line marks the heading (front)."""
        if boxes.shape[0] == 0:
            return
        corners = self._pts(box_corners_bev(boxes))  # (N, 4, 2) pixels
        per_box = isinstance(colors, (list, np.ndarray)) and len(colors) == len(boxes) and not isinstance(colors[0], (int, np.integer))
        overlay = self.img.copy() if fill_alpha > 0 else None
        for i, poly in enumerate(corners):
            color = tuple(int(v) for v in (colors[i] if per_box else colors))
            pts = np.round(poly).astype(np.int32)
            if overlay is not None:
                cv2.fillPoly(overlay, [pts], color)
            if dashed:
                for a, b in zip(pts, np.roll(pts, -1, 0)):
                    self._dashed_line(a, b, color, thickness)
            else:
                cv2.polylines(self.img, [pts], True, color, thickness, cv2.LINE_AA)
            front = ((pts[0] + pts[1]) / 2).astype(np.int32)
            centre = pts.mean(0).astype(np.int32)
            cv2.line(self.img, tuple(centre), tuple(front), color, thickness, cv2.LINE_AA)
        if overlay is not None:
            cv2.addWeighted(overlay, fill_alpha, self.img, 1 - fill_alpha, 0, self.img)
        if labels is not None:
            for i, text in enumerate(labels):
                if text:
                    p = np.round(corners[i].min(0)).astype(int)
                    put_label(self.img, text, (int(p[0]), int(p[1]) - 2), 0.35, (255, 255, 255), (20, 20, 20))

    def _dashed_line(self, a: np.ndarray, b: np.ndarray, color, thickness: int, dash: int = 6) -> None:
        length = float(np.hypot(*(b - a)))
        n = max(int(length / dash), 1)
        for k in range(0, n, 2):
            p = a + (b - a) * k / n
            q = a + (b - a) * min(k + 1, n) / n
            cv2.line(self.img, tuple(p.astype(int)), tuple(q.astype(int)), color, thickness, cv2.LINE_AA)

    def draw_arrows(self, starts: np.ndarray, vectors: np.ndarray, color=(255, 255, 255), scale: float = 1.0,
                    thickness: int = 1, min_norm: float = 0.2) -> None:
        """Velocity arrows: ``starts`` (N, 2) metres, ``vectors`` (N, 2) m/s drawn as ``scale`` seconds of motion."""
        for s, v in zip(starts, vectors):
            if not np.all(np.isfinite(v)) or np.hypot(*v) < min_norm:
                continue
            a = self._pts(np.asarray(s)[None])[0]
            b = self._pts((np.asarray(s) + np.asarray(v) * scale)[None])[0]
            cv2.arrowedLine(self.img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA, tipLength=0.25)

    def draw_polyline(self, xy: np.ndarray, color=PLAN_COLOR, thickness: int = 2, dashed: bool = False,
                      markers: bool = True, start_at_ego: bool = True) -> None:
        if xy.shape[0] == 0:
            return
        pts = xy if not start_at_ego else np.concatenate([np.zeros((1, 2)), xy], 0)
        px = np.round(self._pts(pts)).astype(np.int32)
        if dashed:
            for a, b in zip(px[:-1], px[1:]):
                self._dashed_line(a, b, color, thickness, dash=8)
        else:
            cv2.polylines(self.img, [px], False, color, thickness, cv2.LINE_AA)
        if markers:
            for p in px[1 if start_at_ego else 0:]:
                cv2.circle(self.img, tuple(p), max(thickness + 1, 3), color, -1, cv2.LINE_AA)

    def draw_ego(self, length: float = 4.084, width: float = 1.85, rear_axle_to_center: float = 1.4,
                 color=EGO_COLOR) -> None:
        box = np.array([[rear_axle_to_center, 0.0, 0.0, length, width, 1.5, 0.0]])
        self.draw_boxes(box, color, thickness=2, fill_alpha=0.45)
        tip = self._pts(np.array([[rear_axle_to_center + length / 2 + 1.5, 0.0]]))[0].astype(int)
        base = self._pts(np.array([[rear_axle_to_center + length / 2, 0.0]]))[0].astype(int)
        cv2.arrowedLine(self.img, tuple(base), tuple(tip), color, 2, cv2.LINE_AA, tipLength=0.5)

    def crop_to(self, x_range: Sequence[float], y_range: Sequence[float]) -> np.ndarray:
        c0, r0 = self.to_px(x_range[1], y_range[1])
        c1, r1 = self.to_px(x_range[0], y_range[0])
        return self.img[max(int(r0), 0):int(r1), max(int(c0), 0):int(c1)].copy()


def class_color(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, (200, 200, 200))


def stack_images(images: list[np.ndarray], cols: int, pad: int = 4, bg=(10, 10, 14)) -> np.ndarray:
    """Tile same-height images in a grid (resizing to the first image's size)."""
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.empty((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), np.uint8)
    out[:] = bg
    for i, im in enumerate(images):
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        r, c = divmod(i, cols)
        out[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = im
    return out
