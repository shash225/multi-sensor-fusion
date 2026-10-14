"""Explainability of the feature-selection network.

Everything here is computed from the *actual* model outputs of the current frame:

* raw vs selected feature energy per modality (what was suppressed),
* spatial importance maps ``s_lidar`` / ``s_radar`` (where each sensor is trusted),
* a sensor-dominance map (which sensor the network relies on at every cell),
* channel gates, with the physically meaningful raster channels named
  (LiDAR density / height / intensity, radar density / vx / vy / RCS / time lag),
* per-object importance: mean gate of each sensor inside every detected / GT box,
  aggregated by class ("which sensor does the network trust for pedestrians?").
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from utils.bev_grid import BEVGrid
from utils.raster import rasterize_boxes
from visualization.common import bev_to_display, colorize, heatmap_image, put_label

LIDAR_RGB = np.array([42, 120, 214], np.float32)   # blue  (categorical slot 1)
RADAR_RGB = np.array([235, 104, 52], np.float32)   # orange (categorical slot 2)
NEUTRAL_RGB = np.array([56, 56, 53], np.float32)   # diverging midpoint (dark mode gray)


def energy(feat: np.ndarray) -> np.ndarray:
    return np.sqrt((feat.astype(np.float32) ** 2).sum(0))


def titled(img: np.ndarray, title: str, size: int = 300) -> np.ndarray:
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return put_label(out, title, (6, 18), 0.48)


def importance_map(gate: np.ndarray, title: str, size: int = 300, cmap: str = "magma") -> np.ndarray:
    """Spatial gate ``(H, W)`` in [0, 1] (fixed scale so frames are comparable)."""
    return titled(heatmap_image(gate, cmap, vmin=0.0, vmax=1.0, percentile=None), title, size)


def energy_map(feat: np.ndarray, title: str, size: int = 300, vmax: Optional[float] = None) -> np.ndarray:
    """Heatmap of a ``(C, H, W)`` feature map's channel L2 norm, or of a precomputed ``(H, W)`` energy."""
    e = energy(feat) if feat.ndim == 3 else feat.astype(np.float32)
    return titled(heatmap_image(e, "inferno", vmin=0.0, vmax=vmax), title, size)


def contribution(selected_energy: np.ndarray, raw_energy: np.ndarray) -> np.ndarray:
    """Selected feature energy normalised by the sensor's own raw-energy scale (99th percentile).

    Gates in cells where a sensor has no features change nothing, so "which sensor does the
    network rely on" must be read from what actually passes the selection, not from the gates alone.
    """
    scale = float(np.percentile(raw_energy, 99)) or 1.0
    return np.clip(selected_energy / scale, 0.0, 1.0)


def dominance_map(s_lidar: np.ndarray, s_radar: np.ndarray, size: int = 300, min_gate: float = 0.12,
                  title: str = "Sensor contribution after selection (blue LiDAR / orange radar)") -> np.ndarray:
    """Diverging map: blue = LiDAR dominates, orange = radar dominates, gray = equal; dark = neither."""
    total = s_lidar + s_radar
    share = np.where(total > 1e-6, s_lidar / np.maximum(total, 1e-6), 0.5)  # 1 = LiDAR only
    t = (share - 0.5) * 2  # [-1, 1]
    rgb = np.where(t[..., None] >= 0, NEUTRAL_RGB + (LIDAR_RGB - NEUTRAL_RGB) * t[..., None],
                   NEUTRAL_RGB + (RADAR_RGB - NEUTRAL_RGB) * (-t[..., None]))
    strength = np.clip(np.maximum(s_lidar, s_radar), 0, 1)
    rgb = rgb * np.where(strength[..., None] > min_gate, 1.0, 0.15)
    img = bev_to_display(rgb.astype(np.uint8))
    return titled(img, title, size)


def modality_weight_map(weights: np.ndarray, order: list[str], size: int = 300) -> np.ndarray:
    """Fusion attention ``(M, H, W)`` (softmax over modalities) as an additive RGB composite."""
    palette = {"lidar": LIDAR_RGB, "radar": RADAR_RGB, "camera": np.array([27, 175, 122], np.float32)}
    rgb = np.zeros(weights.shape[1:] + (3,), np.float32)
    for i, m in enumerate(order):
        rgb += weights[i][..., None] * palette.get(m, np.array([200, 200, 200], np.float32))
    img = bev_to_display(np.clip(rgb, 0, 255).astype(np.uint8))
    legend = " / ".join(f"{m}" for m in order)
    return titled(img, f"Fusion attention ({legend})", size)


def object_importance(boxes: np.ndarray, names: list[str], gates: dict[str, np.ndarray], grid: BEVGrid) -> list[dict[str, Any]]:
    """Mean spatial gate of each modality inside every box (``boxes`` in the ego frame)."""
    rows = []
    for box, name in zip(boxes, names):
        mask = rasterize_boxes(box[None].astype(np.float64), grid, inflate=0.25).astype(bool)
        if not mask.any():
            continue
        row = {"class": name, "x": float(box[0]), "y": float(box[1]), "range_m": float(np.hypot(box[0], box[1]))}
        for m, g in gates.items():
            row[f"{m}_importance"] = float(g[mask].mean())
        rows.append(row)
    return rows


def class_importance_summary(rows: list[dict[str, Any]], modalities: list[str]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    out = []
    for cls, items in sorted(by_class.items()):
        entry = {"class": cls, "objects": len(items)}
        for m in modalities:
            entry[f"{m}_importance"] = float(np.mean([it[f"{m}_importance"] for it in items]))
        if len(modalities) == 2:
            a, b = modalities
            entry["trusted_sensor"] = a if entry[f"{a}_importance"] >= entry[f"{b}_importance"] else b
        out.append(entry)
    return out


def channel_gate_table(gates: np.ndarray, names: list[str], top_k: int = 8) -> dict[str, Any]:
    """Named (raster) channels + the top-k learned channels by gate value."""
    named = [(n, float(g)) for n, g in zip(names, gates) if "learned" not in n]
    learned = sorted([(n, float(g)) for n, g in zip(names, gates) if "learned" in n], key=lambda x: -x[1])
    return {"named": named, "top_learned": learned[:top_k],
            "learned_mean": float(np.mean([g for _, g in learned])) if learned else float("nan"),
            "selected_ratio": float((gates > 0.5).mean())}
