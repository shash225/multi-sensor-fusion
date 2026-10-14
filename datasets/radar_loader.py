"""Radar loading: binary PCD parsing, filtering, multi-sweep accumulation.

nuScenes radars (Continental ARS 408) report ~100 detections per sweep with
ego-motion-compensated Doppler velocity (``vx_comp``, ``vy_comp``) in the *sensor*
frame. The official devkit's multi-sweep loader transforms positions but not the
velocity vectors; here velocities are rotated into the reference frame as well,
because velocity is the radar's most valuable cue for dynamic objects.

Output points: ``(N, 10)`` float32 with columns :data:`RADAR_POINT_FEATURES`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from utils.bev_grid import BEVGrid
from utils.logging_utils import WarnOnce

logger = logging.getLogger(__name__)
_warn = WarnOnce(logger)

RADAR_POINT_FEATURES = ("x", "y", "z", "rcs", "vx_comp", "vy_comp", "vx", "vy", "dyn_prop", "time_lag")
_PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4",
              ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}


def read_radar_pcd(path: Path) -> np.ndarray:
    """Parse a nuScenes radar ``.pcd`` (``DATA binary``) into a NumPy structured array."""
    raw = Path(path).read_bytes()
    marker = raw.find(b"DATA binary")
    if marker < 0:
        raise ValueError(f"Unsupported PCD (expected 'DATA binary'): {path}")
    header: dict[str, list[str]] = {}
    for line in raw[:marker].decode("ascii", errors="replace").splitlines():
        if line and not line.startswith("#"):
            key, *values = line.split()
            header[key] = values
    fields, sizes, types = header["FIELDS"], header["SIZE"], header["TYPE"]
    dtype = np.dtype([(f, _PCD_TYPES[(t, int(s))]) for f, s, t in zip(fields, sizes, types)])  # packed
    n_points = int(header["POINTS"][0])
    start = raw.index(b"\n", marker) + 1
    if len(raw) - start < n_points * dtype.itemsize:
        raise ValueError(f"Truncated radar file {path}")
    return np.frombuffer(raw, dtype=dtype, count=n_points, offset=start)


class RadarLoader:
    """Load all radars of a sample (key-frame + previous sweeps) in the reference frame."""

    def __init__(self, root: Path, grid: BEVGrid, num_sweeps: int = 6,
                 invalid_states: Optional[Iterable[int]] = (0,),
                 dynprop_states: Optional[Iterable[int]] = range(7),
                 ambig_states: Optional[Iterable[int]] = (3,)) -> None:
        self.root = Path(root)
        self.grid = grid
        self.num_sweeps = num_sweeps
        self.invalid_states = None if invalid_states is None else np.asarray(list(invalid_states))
        self.dynprop_states = None if dynprop_states is None else np.asarray(list(dynprop_states))
        self.ambig_states = None if ambig_states is None else np.asarray(list(ambig_states))

    def _filter(self, pts: np.ndarray) -> np.ndarray:
        keep = np.ones(len(pts), dtype=bool)
        if self.invalid_states is not None:
            keep &= np.isin(pts["invalid_state"], self.invalid_states)
        if self.dynprop_states is not None:
            keep &= np.isin(pts["dyn_prop"], self.dynprop_states)
        if self.ambig_states is not None:
            keep &= np.isin(pts["ambig_state"], self.ambig_states)
        return pts[keep]

    def load_sweep(self, rec: dict[str, Any]) -> Optional[np.ndarray]:
        path = self.root / rec["path"]
        try:
            pts = self._filter(read_radar_pcd(path))
        except (OSError, ValueError, KeyError) as exc:
            _warn(str(path), f"Skipping radar sweep {path}: {exc}")
            return None
        n = len(pts)
        out = np.zeros((n, len(RADAR_POINT_FEATURES)), dtype=np.float32)
        if n == 0:
            return out
        mat = rec["sensor2ref"]
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float64)
        out[:, 0:3] = xyz @ mat[:3, :3].T + mat[:3, 3]
        out[:, 3] = pts["rcs"]
        rot = mat[:2, :2]  # velocities are vectors: rotate only
        out[:, 4:6] = np.stack([pts["vx_comp"], pts["vy_comp"]], -1) @ rot.T
        out[:, 6:8] = np.stack([pts["vx"], pts["vy"]], -1) @ rot.T
        out[:, 8] = pts["dyn_prop"]
        out[:, 9] = rec["time_lag"]
        return out

    def load(self, info: dict[str, Any], num_sweeps: Optional[int] = None, filter_range: bool = True) -> np.ndarray:
        n = num_sweeps or self.num_sweeps
        chunks = []
        for recs in info["radars"].values():
            for rec in recs[:n]:
                pts = self.load_sweep(rec)
                if pts is not None and len(pts):
                    chunks.append(pts)
        if not chunks:
            return np.zeros((0, len(RADAR_POINT_FEATURES)), dtype=np.float32)
        points = np.concatenate(chunks, 0)
        if filter_range:
            points = points[self.grid.in_range(points[:, 0], points[:, 1])]
        return points
