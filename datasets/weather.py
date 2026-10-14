"""Adverse-weather test data: real-weather scene tags and synthetic sensor corruptions.

Evaluation-only. Nothing here is used unless ``cfg.weather.synthetic`` is set (by
:mod:`evaluation.adverse_weather`), so normal training and evaluation are unchanged.

Real weather (nuScenes)
-----------------------
nuScenes has no weather field; the only source is the free-text scene description.
:func:`scene_weather_tags` derives tags from keywords in that text and always keeps the
text itself. No severity is invented: nuScenes does not say how hard it rained.

Synthetic weather (SYNTHETIC — never presented as real observations)
--------------------------------------------------------------------
Applied consistently to all sensors of a clear-weather frame:

* **Camera fog** — Koschmieder model ``I = J·t + A·(1 − t)``, ``t = exp(−β·d)``,
  ``β = 3.912 / V``. The per-pixel distance ``d`` comes from the key-frame LiDAR
  projected into the image (nearest return per block) and, where there is no LiDAR,
  from intersecting the pixel ray with the ground plane (sky -> ``sky_distance_m``).
* **Camera rain** — optical extinction ``α = 1.45·R^0.64`` dB/km (LISA, Kilic et al. 2021)
  through the same model, plus rain streaks, wet-optics blur and contrast loss
  (a visual approximation, not photometrically calibrated).
* **LiDAR** — mini-LISA detection rule ``ρ·exp(−2αr)/r² ≥ 0.9/R_max²`` with ``α = β`` (fog)
  or the rain extinction; surviving intensities are attenuated by ``exp(−2αr)``. Fog
  also adds near-range backscatter clutter (qualitative; count not calibrated).
* **Radar** — rain: ITU-R P.838-3 two-way attenuation ``2·γ·r`` with ``γ = k·R^α``
  (77 GHz) subtracted from the RCS; points falling below the frame's weakest observed
  RCS are removed. Fog: unchanged (attenuation at 77 GHz over ≤ 100 m is ≪ 1 dB).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from datasets.camera_loader import IMAGENET_MEAN, IMAGENET_STD, lidar_depth_maps

KOSCHMIEDER = 3.912          # ln(1 / 0.02): contrast threshold of the meteorological optical range
DB_PER_NEPER = 10.0 / math.log(10.0)  # 4.343 dB per neper


# --------------------------------------------------------------------- configuration
def load_weather_config(path: Optional[Path] = None) -> dict[str, Any]:
    """``configs/weather.yaml`` -> the ``weather`` mapping (plain dict)."""
    from utils.config import CONFIG_DIR, read_yaml
    return read_yaml(Path(path) if path else CONFIG_DIR / "weather.yaml")["weather"]


def expand_conditions(selection: str, wcfg: Mapping[str, Any]) -> list[str]:
    """``"fog"`` / ``"rain"`` / ``"all"`` / ``"dense_fog,heavy_rain"`` -> condition names (clear first)."""
    names: list[str] = []
    for part in [p.strip() for p in selection.split(",") if p.strip()]:
        if part in wcfg["groups"]:
            names += list(wcfg["groups"][part])
        elif part in wcfg["conditions"]:
            names.append(part)
        else:
            raise ValueError(f"Unknown weather condition {part!r}. "
                             f"Choose from {sorted(wcfg['conditions'])} or groups {sorted(wcfg['groups'])}.")
    names = list(dict.fromkeys(["clear"] + names))  # clear-weather baseline is always evaluated
    return names


def synthetic_spec(name: str, wcfg: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the dataset needs to apply one synthetic condition (stored in the run config)."""
    cond = dict(wcfg["conditions"][name])
    return {"name": name, **cond, "seed": int(wcfg["seed"]),
            "camera": dict(wcfg["camera"]), "lidar": dict(wcfg["lidar"]), "radar": dict(wcfg["radar"])}


# --------------------------------------------------------------------- physics helpers
def fog_extinction(visibility_m: float) -> float:
    """Extinction coefficient β [1/m] from the meteorological optical range (Koschmieder)."""
    return KOSCHMIEDER / float(visibility_m)


def rain_optical_extinction(rain_rate_mmh: float, coef: float = 1.45, exp: float = 0.64) -> float:
    """Optical/NIR rain extinction [1/m] from ``α = coef·R^exp`` dB/km."""
    return coef * float(rain_rate_mmh) ** exp / DB_PER_NEPER / 1000.0


def radar_rain_attenuation_db_per_km(rain_rate_mmh: float, k: float, alpha: float) -> float:
    """ITU-R P.838 specific attenuation γ = k·R^α [dB/km] (one way)."""
    return k * float(rain_rate_mmh) ** alpha


# --------------------------------------------------------------------- real weather
def scene_weather_tags(description: str, rules: Mapping[str, Any]) -> dict[str, Any]:
    """Weather tags from an original nuScenes scene description (keywords only, nothing invented)."""
    text = description.lower()
    excluded = any(k in text for k in rules.get("exclude_keywords", []))
    rain = any(k in text for k in rules["rain_keywords"]) and not excluded
    wet = any(k in text for k in rules.get("wet_keywords", []))
    night = any(k in text for k in rules["night_keywords"])
    fog = any(k in text for k in rules["fog_keywords"])
    if fog:
        condition = "fog"
    elif rain:
        condition = "rain"
    elif wet:
        condition = "wet_road"
    elif night:
        condition = "night"
    else:
        condition = "clear"
    return {"description": description, "condition": condition, "rain": rain, "fog": fog, "wet_road": wet,
            "night": night, "severity": None, "severity_note": "nuScenes provides no weather severity"}


def real_weather_manifest(db: Any, split_scenes: list[str], training_scenes: list[str],
                          rules: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One record per scene of the evaluation split: original description, tags, leakage flag."""
    training = set(training_scenes)
    out = []
    for name in split_scenes:
        scene = db.scene_by_name(name)
        rec = {"scene": name, "num_samples": int(scene["nbr_samples"]), "source": "nuScenes",
               "in_training_split": name in training,
               "sensors": "6 cameras, LIDAR_TOP, 5 radars (synchronised, calibrated)"}
        rec.update(scene_weather_tags(scene["description"], rules))
        out.append(rec)
    return out


# --------------------------------------------------------------------- camera
def _pixel_rays(cams: Mapping[str, np.ndarray], i: int, h: int, w: int) -> np.ndarray:
    """Camera-frame ray directions ``(h, w, 3)`` (z = 1) of the augmented image pixels."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    uv = np.stack([u, v], -1).reshape(-1, 2)
    pr, pt = cams["post_rots"][i].astype(np.float64), cams["post_trans"][i].astype(np.float64)
    orig = (uv - pt[:2]) @ np.linalg.inv(pr[:2, :2]).T
    rays = np.concatenate([orig, np.ones((orig.shape[0], 1))], 1) @ np.linalg.inv(cams["intrinsics"][i].astype(np.float64)).T
    return rays.reshape(h, w, 3)


def _fill_lidar_cells(cell: np.ndarray, max_gap: int, up_extend: int) -> np.ndarray:
    """Densify a sparse LiDAR depth grid (0 = empty) -> inf where still unknown.

    LiDAR rings are horizontal lines in the image, so gaps are bridged *vertically* with the
    nearer of the closest returns above / below (within ``max_gap`` cells); the top-most return of
    each column is extended upward by ``up_extend`` cells (walls, trees, vehicles continue above
    the highest ring); finally isolated holes are closed with a 3x3 minimum.
    """
    h, w = cell.shape
    d = np.where(cell > 0, cell, np.inf)
    above = np.full_like(d, np.inf)
    below = np.full_like(d, np.inf)
    ga = np.full(w, max_gap + 1)
    last_a = np.full(w, np.inf)
    for r in range(h):                       # nearest return above, within max_gap rows
        hit = np.isfinite(d[r])
        last_a = np.where(hit, d[r], last_a)
        ga = np.where(hit, 0, ga + 1)
        above[r] = np.where(ga <= max_gap, last_a, np.inf)
    gb = np.full(w, max_gap + 1)
    last_b = np.full(w, np.inf)
    for r in range(h - 1, -1, -1):           # nearest return below
        hit = np.isfinite(d[r])
        last_b = np.where(hit, d[r], last_b)
        gb = np.where(hit, 0, gb + 1)
        below[r] = np.where(gb <= max_gap, last_b, np.inf)
        # upward extension of the highest return in the column
        below[r] = np.where(~np.isfinite(above[r]) & (gb <= up_extend), last_b, below[r])
    both = np.isfinite(above) & np.isfinite(below)
    filled = np.where(np.isfinite(d), d, np.where(both, np.minimum(above, below), np.inf))
    filled = np.where(np.isfinite(filled), filled, np.where(~np.isfinite(above), below, np.inf))
    padded = np.pad(filled, 1, constant_values=np.inf)
    nb = np.min(np.stack([padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]), 0)
    return np.where(np.isfinite(filled), filled, nb)


def scene_distance(cams: Mapping[str, np.ndarray], key_lidar_ref: np.ndarray, image_size: tuple[int, int],
                   ground_z: float, sky_distance: float, cells_per_height: int = 32) -> np.ndarray:
    """Per-pixel distance along the viewing ray ``(N, H, W)`` [m]: densified LiDAR where available,
    else the ground plane, else ``sky_distance``. Smoothed so fog density varies continuously."""
    h, w = image_size
    stride = max(1, h // int(cells_per_height))
    n = cams["images"].shape[0]
    dist = np.full((n, h, w), sky_distance, dtype=np.float32)
    lidar_z = lidar_depth_maps(key_lidar_ref, cams, image_size, stride, (0.5, sky_distance))  # z-depth per cell
    for i in range(n):
        if not cams["cam_valid"][i]:
            continue
        rays = _pixel_rays(cams, i, h, w)
        norm = np.linalg.norm(rays, axis=-1)
        rot, t = cams["cam2ref"][i][:3, :3].astype(np.float64), cams["cam2ref"][i][:3, 3].astype(np.float64)
        dz = rays @ rot[2]                                   # z component of the ray in the reference frame
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(dz < -1e-6, (ground_z - t[2]) / dz, np.inf)
        ground = np.where(np.isfinite(s) & (s > 0), s * norm, sky_distance)
        filled = _fill_lidar_cells(lidar_z[i], max_gap=3, up_extend=4)
        up = np.repeat(np.repeat(filled, stride, 0), stride, 1)
        up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")[:h, :w]
        lidar = np.where(np.isfinite(up), up * norm, np.inf)
        d = np.minimum(np.minimum(ground, lidar), sky_distance).astype(np.float32)
        # smooth in log-distance (block edges of the LiDAR cells would otherwise show in the fog)
        d = np.exp(cv2.GaussianBlur(np.log(np.maximum(d, 0.5)), (0, 0), max(stride / 2.0, 0.5)))
        dist[i] = np.minimum(d, sky_distance).astype(np.float32)
    return dist


def _denormalise(images: np.ndarray) -> np.ndarray:
    return np.clip(images.transpose(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


def _normalise(rgb: np.ndarray) -> np.ndarray:
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2).astype(np.float32)


def apply_fog(rgb: np.ndarray, dist: np.ndarray, beta: float, airlight: float) -> tuple[np.ndarray, np.ndarray]:
    """Koschmieder fog on ``(H, W, 3)`` linear RGB in [0, 1]; returns image and transmission."""
    t = np.exp(-beta * dist)[..., None]
    return rgb * t + airlight * (1.0 - t), t[..., 0]


def apply_rain_effects(rgb: np.ndarray, rate: float, rng: np.random.Generator, ccfg: Mapping[str, Any],
                       scale: float) -> np.ndarray:
    """Rain streaks + wet-optics blur + contrast loss (visual approximation)."""
    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32).copy()
    mean = out.mean((0, 1), keepdims=True)
    out = mean + (out - mean) * max(0.0, 1.0 - float(ccfg["rain_contrast_loss_per_mmh"]) * rate)
    streaks = np.zeros((h, w), np.float32)
    n = int(round(float(ccfg["rain_streaks_per_mmh"]) * rate * scale * scale))
    angle = np.deg2rad(rng.normal(100.0, 4.0))  # near-vertical, slightly slanted (wind)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        length = rng.uniform(4.0, 12.0) * scale
        x2, y2 = x + length * math.cos(angle), y + length * math.sin(angle)
        cv2.line(streaks, (int(x), int(y)), (int(x2), int(y2)), float(rng.uniform(0.5, 1.0)), 1, cv2.LINE_AA)
    a = float(ccfg["rain_streak_alpha"]) * streaks[..., None]
    out = out * (1.0 - a) + 0.9 * a
    sigma = float(ccfg["rain_blur_per_mmh"]) * rate * scale
    if sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    return np.clip(out, 0.0, 1.0)


def corrupt_display_image(rgb: np.ndarray, cam_rec: Mapping[str, Any], key_lidar_ref: np.ndarray,
                          weather: "SyntheticWeather", seed_index: int, scale: float = 0.5) -> np.ndarray:
    """Apply a synthetic condition to a full-resolution uint8 camera image for *display*.

    Same model as the network input (:meth:`SyntheticWeather.camera`), computed at ``scale`` of
    the original resolution for speed and resized back, so box overlays keep their pixel coordinates.
    """
    h, w = rgb.shape[:2]
    hs, ws = int(round(h * scale)), int(round(w * scale))
    small = cv2.resize(rgb, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cams = {"images": _normalise(small[None]), "intrinsics": np.asarray(cam_rec["intrinsic"], np.float32)[None],
            "post_rots": np.diag([scale, scale, 1.0]).astype(np.float32)[None], "post_trans": np.zeros((1, 3), np.float32),
            "cam2ref": np.asarray(cam_rec["sensor2ref"], np.float32)[None], "cam_valid": np.array([True])}
    imgs, _ = weather.camera(cams, key_lidar_ref, (hs, ws), seed_index)
    out = cv2.resize(_denormalise(imgs)[0], (w, h), interpolation=cv2.INTER_LINEAR)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------- synthetic weather
class SyntheticWeather:
    """Apply one synthetic weather condition to the sensors of a sample (deterministic per index)."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        spec = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        self.spec = spec
        self.name = spec["name"]
        self.kind = spec["kind"]
        self.seed = int(spec.get("seed", 0))
        self.ccfg, self.lcfg, self.rcfg = spec["camera"], spec["lidar"], spec["radar"]
        if self.kind == "fog":
            self.optical_ext = fog_extinction(spec["visibility_m"])
        elif self.kind == "rain":
            self.optical_ext = rain_optical_extinction(spec["rain_rate_mmh"], float(self.ccfg["rain_extinction_coef"]),
                                                       float(self.ccfg["rain_extinction_exp"]))
        elif self.kind == "clear":
            self.optical_ext = 0.0
        else:
            raise ValueError(f"Unknown weather kind {self.kind!r}")

    def rng(self, index: int, sensor: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, int(index), sensor])

    # ---------------------------------------------------------------- camera
    def camera(self, cams: dict[str, np.ndarray], key_lidar_ref: np.ndarray, image_size: tuple[int, int],
               index: int) -> tuple[np.ndarray, dict[str, float]]:
        """Corrupt the normalised ``cams['images']``; returns new images and statistics."""
        images = cams["images"]
        if self.kind == "clear":
            return images, {"camera_mean_transmission": 1.0}
        rgb = _denormalise(images)
        dist = scene_distance(cams, key_lidar_ref, image_size, float(self.ccfg["ground_z"]),
                              float(self.ccfg["sky_distance_m"]), int(self.ccfg["lidar_depth_cells_per_height"]))
        rng = self.rng(index, 0)
        scale = image_size[0] / 128.0
        trans, contrast_in, contrast_out = [], [], []
        for i in range(rgb.shape[0]):
            if not cams["cam_valid"][i]:
                continue
            contrast_in.append(float(rgb[i].mean(-1).std()))
            rgb[i], t = apply_fog(rgb[i], dist[i], self.optical_ext, float(self.ccfg["airlight"]))
            if self.kind == "rain":
                rgb[i] = apply_rain_effects(rgb[i], float(self.spec["rain_rate_mmh"]), rng, self.ccfg, scale)
            trans.append(float(t.mean()))
            contrast_out.append(float(rgb[i].mean(-1).std()))
        stats = {"camera_mean_transmission": float(np.mean(trans)) if trans else float("nan"),
                 "camera_rms_contrast_ratio": float(np.mean(contrast_out) / max(np.mean(contrast_in), 1e-6))
                 if contrast_in else float("nan")}
        return _normalise(rgb), stats

    # ---------------------------------------------------------------- lidar
    def lidar(self, points: np.ndarray, sensor_xyz: np.ndarray, index: int) -> tuple[np.ndarray, dict[str, float]]:
        n_in = int(points.shape[0])
        if self.kind == "clear" or n_in == 0:
            return points, {"lidar_points_in": n_in, "lidar_retained_ratio": 1.0, "lidar_backscatter_points": 0}
        ext = self.optical_ext
        rel = points[:, :3].astype(np.float64) - sensor_xyz[None]
        r = np.maximum(np.linalg.norm(rel, axis=1), 0.5)
        rmax = float(self.lcfg["max_range_m"])
        p_min = 0.9 * r ** 2 / rmax ** 2                       # reflectivity needed at range r (clear weather)
        med = max(float(np.median(points[:, 3])), 1e-3)
        rho = np.maximum(0.9 * points[:, 3] / med, p_min)      # observed -> was detectable
        trans = np.exp(-2.0 * ext * r)
        keep = rho * trans >= p_min
        out = points[keep].copy()
        out[:, 3] = out[:, 3] * trans[keep]
        added = 0
        if self.kind == "fog":
            rng = self.rng(index, 1)
            added = int(round(float(self.lcfg["backscatter_points_per_beta"]) * ext))
            if added:
                rr = rng.uniform(1.5, float(self.lcfg["backscatter_max_range_m"]), added)
                az = rng.uniform(-np.pi, np.pi, added)
                clutter = np.zeros((added, points.shape[1]), np.float32)
                clutter[:, 0] = sensor_xyz[0] + rr * np.cos(az)
                clutter[:, 1] = sensor_xyz[1] + rr * np.sin(az)
                clutter[:, 2] = sensor_xyz[2] + rng.uniform(-1.0, 0.3, added)
                clutter[:, 3] = rng.uniform(0.0, 0.3 * med, added)
                out = np.concatenate([out, clutter], 0)
        stats = {"lidar_points_in": n_in, "lidar_retained_ratio": float(keep.mean()), "lidar_backscatter_points": added,
                 "lidar_max_range_kept_m": float(r[keep].max()) if keep.any() else 0.0}
        return out, stats

    # ---------------------------------------------------------------- radar
    def radar(self, points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        n_in = int(points.shape[0])
        if self.kind != "rain" or n_in == 0:  # fog: unchanged at 77 GHz (see module docstring)
            return points, {"radar_points_in": n_in, "radar_retained_ratio": 1.0, "radar_mean_loss_db": 0.0}
        gamma = radar_rain_attenuation_db_per_km(self.spec["rain_rate_mmh"], float(self.rcfg["rain_k"]),
                                                 float(self.rcfg["rain_alpha"]))
        r = np.hypot(points[:, 0], points[:, 1])
        loss = 2.0 * gamma * r / 1000.0
        floor = float(points[:, 3].min())
        rcs = points[:, 3] - loss
        keep = rcs >= floor
        out = points[keep].copy()
        out[:, 3] = rcs[keep]
        return out, {"radar_points_in": n_in, "radar_retained_ratio": float(keep.mean()),
                     "radar_mean_loss_db": float(loss.mean())}
