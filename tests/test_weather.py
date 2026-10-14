"""Adverse-weather module: tagging, physics monotonicity, determinism, metrics (no dataset needed)."""
from __future__ import annotations

import numpy as np
import pytest

from datasets.weather import (SyntheticWeather, apply_fog, expand_conditions, fog_extinction, load_weather_config,
                              radar_rain_attenuation_db_per_km, rain_optical_extinction, scene_weather_tags,
                              synthetic_spec)
from evaluation.detection_metrics import DetectionEvaluator

W = load_weather_config()


def _lidar(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(2, 60, n)
    az = rng.uniform(-np.pi, np.pi, n)
    pts = np.zeros((n, 5), np.float32)
    pts[:, 0], pts[:, 1], pts[:, 2] = r * np.cos(az), r * np.sin(az), rng.uniform(-1.5, 1.0, n)
    pts[:, 3] = rng.uniform(1, 60, n)
    return pts


def _radar(n=500, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 10), np.float32)
    pts[:, 0], pts[:, 1] = rng.uniform(-50, 50, n), rng.uniform(-50, 50, n)
    pts[:, 3] = rng.uniform(-5, 30, n)
    return pts


def test_expand_conditions_always_includes_clear_first():
    assert expand_conditions("fog", W) == ["clear", "light_fog", "dense_fog"]
    assert expand_conditions("heavy_rain", W) == ["clear", "heavy_rain"]
    with pytest.raises(ValueError):
        expand_conditions("snow", W)


@pytest.mark.parametrize("text,cond,rain", [
    ("Night, after rain, many peds", "wet_road", False),
    ("Rain, intersection, heavy traffic", "rain", True),
    ("Parking lot, parked cars", "clear", False),
    ("Night, big street", "night", False),
])
def test_scene_tags_use_only_the_description(text, cond, rain):
    tags = scene_weather_tags(text, W["real_weather"])
    assert tags["condition"] == cond and tags["rain"] == rain
    assert tags["severity"] is None and tags["description"] == text


def test_physics_constants():
    assert fog_extinction(50) == pytest.approx(3.912 / 50)
    assert rain_optical_extinction(25) > rain_optical_extinction(2.5) > 0
    assert radar_rain_attenuation_db_per_km(25, 1.0733, 0.7104) == pytest.approx(1.0733 * 25 ** 0.7104)


def test_fog_image_limits():
    rgb = np.full((4, 4, 3), 0.1, np.float32)
    near, t0 = apply_fog(rgb, np.zeros((4, 4)), 0.05, 0.8)
    far, t1 = apply_fog(rgb, np.full((4, 4), 1e4), 0.05, 0.8)
    assert np.allclose(near, rgb) and np.allclose(t0, 1)
    assert np.allclose(far, 0.8, atol=1e-3) and np.allclose(t1, 0, atol=1e-3)


def test_clear_is_identity():
    w = SyntheticWeather(synthetic_spec("clear", W))
    pts, rad = _lidar(), _radar()
    assert w.lidar(pts, np.zeros(3), 0)[0] is pts
    assert w.radar(rad)[0] is rad


def test_lidar_fog_monotonic_and_far_first():
    pts = _lidar()
    light = SyntheticWeather(synthetic_spec("light_fog", W)).lidar(pts, np.zeros(3), 0)
    dense = SyntheticWeather(synthetic_spec("dense_fog", W)).lidar(pts, np.zeros(3), 0)
    assert dense[1]["lidar_retained_ratio"] < light[1]["lidar_retained_ratio"] < 1.0
    assert dense[1]["lidar_max_range_kept_m"] < light[1]["lidar_max_range_kept_m"]
    assert dense[1]["lidar_backscatter_points"] > light[1]["lidar_backscatter_points"] > 0


def test_lidar_rain_deterministic():
    pts = _lidar()
    w = SyntheticWeather(synthetic_spec("heavy_rain", W))
    a, b = w.lidar(pts, np.zeros(3), 7)[0], w.lidar(pts, np.zeros(3), 7)[0]
    assert np.array_equal(a, b)


def test_radar_fog_unchanged_rain_attenuates():
    rad = _radar()
    assert SyntheticWeather(synthetic_spec("dense_fog", W)).radar(rad)[0] is rad
    light = SyntheticWeather(synthetic_spec("light_rain", W)).radar(rad)[1]
    heavy = SyntheticWeather(synthetic_spec("heavy_rain", W)).radar(rad)[1]
    assert heavy["radar_mean_loss_db"] > light["radar_mean_loss_db"] > 0
    assert heavy["radar_retained_ratio"] <= light["radar_retained_ratio"] <= 1.0


def test_precision_recall_counts():
    ev = DetectionEvaluator(["car", "pedestrian"])
    gt = np.array([[10, 0, 0, 4, 2, 1.5, 0, 0, 0], [20, 5, 0, 4, 2, 1.5, 0, 0, 0]], np.float32)
    pred = np.array([[10.5, 0, 0, 4, 2, 1.5, 0, 0, 0], [30, 0, 0, 4, 2, 1.5, 0, 0, 0]], np.float32)
    ev.add_sample("t", pred, np.array([0.9, 0.8]), np.array([0, 0]), gt, np.array([0, 0]))
    pr = ev.precision_recall(2.0, 0.3)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (1, 1, 1)
    assert pr["precision"] == pytest.approx(0.5) and pr["recall"] == pytest.approx(0.5)
    assert pr["mean_center_error_m"] == pytest.approx(0.5)
