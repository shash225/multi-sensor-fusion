"""Metric and planner sanity tests."""
from __future__ import annotations

import numpy as np

from evaluation.detection_metrics import DetectionEvaluator
from evaluation.trajectory_metrics import sample_planning_metrics
from planning.astar import astar
from planning.mpc import BicycleILQR
from planning.trajectory_planner import BaselinePlanner
from utils.config import load_config

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian",
           "traffic_cone"]


def _gt():
    boxes = np.array([[10.0, 2.0, 0.5, 4.5, 1.9, 1.6, 0.1, 3.0, 0.0], [20.0, -5.0, 0.5, 0.7, 0.7, 1.7, 1.0, 1.0, 0.5]])
    return boxes, np.array([0, 8])


def test_perfect_detections_give_full_ap_and_zero_errors() -> None:
    ev = DetectionEvaluator(CLASSES)
    boxes, labels = _gt()
    for i in range(5):
        ev.add_sample(f"s{i}", boxes, np.array([0.9, 0.8]), labels, boxes, labels,
                      ["vehicle.moving", "pedestrian.moving"])
    res = ev.evaluate()
    assert np.isclose(res["per_class"]["car"]["AP"], 1.0) and np.isclose(res["per_class"]["pedestrian"]["AP"], 1.0)
    assert res["per_class"]["car"]["trans_err"] < 1e-6 and res["per_class"]["car"]["vel_err"] < 1e-6
    assert np.isclose(res["mAP_present_classes"], 1.0)


def test_shifted_detection_counts_translation_error() -> None:
    ev = DetectionEvaluator(CLASSES)
    boxes, labels = _gt()
    pred = boxes.copy()
    pred[:, 0] += 0.75
    for i in range(5):
        ev.add_sample(f"s{i}", pred, np.array([0.9, 0.8]), labels, boxes, labels)
    car = ev.evaluate()["per_class"]["car"]
    assert np.isclose(car["AP@0.5"], 0.0) and np.isclose(car["AP@1.0"], 1.0)
    assert np.isclose(car["trans_err"], 0.75, atol=1e-6)


def test_trajectory_metrics() -> None:
    gt = np.stack([np.arange(1, 7) * 2.0, np.zeros(6)], -1)
    pred = gt + np.array([0.0, 1.0])
    m = sample_planning_metrics(pred, gt, np.ones(6, bool), 0.5)
    assert np.isclose(m["ade"], 1.0) and np.isclose(m["fde"], 1.0)
    assert np.isclose(m["lateral_error"], 1.0) and np.isclose(m["longitudinal_error"], 0.0)


def test_astar_routes_around_wall() -> None:
    cost = np.zeros((20, 20))
    cost[5:15, 10] = np.inf
    path, reached = astar(cost, (10, 2), (10, 18))
    assert reached and all(np.isfinite(cost[p]) for p in path)
    assert any(r < 5 or r >= 15 for r, c in path if c == 10)


def test_mpc_tracks_straight_reference() -> None:
    cfg = load_config(use_local=False)
    mpc = BicycleILQR(cfg.planning.mpc, cfg.planning.ego)
    n = 30
    ref = np.zeros((n, 5))
    ref[:, 0] = np.arange(1, n + 1) * 0.1 * 5.0
    ref[:, 3] = 5.0
    res = mpc.solve(np.array([0, 0.5, 0, 5.0, 0]), ref)  # starts 0.5 m left of the reference
    assert abs(res.states[-1, 1]) < 0.2
    assert abs(res.states[-1, 0] - ref[-1, 0]) < 0.5


def test_baseline_planner_brakes_or_avoids_static_obstacle() -> None:
    cfg = load_config(use_local=False)
    planner = BaselinePlanner(cfg)
    obstacle = np.array([[12.0, 0.0, 0.5, 4.5, 2.0, 1.6, 0.0, 0.0, 0.0]])
    dets = {"boxes": obstacle, "labels": np.array([0]), "scores": np.array([0.9])}
    res = planner.plan(np.array([6.0, 0.0, 0.0]), 2, dets)
    assert res.waypoints.shape == (6, 2)
    assert not res.collision
