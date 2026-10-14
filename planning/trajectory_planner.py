"""Trajectory planners.

``BaselinePlanner`` (classical, no training required) — path/velocity decomposition::

    detections ─► occupancy (static C-space + CV-forecast dynamic occupancy + LiDAR/map costs)
    command ────► goal point (lookahead ∝ speed, bearing from LEFT/RIGHT/STRAIGHT)
                  A* geometric path on the static cost map
                  speed profile: best constant acceleration that keeps the ego footprint
                                 free of the dynamic occupancy at every step
                  iLQR MPC tracks (path, speed) with a kinematic bicycle model
    output: waypoints (T, 2) at dt = 0.5 s + collision flag

``LearnedPlanner`` wraps the network's planning head; ``HybridPlanner`` uses the learned
trajectory but falls back to the baseline whenever the learned one collides with the
predicted occupancy (safety layer). All planners share the ``plan()`` interface, so a
learned planner can replace the baseline by changing ``planning.planner`` in the config.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from planning.astar import astar, nearest_free
from planning.mpc import BicycleILQR
from planning.occupancy_grid import OccupancyBuilder

COMMAND_NAMES = ("RIGHT", "LEFT", "STRAIGHT")


@dataclass
class PlanResult:
    waypoints: np.ndarray                       # (T, 2) rear-axle positions at t = dt..T·dt
    yaws: np.ndarray                            # (T,)
    speeds: np.ndarray                          # (T,)
    collision: bool                             # footprint intersects predicted occupancy
    planner: str
    command: int
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))      # geometric (A*) path
    dense: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))     # MPC states at 10 Hz
    goal: Optional[np.ndarray] = None
    accel: float = 0.0
    time_ms: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


def _smooth_path(path: np.ndarray, window: int = 5) -> np.ndarray:
    if len(path) < 3:
        return path
    k = min(window, len(path))
    pad = np.concatenate([np.repeat(path[:1], k // 2, 0), path, np.repeat(path[-1:], k // 2, 0)])
    kernel = np.ones(k) / k
    out = np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(2)], -1)
    out[0] = path[0]
    return out


def _resample(path: np.ndarray, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return path[:1], np.zeros(1)
    grid = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(grid, s, path[:, i]) for i in range(2)], -1), grid


def _pose_at(path: np.ndarray, arclen: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate positions/headings at arc lengths ``s`` (extrapolates along the last heading)."""
    heading = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0])) if len(path) > 1 else np.zeros(1)
    xy = np.stack([np.interp(s, arclen, path[:, 0]), np.interp(s, arclen, path[:, 1])], -1)
    over = s > arclen[-1]
    if over.any():
        h = heading[-1]
        xy[over] = path[-1] + (s[over] - arclen[-1])[:, None] * np.array([np.cos(h), np.sin(h)])
    yaw = np.interp(s, arclen, np.unwrap(heading))
    return xy, yaw


class BaselinePlanner:
    name = "baseline"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        p = cfg.planning
        self.p = p
        self.occ = OccupancyBuilder(p)
        self.grid = self.occ.grid
        self.mpc = BicycleILQR(p.mpc, p.ego)
        self.horizon, self.dt = int(p.horizon), float(p.dt)
        self.classes = list(cfg.classes)

    # ------------------------------------------------------------- helpers
    def _goal(self, speed: float, command: int) -> np.ndarray:
        a = self.p.astar
        dist = float(np.clip(max(speed, 2.0) * self.horizon * self.dt * 1.2, a.lookahead_min, a.lookahead_max))
        theta = {0: -1.0, 1: 1.0, 2: 0.0}[int(command)] * np.deg2rad(float(a.turn_angle_deg))
        return np.array([dist * np.cos(theta), dist * np.sin(theta)])

    def _to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        ix = int(np.clip(np.floor((xy[0] - self.grid.x_range[0]) / self.grid.resolution), 0, self.grid.nx - 1))
        iy = int(np.clip(np.floor((xy[1] - self.grid.y_range[0]) / self.grid.resolution), 0, self.grid.ny - 1))
        return iy, ix

    def _to_metric(self, cells: list[tuple[int, int]]) -> np.ndarray:
        c = np.asarray(cells, dtype=np.float64)
        return np.stack([(c[:, 1] + 0.5) * self.grid.resolution + self.grid.x_range[0],
                         (c[:, 0] + 0.5) * self.grid.resolution + self.grid.y_range[0]], -1)

    def collision_check(self, waypoints: np.ndarray, yaws: np.ndarray, dyn_occ: np.ndarray) -> np.ndarray:
        """Per-step collision flags ``(T,)`` of the ego footprint against ``dyn_occ (T, ny, nx)``."""
        pts = self.occ.footprint_points(waypoints, yaws)
        return np.array([self.occ.occupied_at(dyn_occ[t], pts[t]).any() for t in range(len(waypoints))])

    def build_scene(self, detections: dict[str, np.ndarray], lidar_points: Optional[np.ndarray] = None,
                    map_drivable: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """Occupancy layers used by planning (also returned for visualisation)."""
        occ_cfg = self.p.occupancy
        boxes = detections["boxes"]
        keep = detections["scores"] >= float(occ_cfg.min_score) if len(boxes) else np.zeros(0, bool)
        boxes = boxes[keep]
        speed = np.hypot(boxes[:, 7], boxes[:, 8]) if len(boxes) else np.zeros(0)
        moving = speed > 1.0
        half_w = float(self.p.ego.width) / 2
        static_occ = self.occ.static_occupancy(boxes[~moving], inflate=half_w + float(occ_cfg.inflation))
        dyn_occ = self.occ.dynamic_occupancy(boxes, inflate=float(occ_cfg.inflation))
        cost = 0.5 * float(self.p.astar.obstacle_clearance_cost) * self.occ.clearance_cost(static_occ)
        lidar_occ = np.zeros(self.grid.shape, np.uint8)
        if occ_cfg.use_lidar_obstacles and lidar_points is not None:
            lidar_occ = self.occ.lidar_obstacles(lidar_points, tuple(occ_cfg.lidar_z_band))
            lidar_occ = cv2.dilate(lidar_occ, np.ones((3, 3), np.uint8))
            cost += float(occ_cfg.lidar_cost) * lidar_occ
        if map_drivable is not None:
            cost += float(occ_cfg.map_cost) * (1 - map_drivable.astype(np.float32))
        cost = cost.astype(np.float64)
        cost[static_occ > 0] = np.inf
        return {"static_occ": static_occ, "dynamic_occ": dyn_occ, "lidar_occ": lidar_occ, "cost": cost}

    # ---------------------------------------------------------------- plan
    def plan(self, ego_state: np.ndarray, command: int, detections: dict[str, np.ndarray],
             lidar_points: Optional[np.ndarray] = None, map_drivable: Optional[np.ndarray] = None,
             **_: Any) -> PlanResult:
        t0 = time.perf_counter()
        speed, yaw_rate, accel0 = (float(v) for v in ego_state[:3])
        speed = max(speed, 0.0)
        scene = self.build_scene(detections, lidar_points, map_drivable)
        cost = scene["cost"].copy()
        start = self._to_cell(np.zeros(2))
        # The ego may start inside an inflated obstacle (close parked car): clear its own footprint.
        r, c = start
        cost[max(r - 3, 0):r + 4, max(c - 2, 0):c + 8] = np.where(
            np.isfinite(cost[max(r - 3, 0):r + 4, max(c - 2, 0):c + 8]), cost[max(r - 3, 0):r + 4, max(c - 2, 0):c + 8], 50.0)

        goal_xy = self._goal(speed, command)
        goal_cell = nearest_free(cost, self._to_cell(goal_xy)) or self._to_cell(goal_xy)
        cells, reached = astar(cost, start, goal_cell, int(self.p.astar.max_expansions))
        path = self._to_metric(cells) if len(cells) > 1 else np.stack([np.zeros(2), goal_xy])
        path[0] = 0.0
        path = _smooth_path(path)
        path, arclen = _resample(path)
        if len(path) < 2:
            path = np.array([[0.0, 0.0], [1.0, 0.0]])
            arclen = np.array([0.0, 1.0])

        # Speed profile: best constant acceleration that stays collision free.
        times = np.arange(1, self.horizon + 1) * self.dt
        s_ref = max(speed * times[-1] + 0.5 * np.clip(accel0, -2.0, 1.0) * times[-1] ** 2, 0.0)
        best = None
        for a in self.p.speed.accel_candidates:
            v = np.clip(speed + a * times, 0.0, float(self.p.ego.max_speed))
            # Distance with clamped speed (exact for piecewise-constant acceleration).
            s = np.cumsum(np.concatenate([[0.5 * (speed + v[0]) * self.dt], 0.5 * (v[:-1] + v[1:]) * self.dt]))
            xy, yaw = _pose_at(path, arclen, s)
            hits = self.collision_check(xy, yaw, scene["dynamic_occ"])
            c = (float(self.p.speed.progress_weight) * abs(s[-1] - s_ref) + float(self.p.speed.comfort_weight) * a * a
                 + 1e3 * hits.sum())
            if best is None or c < best[0]:
                best = (c, float(a), s, v)
        _, a_best, s_prof, v_prof = best

        # MPC tracking at 10 Hz.
        n = int(round(times[-1] / float(self.p.mpc.dt)))
        t_fine = np.arange(1, n + 1) * float(self.p.mpc.dt)
        s_fine = np.interp(t_fine, np.concatenate([[0.0], times]), np.concatenate([[0.0], s_prof]))
        v_fine = np.interp(t_fine, np.concatenate([[0.0], times]), np.concatenate([[speed], v_prof]))
        xy_ref, yaw_ref = _pose_at(path, arclen, s_fine)
        ref = np.concatenate([xy_ref, yaw_ref[:, None], v_fine[:, None], np.zeros((n, 1))], 1)
        steer0 = float(np.clip(np.arctan(yaw_rate * float(self.p.ego.wheelbase) / max(speed, 0.5)), -0.5, 0.5))
        s0 = np.array([0.0, 0.0, 0.0, speed, steer0])
        u0 = np.tile([a_best, 0.0], (n, 1))
        res = self.mpc.solve(s0, ref, u0)
        idx = np.round(times / float(self.p.mpc.dt)).astype(int)
        waypoints, yaws, speeds = res.states[idx, :2], res.states[idx, 2], res.states[idx, 3]
        collision = bool(self.collision_check(waypoints, yaws, scene["dynamic_occ"]).any())
        return PlanResult(waypoints.astype(np.float32), yaws.astype(np.float32), speeds.astype(np.float32), collision,
                          self.name, int(command), path.astype(np.float32), res.states.astype(np.float32), goal_xy,
                          a_best, (time.perf_counter() - t0) * 1000.0,
                          {"astar_reached": reached, "mpc_cost": res.cost, "mpc_iterations": res.iterations, **scene})


class LearnedPlanner:
    """Uses the network's planning-head waypoints; reports collisions against predicted occupancy."""

    name = "learned"

    def __init__(self, cfg: Any) -> None:
        self.checker = BaselinePlanner(cfg)
        self.dt = float(cfg.planning.dt)

    def plan(self, ego_state: np.ndarray, command: int, detections: dict[str, np.ndarray],
             learned_waypoints: Optional[np.ndarray] = None, lidar_points: Optional[np.ndarray] = None,
             **_: Any) -> PlanResult:
        if learned_waypoints is None:
            raise ValueError("LearnedPlanner needs the planning-head output (model.trajectory_head.enabled)")
        t0 = time.perf_counter()
        wp = np.asarray(learned_waypoints, dtype=np.float64)
        prev = np.concatenate([np.zeros((1, 2)), wp[:-1]])
        d = wp - prev
        yaws = np.arctan2(d[:, 1], d[:, 0] + 1e-9)
        yaws[np.linalg.norm(d, axis=1) < 0.05] = 0.0
        speeds = np.linalg.norm(d, axis=1) / self.dt
        scene = self.checker.build_scene(detections, lidar_points)
        collision = bool(self.checker.collision_check(wp, yaws, scene["dynamic_occ"]).any())
        return PlanResult(wp.astype(np.float32), yaws.astype(np.float32), speeds.astype(np.float32), collision, self.name,
                          int(command), time_ms=(time.perf_counter() - t0) * 1000.0, debug=scene)


class HybridPlanner:
    """Learned trajectory with a classical safety fallback."""

    name = "hybrid"

    def __init__(self, cfg: Any) -> None:
        self.learned, self.baseline = LearnedPlanner(cfg), BaselinePlanner(cfg)

    def plan(self, ego_state: np.ndarray, command: int, detections: dict[str, np.ndarray], **kwargs: Any) -> PlanResult:
        res = self.learned.plan(ego_state, command, detections, **kwargs)
        if res.collision:
            fallback = self.baseline.plan(ego_state, command, detections, **kwargs)
            fallback.planner = "hybrid(baseline fallback)"
            return fallback
        res.planner = "hybrid(learned)"
        return res


def build_planner(cfg: Any, name: Optional[str] = None):
    name = name or str(cfg.planning.planner)
    return {"baseline": BaselinePlanner, "learned": LearnedPlanner, "hybrid": HybridPlanner}[name](cfg)
