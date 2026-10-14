"""Evaluate planners on a split with *ground-truth* objects as perception (oracle upper bound),
or with the constant-velocity baseline, without needing a trained network.

    python scripts/eval_planner.py --planner baseline
    python scripts/eval_planner.py --planner constant_velocity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from datasets.info_builder import load_infos  # noqa: E402
from datasets.lidar_loader import LidarLoader  # noqa: E402
from evaluation.trajectory_metrics import aggregate, sample_planning_metrics  # noqa: E402
from planning.trajectory_planner import build_planner  # noqa: E402
from utils.bev_grid import BEVGrid  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", default="baseline", choices=["baseline", "constant_velocity", "logged"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(overrides=args.set)
    infos = load_infos(cfg, args.split or cfg.dataset.val_split)
    classes = list(cfg.classes)
    collision_labels = [classes.index(c) for c in cfg.planning.collision_classes]
    planner = build_planner(cfg, "baseline") if args.planner == "baseline" else None
    lidar = LidarLoader(Path(cfg.dataset.root), BEVGrid.from_config(cfg.bev), 1)
    dt, horizon = float(cfg.planning.dt), int(cfg.planning.horizon)
    results, times, flagged = [], [], 0
    for info in infos:
        fut = info["ego_future"]
        if not fut["mask"].all():
            continue  # standard protocol: only samples with a complete 3 s future
        es = info["ego_state"]
        state = np.array([es["speed"], es["yaw_rate"], es["accel"]])
        if args.planner == "logged":  # sanity check: the human trajectory scored against itself
            pred = fut["xy"].copy()
        elif planner is None:
            t = np.arange(1, horizon + 1) * dt
            pred = np.stack([max(es["speed"], 0.0) * t, np.zeros(horizon)], -1)
        else:
            gt = info["gt"]
            dets = {"boxes": gt["boxes"].astype(np.float64), "labels": gt["labels"], "scores": np.ones(len(gt["labels"]))}
            res = planner.plan(state, info["command"], dets, lidar_points=lidar.load_keyframe(info))
            pred, flagged = res.waypoints, flagged + int(res.collision)
            times.append(res.time_ms)
        results.append(sample_planning_metrics(pred, fut["xy"], fut["mask"], dt, info["agent_future"], collision_labels,
                                               cfg.planning.ego))
    summary = aggregate(results)
    if times:
        summary["planner_ms_mean"] = float(np.mean(times))
        summary["planner_self_reported_collisions"] = flagged
    print(json.dumps({"planner": args.planner, **summary}, indent=2))


if __name__ == "__main__":
    main()
