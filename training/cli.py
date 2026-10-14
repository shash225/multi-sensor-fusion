"""Shared command-line entry point for all training scripts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.trainer import Trainer  # noqa: E402

CURRICULUM = ("camera_bev", "lidar", "radar", "fusion", "detection", "trajectory", "end_to_end")


def run(default_stage: Optional[str] = None, argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train one stage of the multi-sensor fusion curriculum.")
    if default_stage is None:
        parser.add_argument("--stage", required=True, help=f"one of {CURRICULUM + ('joint', 'curriculum')}")
    parser.add_argument("--experiment", default="F_cam_lidar_radar_fs", help="entry of configs/ablation.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override the stage's epoch count")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.batch_size=2")
    args = parser.parse_args(argv)
    stage = default_stage or args.stage
    stages = CURRICULUM if stage == "curriculum" else (stage,)
    for s in stages:
        trainer = Trainer(args.experiment, s, args.set, args.epochs, args.resume if len(stages) == 1 else None)
        res = trainer.fit()
        det = res.get("detection", {})
        print(json.dumps({"stage": s, "checkpoint": str(trainer.ckpt_dir / "best.pt"),
                          "NDS": det.get("NDS"), "mAP": det.get("mAP"),
                          "planning": res.get("planning_learned")}, indent=2, default=float))


if __name__ == "__main__":
    run()
