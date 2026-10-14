"""Cross-check the in-memory nuScenes metrics against the official devkit.

Perturbed ground truth (noise on position / size / yaw / velocity, 20 % misses, false
positives, random scores) is evaluated with both implementations; the numbers must agree.

    python scripts/check_eval_vs_devkit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from datasets.info_builder import load_infos  # noqa: E402
from evaluation.detection_metrics import DetectionEvaluator  # noqa: E402
from evaluation.official_eval import run_official_eval, to_submission  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402


def fake_predictions(info: dict, rng: np.random.Generator, n_classes: int) -> dict[str, np.ndarray]:
    gt = info["gt"]
    keep = rng.random(len(gt["labels"])) > 0.2
    boxes = gt["boxes"][keep].astype(np.float64).copy()
    labels = gt["labels"][keep].copy()
    boxes[:, :2] += rng.normal(0, 0.4, (len(boxes), 2))
    boxes[:, 2] += rng.normal(0, 0.2, len(boxes))
    boxes[:, 3:6] *= rng.uniform(0.85, 1.15, (len(boxes), 3))
    boxes[:, 6] += rng.normal(0, 0.15, len(boxes))
    boxes[:, 7:9] += rng.normal(0, 0.5, (len(boxes), 2))
    n_fp = rng.integers(0, 6)
    fp = np.zeros((n_fp, 9))
    fp[:, :2] = rng.uniform(-40, 40, (n_fp, 2))
    fp[:, 3:6] = [4.0, 1.8, 1.6]
    boxes = np.concatenate([boxes, fp])
    labels = np.concatenate([labels, rng.integers(0, n_classes, n_fp)])
    scores = rng.uniform(0.05, 1.0, len(boxes))
    return {"boxes": boxes, "labels": labels, "scores": scores}


def main() -> None:
    setup_logging()
    cfg = load_config()
    infos = load_infos(cfg, "mini_val")
    classes = list(cfg.classes)
    rng = np.random.default_rng(0)
    preds = {info["token"]: fake_predictions(info, rng, len(classes)) for info in infos}

    ours = DetectionEvaluator(classes)
    for info in infos:
        p, g = preds[info["token"]], info["gt"]
        ours.add_sample(info["token"], p["boxes"], p["scores"], p["labels"], g["boxes"], g["labels"], g["attributes"],
                        g["num_lidar_pts"] + g["num_radar_pts"], g["velocity_valid"], bike_racks=g["bike_racks"])
    mine = ours.evaluate()

    sub = to_submission(preds, {i["token"]: i for i in infos}, classes, {"lidar": True})
    official = run_official_eval(sub, cfg.dataset.root, cfg.dataset.version, "mini_val",
                                 resolve_path(cfg.runtime.output_dir) / "eval_check")
    tp = official["tp_errors"]
    rows = [("NDS", mine["NDS"], official["nd_score"]), ("mAP", mine["mAP"], official["mean_ap"]),
            ("mATE", mine["mATE"], tp["trans_err"]), ("mASE", mine["mASE"], tp["scale_err"]),
            ("mAOE", mine["mAOE"], tp["orient_err"]), ("mAVE", mine["mAVE"], tp["vel_err"]),
            ("mAAE", mine["mAAE"], tp["attr_err"])]
    print(f"\n{'metric':<8}{'ours':>10}{'devkit':>10}{'|diff|':>10}")
    for name, a, b in rows:
        print(f"{name:<8}{a:>10.4f}{b:>10.4f}{abs(a - b):>10.4f}")
    for c in classes:
        print(f"  AP {c:<22}{mine['per_class'][c]['AP']:>8.4f}{official['mean_dist_aps'][c]:>10.4f}")
    worst = max(abs(a - b) for _, a, b in rows)
    print(json.dumps({"max_abs_diff": worst, "agree": worst < 1e-3}))


if __name__ == "__main__":
    main()
