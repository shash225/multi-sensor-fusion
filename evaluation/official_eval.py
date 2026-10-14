"""Export detections to the official nuScenes submission format and run the devkit evaluation.

The in-memory evaluator (``detection_metrics.py``) is used during training and in the
dashboard; this module cross-checks it with ``nuscenes.eval.detection.evaluate.DetectionEval``.
Requires ``pip install nuscenes-devkit --no-deps`` (+ pyquaternion, shapely, descartes).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.detection_metrics import CLASS_RANGE, default_attribute
from utils.geometry import rotmat_to_quat


def box_to_global(box: np.ndarray, ref_ego2global: np.ndarray) -> dict[str, Any]:
    """Reference-frame box -> nuScenes global translation / size / rotation / velocity."""
    rot, t = ref_ego2global[:3, :3], ref_ego2global[:3, 3]
    centre = rot @ box[:3] + t
    yaw_rot = np.array([[np.cos(box[6]), -np.sin(box[6]), 0], [np.sin(box[6]), np.cos(box[6]), 0], [0, 0, 1]])
    quat = rotmat_to_quat(rot @ yaw_rot)
    vel = (rot @ np.array([box[7], box[8], 0.0]))[:2]
    return {"translation": centre.tolist(), "size": [float(box[4]), float(box[3]), float(box[5])],  # [w, l, h]
            "rotation": quat.tolist(), "velocity": vel.tolist()}


def to_submission(results: dict[str, dict[str, np.ndarray]], infos_by_token: dict[str, dict], classes: list[str],
                  modalities: dict[str, bool]) -> dict[str, Any]:
    out: dict[str, list] = {}
    for token, det in results.items():
        info = infos_by_token[token]
        entries = []
        for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
            name = classes[int(label)]
            if np.hypot(box[0], box[1]) > CLASS_RANGE[name]:
                continue
            entries.append({"sample_token": token, **box_to_global(np.asarray(box, np.float64), info["ref_ego2global"]),
                            "detection_name": name, "detection_score": float(score),
                            "attribute_name": default_attribute(name, np.asarray(box[7:9]))})
        out[token] = entries[:500]
    return {"meta": {"use_camera": modalities.get("camera", False), "use_lidar": modalities.get("lidar", False),
                     "use_radar": modalities.get("radar", False), "use_map": False, "use_external": False},
            "results": out}


def run_official_eval(submission: dict[str, Any], dataroot: str, version: str, eval_set: str, out_dir: Path) -> dict:
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results_nusc.json"
    path.write_text(json.dumps(submission))
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    evaluator = DetectionEval(nusc, config_factory("detection_cvpr_2019"), str(path), eval_set, str(out_dir), verbose=False)
    metrics, _ = evaluator.evaluate()
    return metrics.serialize()
