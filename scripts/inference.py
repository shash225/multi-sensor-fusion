"""Run the full pipeline over a nuScenes scene and export composite frames, an MP4 and results JSON.

    python scripts/inference.py --scene scene-0103 --video
    python scripts/inference.py --scene scene-0916 --checkpoint checkpoints/F_cam_lidar_radar_fs/joint/best.pt
    python scripts/inference.py --scene scene-0103 --mode ground_truth      # no model needed

Without ``--checkpoint`` the newest checkpoint of ``--experiment`` is used; if none exists
the script runs in ground-truth mode and says so on every frame (never fake predictions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.ui import panels  # noqa: E402
from app.ui.resources import find_checkpoint  # noqa: E402
from evaluation.detection_metrics import DetectionEvaluator  # noqa: E402
from evaluation.official_eval import box_to_global  # noqa: E402
from evaluation.trajectory_metrics import aggregate  # noqa: E402
from inference.pipeline import FusionPipeline  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from visualization.common import put_label, stack_images  # noqa: E402


def compose(res, pipe: FusionPipeline, images: dict, score_min: float) -> np.ndarray:
    """1920x1080 frame: camera | fusion tiles (top), driving BEV | trajectory (bottom)."""
    cam, _ = panels.camera_panel(res, images, "CAM_FRONT", True, True, score_min)
    cam = cv2.resize(cam if cam is not None else np.zeros((900, 1600, 3), np.uint8), (960, 540))
    tiles = panels.fusion_tiles(res, images, res.mode == "model" and pipe.model is not None and pipe.model.use_feature_selection)
    grid = cv2.resize(stack_images([cv2.resize(t, (300, 300)) for t in tiles], cols=3, pad=4), (960, 540))
    bev = cv2.resize(panels.driving_bev(res, pipe, {"show_map": True, "show_lidar": True, "show_radar": True, "show_gt": True,
                                                    "show_tracks": True, "score_min": score_min}), (540, 540))
    traj = cv2.resize(panels.trajectory_closeup(res, pipe), (420, 420))
    side = np.full((540, 420, 3), 16, np.uint8)
    side[:420] = traj
    fm = res.frame_metrics
    lines = [f"mode: {'MODEL PREDICTION' if res.mode == 'model' and res.model_trained else 'GROUND TRUTH / DATASET VISUALIZATION'}",
             f"command {res.command_name}  speed {res.info['ego_state']['speed']:.1f} m/s"]
    if "planning" in fm:
        p = fm["planning"]
        lines.append(f"ADE {p['ade']:.2f} m  FDE {p['fde']:.2f} m  collision {'YES' if p.get('collision') else 'no'}")
    if "detection" in fm:
        lines.append(f"frame NDS {fm['detection']['NDS']:.3f}  mAP {fm['detection']['mAP']:.3f}")
    lines.append(f"pipeline {res.timings_ms.get('total', 0):.0f} ms")
    for i, text in enumerate(lines):
        put_label(side, text, (8, 440 + i * 20), 0.45, bg=None)
    board = np.full((540, 1920 - 540 - 420, 3), 16, np.uint8)
    title = "MODEL PREDICTION" if res.mode == "model" and res.model_trained else "GROUND TRUTH / DATASET VISUALIZATION"
    put_label(board, title, (16, 34), 0.75, (255, 255, 255), (30, 90, 40) if "MODEL" in title else (120, 90, 20))
    cols_x = (16, 230, 330, 430, 530)  # fixed column positions (Hershey fonts are proportional)
    for x, head in zip(cols_x, ("Object", "conf", "x (m)", "y (m)", "v (m/s)")):
        put_label(board, head, (x, 80), 0.5, (170, 175, 190), None)
    d = res.detections
    if d is not None and len(d["boxes"]):
        order = [i for i in np.argsort(np.hypot(d["boxes"][:, 0], d["boxes"][:, 1])) if d["scores"][i] >= score_min][:16]
        for row, i in enumerate(order):
            b = d["boxes"][i]
            cells = (d["names"][i].replace("_", " "), f"{d['scores'][i]:.2f}", f"{b[0]:.1f}", f"{b[1]:.1f}",
                     f"{np.hypot(b[7], b[8]):.1f}")
            for x, text in zip(cols_x, cells):
                put_label(board, text, (x, 108 + row * 24), 0.5, (230, 232, 238), None)
    bottom = np.concatenate([bev, side, board], 1)
    top = np.concatenate([cam, grid], 1)
    frame = np.concatenate([top, bottom], 0)
    put_label(frame, f"{res.info['scene_name']}  frame {res.info.get('frame_idx', 0) + 1}/{res.info.get('num_frames', 0)}",
              (980, 1070), 0.55)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="scene-0103")
    parser.add_argument("--experiment", default="F_cam_lidar_radar_fs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="model", choices=["model", "ground_truth", "dataset"])
    parser.add_argument("--planner", default="baseline", choices=["baseline", "learned", "hybrid"])
    parser.add_argument("--score", type=float, default=0.3)
    parser.add_argument("--video", action="store_true", help="also write an MP4 (2 fps = real time)")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(overrides=args.set)
    ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint(cfg, args.experiment)
    mode = args.mode
    if mode == "model" and ckpt is None:
        print(f"No checkpoint for {args.experiment}: falling back to ground-truth mode (labelled on every frame).")
        mode = "ground_truth"
    pipe = FusionPipeline(cfg, ckpt if mode == "model" else None, split="all")
    scene = next((s for s in pipe.scenes() if s["name"] == args.scene), None)
    if scene is None:
        raise SystemExit(f"Scene {args.scene} not found; available: {[s['name'] for s in pipe.scenes()]}")
    out_dir = resolve_path(cfg.runtime.output_dir) / "inference" / f"{args.scene}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    det_eval = DetectionEvaluator(pipe.classes)
    plans, results = [], {}
    for k in range(scene["count"]):
        res = pipe.run(scene["start"] + k, mode, args.planner)
        images = panels.load_images(Path(cfg.dataset.root), res)
        frame = compose(res, pipe, images, args.score)
        cv2.imwrite(str(out_dir / f"frame_{k:03d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.video:
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{args.scene}_{mode}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        pipe.add_to_evaluator(det_eval, res)
        if "planning" in res.frame_metrics:
            plans.append(res.frame_metrics["planning"])
        if res.detections is not None:
            d = res.detections
            results[res.info["token"]] = [{**box_to_global(b, res.info["ref_ego2global"]), "detection_name": n,
                                           "detection_score": float(s)} for b, s, n in zip(d["boxes"], d["scores"], d["names"])]
        print(f"frame {k + 1}/{scene['count']}  {res.timings_ms.get('total', 0):.0f} ms")
    if writer is not None:
        writer.release()
    summary = {"scene": args.scene, "mode": mode, "checkpoint": str(ckpt) if mode == "model" else None,
               "planning": aggregate(plans) if plans else None}
    if mode == "model":
        det = det_eval.evaluate()
        summary["detection"] = {k: v for k, v in det.items() if k != "per_class"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "detections_global.json").write_text(json.dumps(results))
    print(json.dumps(summary, indent=2, default=float))
    print(f"frames written to {out_dir}")


if __name__ == "__main__":
    main()
