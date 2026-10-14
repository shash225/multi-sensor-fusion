"""Render every Bird's-Eye-View representation of one nuScenes frame into a single image.

Row 1 (sensor BEVs):  camera BEV (geometric IPM) | camera BEV (learned LSS) | LiDAR BEV | radar BEV
Row 2 (network BEVs): LiDAR features | radar features | LiDAR importance | radar importance | fused BEV
Plus the full autonomous-driving BEV scene (map, points, boxes, velocities, trajectories).

    python scripts/render_bev.py --scene scene-0103 --frame 12
    python scripts/render_bev.py --scene scene-0916 --frame 20 --device cuda

Learned BEVs need a checkpoint (newest one of --experiment); without one only the
sensor/geometric BEVs and the ground-truth scene are drawn, and the image says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.ui import panels  # noqa: E402
from app.ui.resources import find_checkpoint  # noqa: E402
from inference.pipeline import FusionPipeline  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from visualization.camera_view import camera_ipm_bev  # noqa: E402
from visualization.common import put_label, stack_images  # noqa: E402
from visualization.feature_selection import energy_map, importance_map, titled  # noqa: E402
from visualization.lidar_view import render_lidar_bev  # noqa: E402
from visualization.radar_view import render_radar_bev  # noqa: E402

TILE = 360


def blank(text: str) -> np.ndarray:
    img = np.full((TILE, TILE, 3), 22, np.uint8)
    return put_label(img, text, (10, TILE // 2), 0.5, (170, 170, 180), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="scene-0103")
    parser.add_argument("--frame", type=int, default=12, help="key-frame index inside the scene")
    parser.add_argument("--experiment", default="F_cam_lidar_radar_fs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(overrides=[f"runtime.device={args.device}", *args.set])
    ckpt = find_checkpoint(cfg, args.experiment)
    pipe = FusionPipeline(cfg, ckpt, split="all")
    scene = next(s for s in pipe.scenes() if s["name"] == args.scene)
    index = scene["start"] + min(args.frame, scene["count"] - 1)
    res = pipe.run(index, "model" if ckpt else "ground_truth")
    images = panels.load_images(Path(cfg.dataset.root), res)
    m = res.maps

    row1 = [titled(camera_ipm_bev(images, res.info["cams"], (-25, 25), (-25, 25), 8.0), "Camera BEV - geometric IPM (25 m)", TILE),
            energy_map(m["camera_raw_energy"], "Camera BEV - learned LSS features", TILE) if "camera_raw_energy" in m
            else blank("Camera LSS BEV: needs a checkpoint"),
            titled(render_lidar_bev(res.lidar_points, (-50, 50), (-50, 50), 4.0, "LiDAR BEV"), "", TILE),
            titled(render_radar_bev(res.radar_points, (-50, 50), (-50, 50), 4.0, "Radar BEV"), "", TILE)]
    row2 = [energy_map(m["lidar_raw_energy"], "LiDAR BEV features", TILE) if "lidar_raw_energy" in m else blank("LiDAR features: needs a checkpoint"),
            energy_map(m["radar_raw_energy"], "Radar BEV features", TILE) if "radar_raw_energy" in m else blank("Radar features: needs a checkpoint"),
            importance_map(m["lidar_importance"], "LiDAR importance (selection gate)", TILE) if "lidar_importance" in m
            else blank("LiDAR importance: needs a checkpoint"),
            importance_map(m["radar_importance"], "Radar importance (selection gate)", TILE) if "radar_importance" in m
            else blank("Radar importance: needs a checkpoint")]
    fused = energy_map(m["fused_energy"], "Fused BEV (input to detection + planning)", TILE) if "fused_energy" in m \
        else blank("Fused BEV: needs a checkpoint")
    grid = stack_images(row1 + row2, cols=4, pad=6)
    scene_bev = panels.driving_bev(res, pipe, {"show_map": True, "show_lidar": True, "show_radar": True, "show_gt": True,
                                                "show_tracks": True, "score_min": 0.3})
    scene_bev = cv2.resize(scene_bev, (grid.shape[0], grid.shape[0]))
    fused = cv2.resize(fused, (grid.shape[0] // 2, grid.shape[0] // 2))
    side = np.full((grid.shape[0], grid.shape[0] // 2, 3), 16, np.uint8)
    side[: fused.shape[0]] = fused
    note = (f"checkpoint: {ckpt.parent.parent.name}/{ckpt.parent.name}, epoch {pipe.checkpoint_meta.get('epoch')}"
            if ckpt else "no checkpoint: ground-truth objects shown")
    for i, text in enumerate([f"{args.scene} frame {args.frame}", note, "BEV: x,y in [-50, 50) m, 0.5 m cells (200x200)",
                              "forward = up, left = left, ego at centre"]):
        put_label(side, text, (10, fused.shape[0] + 30 + 24 * i), 0.45, (220, 222, 230), None)
    out = np.concatenate([grid, side, scene_bev], 1)
    out_dir = resolve_path(cfg.runtime.output_dir) / "bev"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.scene}_frame{args.frame:03d}_bev.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(path)


if __name__ == "__main__":
    main()
