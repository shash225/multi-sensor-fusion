"""Phase 2 - verify sensor synchronisation and calibration on real nuScenes data.

Checks (all numbers are printed and saved to ``outputs/validation/report.json``):

1. **Synchronisation**: timestamp offset of every camera / radar key-frame w.r.t. the
   LIDAR_TOP key-frame.
2. **LiDAR calibration + ego pose**: points of the key-frame sweep that fall inside each
   GT box (after sensor->ref and global->ref transforms) are counted and compared with
   the annotation's ``num_lidar_pts``. Any error in quaternion order, size order
   (``[w, l, h]``), yaw extraction or frame chaining breaks this agreement.
3. **Radar calibration**: radar returns near GT boxes vs ``num_radar_pts``.
4. **Camera augmentation round trip**: pixel -> augmented pixel -> lifted 3D point.
5. **Images**: LiDAR depth + GT boxes projected into all cameras, and a BEV view with
   LiDAR, radar velocities, GT boxes and the logged ego trajectory.

Usage::

    python scripts/validate_sensors.py --samples 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from datasets.camera_loader import CameraLoader, image_aug_matrices  # noqa: E402
from datasets.info_builder import load_infos  # noqa: E402
from datasets.lidar_loader import LidarLoader  # noqa: E402
from datasets.radar_loader import RadarLoader  # noqa: E402
from utils.bev_grid import BEVGrid  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.geometry import points_in_boxes  # noqa: E402
from utils.logging_utils import get_logger, setup_logging  # noqa: E402
from visualization.camera_view import camera_mosaic, draw_boxes_on_image, draw_lidar_on_image, load_image  # noqa: E402
from visualization.common import RADAR_COLOR, BEVCanvas, class_color, colorize, put_label  # noqa: E402

logger = get_logger("validate")


def sync_report(infos: list[dict]) -> dict:
    offsets: dict[str, list[float]] = {}
    for info in infos:
        ref = info["lidar"][0]["timestamp"]
        for cam, rec in info["cams"].items():
            offsets.setdefault(cam, []).append((rec["timestamp"] - ref) / 1000.0)
        for radar, recs in info["radars"].items():
            offsets.setdefault(radar, []).append((recs[0]["timestamp"] - ref) / 1000.0)
    return {k: {"mean_ms": float(np.mean(v)), "max_abs_ms": float(np.max(np.abs(v)))} for k, v in sorted(offsets.items())}


def lidar_box_agreement(infos: list[dict], lidar: LidarLoader) -> dict:
    ours, ref = [], []
    for info in infos:
        rec = info["lidar"][0]
        pts = lidar.load_sweep({**rec, "time_lag": 0.0})
        gt = info["gt"]
        if pts is None or gt["boxes"].shape[0] == 0:
            continue
        counts = points_in_boxes(pts[:, :3].astype(np.float64), gt["boxes"].astype(np.float64)).sum(0)
        ours.extend(counts.tolist())
        ref.extend(gt["num_lidar_pts"].tolist())
    ours_a, ref_a = np.asarray(ours), np.asarray(ref)
    rel = np.abs(ours_a - ref_a) / np.maximum(ref_a, 1)
    return {
        "boxes": int(len(ours_a)),
        "exact_match_rate": float(np.mean(ours_a == ref_a)),
        "within_10pct_rate": float(np.mean(rel <= 0.10)),
        "mean_abs_diff": float(np.mean(np.abs(ours_a - ref_a))),
        "pearson_r": float(np.corrcoef(ours_a, ref_a)[0, 1]),
        "boxes_with_pts_found_rate": float(np.mean((ours_a > 0)[ref_a > 0])),
    }


def radar_box_agreement(infos: list[dict], root: Path, grid: BEVGrid) -> dict:
    loader = RadarLoader(root, grid, num_sweeps=1, invalid_states=None, dynprop_states=None, ambig_states=None)
    ours, ref = [], []
    for info in infos:
        pts = loader.load(info, num_sweeps=1, filter_range=False)
        gt = info["gt"]
        if gt["boxes"].shape[0] == 0:
            continue
        boxes = gt["boxes"].astype(np.float64).copy()
        boxes[:, 3:6] += 1.0  # radar returns scatter around the surface; generous margin
        boxes[:, 5] += 10.0   # radar z is unreliable
        counts = points_in_boxes(pts[:, :3].astype(np.float64), boxes).sum(0) if len(pts) else np.zeros(len(boxes))
        ours.extend(counts.tolist())
        ref.extend(gt["num_radar_pts"].tolist())
    o, r = np.asarray(ours), np.asarray(ref)
    return {"boxes": int(len(o)), "pearson_r": float(np.corrcoef(o, r)[0, 1]),
            "detected_when_annotated_rate": float(np.mean((o > 0)[r > 0])) if (r > 0).any() else None}


def augmentation_roundtrip(cfg, info: dict) -> dict:
    """Project LiDAR points into augmented images and lift them back with the encoder's formula."""
    root = Path(cfg.dataset.root)
    cam_loader = CameraLoader(root, cfg.dataset.cameras, tuple(cfg.dataset.camera.image_size), cfg.dataset.camera)
    lidar = LidarLoader(root, BEVGrid.from_config(cfg.bev), 1)
    pts = lidar.load_keyframe(info)[:, :3].astype(np.float64)
    rng = np.random.default_rng(0)
    errs = []
    for cam in cfg.dataset.cameras:
        rec = info["cams"][cam]
        aug = cam_loader.sample_aug(rec["width"], rec["height"], train=True, rng=rng)
        post_rot, post_tran = image_aug_matrices(aug)
        ref2cam = np.linalg.inv(rec["sensor2ref"])
        pc = pts @ ref2cam[:3, :3].T + ref2cam[:3, 3]
        pc = pc[pc[:, 2] > 1.0]
        uv = (pc @ rec["intrinsic"].T)[:, :2] / pc[:, 2:3]
        uvd_aug = np.concatenate([uv @ post_rot[:2, :2].T + post_tran[:2], pc[:, 2:3]], 1)
        # Inverse (same algebra as models/camera_bev/lss.py::get_geometry).
        p = (uvd_aug - post_tran) @ np.linalg.inv(post_rot).T
        p = np.concatenate([p[:, :2] * p[:, 2:3], p[:, 2:3]], 1)
        back = p @ (rec["sensor2ref"][:3, :3] @ np.linalg.inv(rec["intrinsic"])).T + rec["sensor2ref"][:3, 3]
        fwd = pc @ rec["sensor2ref"][:3, :3].T + rec["sensor2ref"][:3, 3]
        errs.append(float(np.abs(back - fwd).max()))
    return {"max_abs_error_m": max(errs)}


def render_sample(cfg, info: dict, out_dir: Path) -> list[Path]:
    root = Path(cfg.dataset.root)
    grid = BEVGrid.from_config(cfg.bev)
    lidar = LidarLoader(root, grid, int(cfg.dataset.lidar.sweeps))
    f = cfg.dataset.radar.filters
    radar = RadarLoader(root, grid, int(cfg.dataset.radar.sweeps), f.invalid_states, f.dynprop_states, f.ambig_states)
    pts = lidar.load(info)
    key_pts = lidar.load_keyframe(info)
    rpts = radar.load(info)
    gt = info["gt"]
    # Cameras.
    images = {}
    for cam, rec in info["cams"].items():
        img = load_image(root, rec)
        if img is None:
            continue
        img = draw_lidar_on_image(img, key_pts, rec)
        images[cam] = draw_boxes_on_image(img, gt["boxes"], rec, gt["names"])
    mosaic = camera_mosaic(images, width=1800)
    # BEV.
    canvas = BEVCanvas(cfg.bev.x_range, cfg.bev.y_range, ppm=8)
    canvas.draw_range_rings()
    canvas.draw_points(pts[:, :2], colorize(pts[:, 2], "viridis", -2.0, 3.0))
    canvas.draw_boxes(gt["boxes"], [class_color(n) for n in gt["names"]], thickness=2)
    canvas.draw_arrows(gt["boxes"][:, :2], gt["boxes"][:, 7:9], (255, 255, 255), scale=1.0, thickness=2)
    canvas.draw_points(rpts[:, :2], RADAR_COLOR, radius=3)
    canvas.draw_arrows(rpts[:, :2], rpts[:, 4:6], RADAR_COLOR, scale=1.0)
    fut = info["ego_future"]
    canvas.draw_polyline(fut["xy"][fut["mask"]], (255, 255, 255), dashed=True)
    canvas.draw_ego()
    put_label(canvas.img, f"{info['scene_name']} frame {info.get('frame_idx', 0)}  LiDAR {len(pts)} pts | radar {len(rpts)} pts | "
              f"GT {len(gt['boxes'])} | speed {info['ego_state']['speed']:.1f} m/s", (10, 24), 0.6)
    paths = [out_dir / f"{info['scene_name']}_{info.get('frame_idx', 0):03d}_cameras.jpg",
             out_dir / f"{info['scene_name']}_{info.get('frame_idx', 0):03d}_bev.jpg"]
    cv2.imwrite(str(paths[0]), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(paths[1]), cv2.cvtColor(canvas.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default=None, help="split to validate (default: dataset.val_split)")
    parser.add_argument("--samples", type=int, default=60, help="samples used for the numeric checks")
    parser.add_argument("--render", type=int, default=2, help="number of samples rendered to images")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(overrides=args.set)
    split = args.split or cfg.dataset.val_split
    infos = load_infos(cfg, split)
    out_dir = resolve_path(cfg.runtime.output_dir) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = infos[:: max(1, len(infos) // args.samples)][: args.samples]
    grid = BEVGrid.from_config(cfg.bev)
    report = {
        "split": split,
        "samples_checked": len(sel),
        "sync_offsets_vs_lidar": sync_report(sel),
        "lidar_points_in_gt_boxes": lidar_box_agreement(sel, LidarLoader(Path(cfg.dataset.root), grid, 1, None, 0.0)),
        "radar_points_near_gt_boxes": radar_box_agreement(sel, Path(cfg.dataset.root), grid),
        "camera_augmentation_roundtrip": augmentation_roundtrip(cfg, sel[0]),
    }
    report["rendered"] = [str(p) for i in range(min(args.render, len(sel)))
                          for p in render_sample(cfg, sel[i * len(sel) // max(args.render, 1)], out_dir)]
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
