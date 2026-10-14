"""Panel renderers for the dashboard. Every image / table comes from a processed :class:`FrameResult`."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from inference.pipeline import FrameResult, FusionPipeline
from utils.geometry import yaw_from_rotmat
from visualization.bev_view import SceneLayers, render_driving_scene
from visualization.camera_view import camera_ipm_bev, camera_mosaic, draw_boxes_on_image, load_image
from visualization.common import heatmap_image, put_label
from visualization.feature_selection import (channel_gate_table, class_importance_summary, contribution, dominance_map,
                                             energy_map, importance_map, modality_weight_map, object_importance, titled)
from visualization.lidar_view import render_lidar_bev
from visualization.radar_view import render_radar_bev
from visualization.trajectory_view import render_trajectory_closeup

TILE = 300


def load_images(root: Path, res: FrameResult) -> dict[str, Optional[np.ndarray]]:
    return {cam: load_image(root, rec) for cam, rec in res.info["cams"].items()}


def object_sets(res: FrameResult) -> tuple[Optional[dict], dict]:
    gt = res.info["gt"]
    keep = (gt["num_lidar_pts"] + gt["num_radar_pts"]) > 0
    gt_set = {"boxes": gt["boxes"][keep], "names": [n for n, k in zip(gt["names"], keep) if k],
              "future_xy": gt["future_xy"][keep], "future_mask": gt["future_mask"][keep]}
    pred = res.detections if res.mode == "model" else None
    return pred, gt_set


def camera_panel(res: FrameResult, images: dict[str, Optional[np.ndarray]], cam: str, show_pred: bool, show_gt: bool,
                 score_min: float) -> tuple[Optional[np.ndarray], np.ndarray]:
    pred, gt = object_sets(res)
    annotated = {}
    for name, img in images.items():
        if img is None:
            annotated[name] = None
            continue
        rec = res.info["cams"][name]
        out = img
        if show_gt or res.mode != "model":
            out = draw_boxes_on_image(out, gt["boxes"], rec, gt["names"], color=(120, 255, 120) if res.mode == "model" else None,
                                      thickness=2, show_labels=res.mode != "model")
        if show_pred and pred is not None and len(pred["boxes"]):
            sel = pred["scores"] >= score_min
            out = draw_boxes_on_image(out, pred["boxes"][sel], rec, [n for n, s in zip(pred["names"], sel) if s],
                                      pred["scores"][sel], thickness=3)
        annotated[name] = out
    return annotated.get(cam), camera_mosaic(annotated, width=1500)


def fusion_tiles(res: FrameResult, images: dict[str, Optional[np.ndarray]], fs_available: bool) -> list[np.ndarray]:
    """Camera BEV | LiDAR BEV | Radar BEV | LiDAR importance | Radar importance | Fused BEV."""
    tiles = []
    m = res.maps
    if "camera_raw_energy" in m:
        tiles.append(energy_map(m["camera_raw_energy"], "Camera BEV (learned LSS features)", TILE))
    else:
        # Flat-ground IPM is only meaningful close to the vehicle; keep +-25 m.
        ipm = camera_ipm_bev(images, res.info["cams"], (-25, 25), (-25, 25), 6.0)
        tiles.append(titled(ipm, "Camera BEV (geometric IPM, 25 m)", TILE))
    tiles.append(titled(render_lidar_bev(res.lidar_points, (-50, 50), (-50, 50), 3.0, "LiDAR BEV"), "", TILE)
                 if "lidar_raw_energy" not in m else energy_map(m["lidar_raw_energy"], "LiDAR BEV features", TILE))
    tiles.append(titled(render_radar_bev(res.radar_points, (-50, 50), (-50, 50), 3.0, "Radar BEV"), "", TILE)
                 if "radar_raw_energy" not in m else energy_map(m["radar_raw_energy"], "Radar BEV features", TILE))
    for mod in ("lidar", "radar"):
        key = f"{mod}_importance"
        if key in m:
            tiles.append(importance_map(m[key], f"{'LiDAR' if mod == 'lidar' else 'Radar'} importance (gate)", TILE))
        else:
            blank = np.full((TILE, TILE, 3), 22, np.uint8)
            msg = "needs trained model" if res.mode != "model" else ("feature selection off" if not fs_available else "sensor not used")
            tiles.append(put_label(blank, f"{mod.upper()} importance: {msg}", (8, TILE // 2), 0.45, (170, 170, 180), None))
    if "fused_energy" in m:
        tiles.append(energy_map(m["fused_energy"], "Fused BEV", TILE))
    else:
        blank = np.full((TILE, TILE, 3), 22, np.uint8)
        tiles.append(put_label(blank, "Fused BEV: needs trained model", (8, TILE // 2), 0.45, (170, 170, 180), None))
    return tiles


def driving_bev(res: FrameResult, pipe: FusionPipeline, opts: dict[str, Any]) -> np.ndarray:
    pred, gt = object_sets(res)
    if pred is not None:  # show only detections above the dashboard's confidence threshold
        keep = pred["scores"] >= float(opts.get("score_min", 0.0))
        pred = {"boxes": pred["boxes"][keep], "names": [n for n, k in zip(pred["names"], keep) if k],
                "scores": pred["scores"][keep],
                "track_ids": pred["track_ids"][keep] if "track_ids" in pred else None,
                "histories": [h for h, k in zip(pred.get("histories", []), keep) if k] if "histories" in pred else None}
    xr, yr, ppm = (-25, 55), (-40, 40), 8.0
    patch = pipe.map_patch(res.info, xr, yr, ppm) if opts.get("show_map") else None
    fut = res.info["ego_future"]
    show_pred_boxes = pred is not None
    objs = pred if pred is not None else ({"boxes": gt["boxes"], "names": gt["names"], "track_ids": res.detections.get("track_ids")
                                           if res.detections else None, "histories": res.detections.get("histories")
                                           if res.detections else None} if res.mode == "ground_truth" else None)
    layers = SceneLayers(
        lidar_points=res.lidar_points, radar_points=res.radar_points, map_patch=patch,
        det_boxes=objs["boxes"] if objs else None, det_names=objs["names"] if objs else [],
        track_ids=objs.get("track_ids") if objs else None, track_histories=objs.get("histories") if objs else None,
        gt_boxes=gt["boxes"] if (show_pred_boxes and opts.get("show_gt")) or res.mode == "dataset" else None,
        gt_names=gt["names"],
        gt_future=gt["future_xy"] if opts.get("show_object_futures") else None, gt_future_mask=gt["future_mask"],
        plan=res.plan.waypoints if res.plan is not None else None,
        plan_collision=bool(res.plan.collision) if res.plan is not None else False,
        ego_future=fut["xy"] if opts.get("show_gt_traj", True) else None, ego_future_mask=fut["mask"],
        ego_past=res.info["ego_past"])
    title = {"model": "AI PREDICTION", "ground_truth": "GROUND TRUTH SIMULATION", "dataset": "DATASET VIEW"}[res.mode]
    if res.mode == "model" and not res.model_trained:
        title = "UNTRAINED NETWORK OUTPUT"
    return render_driving_scene(layers, xr, yr, ppm, show_lidar=opts.get("show_lidar", True),
                                show_radar=opts.get("show_radar", True), show_gt=True, show_pred=True,
                                show_tracks=opts.get("show_tracks", True), title=title, ego_cfg=pipe.cfg.planning.ego)


def detection_table(res: FrameResult, score_min: float) -> pd.DataFrame:
    d = res.detections
    if d is None or not len(d["boxes"]):
        return pd.DataFrame(columns=["Object", "Track", "Confidence", "X (m)", "Y (m)", "Range (m)", "Velocity (m/s)"])
    rows = []
    for i, (b, s, n) in enumerate(zip(d["boxes"], d["scores"], d["names"])):
        if s < score_min:
            continue
        rows.append({"Object": n.replace("_", " ").title(), "Track": int(d["track_ids"][i]) if "track_ids" in d else -1,
                     "Confidence": float(s), "X (m)": float(b[0]), "Y (m)": float(b[1]),
                     "Range (m)": float(np.hypot(b[0], b[1])), "Velocity (m/s)": float(np.hypot(b[7], b[8]))})
    return pd.DataFrame(rows).sort_values("Range (m)").reset_index(drop=True) if rows else pd.DataFrame(rows)


def gt_comparison(res: FrameResult, score_min: float, match_dist: float = 2.0) -> dict[str, int]:
    """Greedy centre-distance matching (same rule as the nuScenes TP threshold, 2 m, same class)."""
    pred, gt = object_sets(res)
    if pred is None:
        return {}
    sel = pred["scores"] >= score_min
    pb, pn = pred["boxes"][sel], [n for n, s in zip(pred["names"], sel) if s]
    order = np.argsort(-pred["scores"][sel])
    taken, tp = set(), 0
    for i in order:
        best, best_d = None, match_dist
        for j, (g, gn) in enumerate(zip(gt["boxes"], gt["names"])):
            if j in taken or gn != pn[i]:
                continue
            dist = float(np.hypot(*(g[:2] - pb[i, :2])))
            if dist < best_d:
                best, best_d = j, dist
        if best is not None:
            taken.add(best)
            tp += 1
    return {"ground_truth": len(gt["boxes"]), "predicted": int(sel.sum()), "true_positive": tp,
            "false_positive": int(sel.sum()) - tp, "missed": len(gt["boxes"]) - tp}


def ego_status(res: FrameResult) -> dict[str, float]:
    pose = res.info["ref_ego2global"]
    es = res.info["ego_state"]
    return {"x": float(pose[0, 3]), "y": float(pose[1, 3]), "yaw_deg": float(np.rad2deg(yaw_from_rotmat(pose[:3, :3]))),
            "speed": float(es["speed"]), "yaw_rate": float(es["yaw_rate"]), "accel": float(es["accel"])}


def trajectory_closeup(res: FrameResult, pipe: FusionPipeline) -> np.ndarray:
    pred, gt = object_sets(res)
    objs = pred if pred is not None else gt
    fut = res.info["ego_future"]
    sel = (objs.get("scores", np.ones(len(objs["boxes"]))) >= 0.3) if pred is not None else np.ones(len(objs["boxes"]), bool)
    return render_trajectory_closeup(res.plan.waypoints if res.plan is not None else None, fut["xy"], fut["mask"],
                                     objs["boxes"][sel], [n for n, s in zip(objs["names"], sel) if s], res.info["ego_past"],
                                     collision=bool(res.plan.collision) if res.plan else False, ego_cfg=pipe.cfg.planning.ego)


def explain_selection(res: FrameResult, pipe: FusionPipeline, score_min: float) -> dict[str, Any]:
    """Raw -> importance -> selected images per modality, channel gates and per-class sensor trust."""
    m = res.maps
    out: dict[str, Any] = {"rows": {}, "channels": {}, "class_trust": [], "dominance": None, "attention": None}
    for mod in ("lidar", "radar"):
        if f"{mod}_importance" not in m:
            continue
        raw, sel = m[f"{mod}_raw_energy"], m[f"{mod}_selected_energy"]
        vmax = float(np.percentile(raw, 99.5)) or 1.0
        name = "LiDAR" if mod == "lidar" else "Radar"
        out["rows"][mod] = [energy_map(raw, f"Raw {name} features", TILE, vmax),
                            importance_map(m[f"{mod}_importance"], f"{name} feature importance", TILE),
                            energy_map(sel, f"Selected {name} features", TILE, vmax)]
        if mod in res.channel_gates:
            cg = res.channel_gates[mod]
            out["channels"][mod] = channel_gate_table(cg["gates"], cg["names"])
    if "lidar_importance" in m and "radar_importance" in m:
        out["dominance"] = dominance_map(contribution(m["lidar_selected_energy"], m["lidar_raw_energy"]),
                                         contribution(m["radar_selected_energy"], m["radar_raw_energy"]), TILE)
    if "modality_weights" in m:
        out["attention"] = modality_weight_map(m["modality_weights"], res.raw_outputs.get("modality_order", []), TILE)
    gates = {mod: m[f"{mod}_importance"] for mod in ("lidar", "radar") if f"{mod}_importance" in m}
    if gates and res.detections is not None:
        d = res.detections
        sel = d["scores"] >= score_min
        rows = object_importance(d["boxes"][sel], [n for n, s in zip(d["names"], sel) if s], gates, pipe.grid)
        out["class_trust"] = class_importance_summary(rows, list(gates))
        out["object_rows"] = rows
    if "detection_heatmap" in m:
        out["heatmap"] = titled(heatmap_image(m["detection_heatmap"], "inferno", vmin=0.0, vmax=1.0, percentile=None),
                                "Detection centre heatmap", TILE)
    return out
