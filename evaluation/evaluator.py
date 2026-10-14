"""Run a trained model over a split and compute every metric family.

* detection: nuScenes mAP / mATE / mASE / mAOE / mAVE / mAAE / NDS
* planning: ADE / FDE / L2@1-3 s / collision rate / lateral & longitudinal error / smoothness
  (learned planning head and, optionally, the classical baseline fed with the model's detections)
* feature selection: selected-feature ratios and mean LiDAR / radar importance
* latency: per-stage and total network time (CUDA-synchronised)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.detection_metrics import DetectionEvaluator
from evaluation.trajectory_metrics import aggregate, sample_planning_metrics
from models.fusion_model import ALL_LOSSES, MultiSensorFusionNet
from planning.trajectory_planner import BaselinePlanner
from utils.config import Config
from utils.device import move_to_device
from utils.timer import StageTimer

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(model: MultiSensorFusionNet, loader: DataLoader, cfg: Config, device: torch.device,
                   use_amp: bool = False, compute_losses: bool = True, run_baseline_planner: bool = False,
                   max_batches: Optional[int] = None) -> dict[str, Any]:
    model.eval()
    infos = loader.dataset.infos
    classes = list(cfg.classes)
    collision_labels = [classes.index(c) for c in cfg.planning.collision_classes]
    dt = float(cfg.planning.dt)
    det_eval = DetectionEvaluator(classes)
    learned_plan, baseline_plan = [], []
    fs_stats: dict[str, list[float]] = defaultdict(list)
    loss_sums: dict[str, float] = defaultdict(float)
    stage_ms: dict[str, list[float]] = defaultdict(list)
    fusion_w: dict[str, list[float]] = defaultdict(list)
    weather_stats: dict[str, list[float]] = defaultdict(list)
    baseline = BaselinePlanner(cfg) if run_baseline_planner else None
    n_batches = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = move_to_device(batch, device)
        timer = StageTimer()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch, timer=timer)
            if compute_losses and "heatmap" in batch:
                for k, v in model.compute_losses(out, batch, cfg.loss, ALL_LOSSES).items():
                    loss_sums[k] += float(v)
        for k, v in timer.as_dict().items():
            stage_ms[k].append(v)
        stage_ms["total"].append(timer.total_ms)
        dets = model.decode(out, score_threshold=float(cfg.eval.score_threshold))
        if "selection" in out:
            for k, v in model.feature_selection.selection_stats(out["selection"]).items():
                fs_stats[k].append(v)
        weights = (out.get("fusion_info") or {}).get("modality_weights")
        if weights is not None:  # attention fusion: mean per-cell trust of each sensor
            for i, m in enumerate(out["fusion_info"]["modality_order"]):
                fusion_w[m].append(float(weights[:, i].float().mean()))
        for b, meta in enumerate(batch["meta"]):
            for k, v in (meta.get("weather") or {}).items():  # synthetic-weather sensor statistics
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    weather_stats[k].append(float(v))
            info = infos[meta["index"]]
            gt = info["gt"]
            d = {k: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for k, v in dets[b].items()}
            det_eval.add_sample(info["token"], d["boxes"], d["scores"], d["labels"], gt["boxes"], gt["labels"],
                                gt["attributes"], gt["num_lidar_pts"] + gt["num_radar_pts"], gt["velocity_valid"],
                                bike_racks=gt.get("bike_racks"))
            fut = info["ego_future"]
            if not fut["mask"].all():
                continue
            if "plan" in out:
                pred = out["plan"]["waypoints"][b].float().cpu().numpy()
                learned_plan.append(sample_planning_metrics(pred, fut["xy"], fut["mask"], dt, info["agent_future"],
                                                            collision_labels, cfg.planning.ego))
            if baseline is not None:
                es = info["ego_state"]
                res = baseline.plan(np.array([es["speed"], es["yaw_rate"], es["accel"]]), info["command"], d)
                baseline_plan.append(sample_planning_metrics(res.waypoints, fut["xy"], fut["mask"], dt, info["agent_future"],
                                                             collision_labels, cfg.planning.ego))
        n_batches += 1
    result: dict[str, Any] = {"detection": det_eval.evaluate()}
    if learned_plan:
        result["planning_learned"] = aggregate(learned_plan)
    if baseline_plan:
        result["planning_baseline"] = aggregate(baseline_plan)
    if fs_stats:
        result["feature_selection"] = {k: float(np.mean(v)) for k, v in fs_stats.items()}
    if fusion_w:
        result["fusion_weights"] = {k: float(np.mean(v)) for k, v in fusion_w.items()}
    if weather_stats:
        result["weather_stats"] = {k: float(np.nanmean(v)) for k, v in weather_stats.items()}
    if loss_sums:
        result["val_losses"] = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    batch_size = max(int(cfg.train.get("eval_batch_size", 1)), 1)
    result["latency_ms"] = {k: float(np.median(v)) / batch_size for k, v in stage_ms.items()}
    return result


def summary_line(res: dict[str, Any]) -> str:
    det = res["detection"]
    parts = [f"NDS {det['NDS']:.4f}", f"mAP {det['mAP']:.4f}", f"mAP(present) {det['mAP_present_classes']:.4f}",
             f"mAVE {det['mAVE']:.3f}"]
    if "planning_learned" in res:
        p = res["planning_learned"]
        parts += [f"ADE {p.get('ade', float('nan')):.3f}", f"FDE {p.get('fde', float('nan')):.3f}",
                  f"col {p.get('collision_rate', float('nan')):.3f}"]
    parts.append(f"{res['latency_ms'].get('total', 0):.1f} ms")
    return " | ".join(parts)
