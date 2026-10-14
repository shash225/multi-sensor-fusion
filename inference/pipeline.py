"""Frame-by-frame inference pipeline shared by the dashboard and ``scripts/inference.py``.

Three honest modes (no synthetic data anywhere):

* ``dataset``      - raw nuScenes sensors only (works before any training).
* ``ground_truth`` - nuScenes annotations as "perception", logged ego trajectory; the
                     classical planner can be run on the GT objects (clearly labelled).
* ``model``        - the trained network: camera BEV -> LiDAR/radar encoders -> feature
                     selection -> fusion -> detection -> tracking -> planning. Requires a
                     checkpoint; without one the caller must label outputs as UNTRAINED.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from datasets.info_builder import COMMANDS, load_infos
from datasets.map_prior import MapPrior
from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch
from datasets.splits import MINI_TRAIN, MINI_VAL
from datasets.weather import (SyntheticWeather, corrupt_display_image, load_weather_config,
                              synthetic_spec)
from evaluation.detection_metrics import DetectionEvaluator
from evaluation.trajectory_metrics import sample_planning_metrics
from models.fusion_model import MultiSensorFusionNet, build_model
from planning.tracking import MultiObjectTracker
from planning.trajectory_planner import PlanResult, build_planner
from utils.bev_grid import BEVGrid
from utils.checkpoint import load_checkpoint
from utils.config import Config, resolve_path
from utils.device import amp_enabled, get_device, move_to_device
from utils.timer import StageTimer
from visualization.camera_view import load_image

logger = logging.getLogger(__name__)
MACHINE_KEYS = ("dataset.root", "dataset.version", "dataset.cache_dir", "dataset.num_workers", "runtime.device",
                "runtime.output_dir", "runtime.checkpoint_dir")


@dataclass
class FrameResult:
    index: int
    info: dict[str, Any]
    mode: str
    model_trained: bool
    lidar_points: np.ndarray
    radar_points: np.ndarray
    detections: Optional[dict[str, Any]] = None            # boxes (K,9) scores labels names track_ids histories
    plan: Optional[PlanResult] = None
    plan_source: str = ""
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    channel_gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    selection_stats: dict[str, float] = field(default_factory=dict)
    frame_metrics: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    raw_outputs: dict[str, Any] = field(default_factory=dict)
    weather: Optional[dict[str, Any]] = None                # synthetic weather applied (name, label, sensor stats)

    @property
    def command_name(self) -> str:
        return COMMANDS[int(self.info["command"])]


def split_of_scene(name: str) -> str:
    if name in MINI_TRAIN:
        return "train"
    if name in MINI_VAL:
        return "val"
    try:
        from nuscenes.utils.splits import create_splits_scenes
        s = create_splits_scenes()
        return "train" if name in s["train"] else "val" if name in s["val"] else "test"
    except ImportError:
        return "unknown"


class FusionPipeline:
    """Loads the data (and optionally a trained checkpoint) and processes frames on demand."""

    def __init__(self, base_cfg: Config, checkpoint: Optional[Path] = None, split: str = "all",
                 allow_untrained_model: bool = False) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        if self.checkpoint is not None:
            ckpt = load_checkpoint(self.checkpoint)
            cfg = Config(ckpt["config"])
            for key in MACHINE_KEYS:  # data location / device come from *this* machine, not the training one
                cfg.set_path(key, base_cfg.get_path(key))
            self.checkpoint_meta = {"epoch": ckpt.get("epoch"), "metrics": ckpt.get("metrics", {}),
                                    "experiment": ckpt.get("extra", {}).get("experiment"),
                                    "stage": ckpt.get("extra", {}).get("stage")}
        else:
            cfg, ckpt, self.checkpoint_meta = base_cfg, None, {}
        self.cfg = cfg
        self.device = get_device(cfg.runtime.device)
        self.use_amp = amp_enabled(self.device, bool(cfg.train.amp))
        self.grid = BEVGrid.from_config(cfg.bev)
        self.classes = list(cfg.classes)
        self.infos = load_infos(cfg, split)
        self.dataset = NuScenesFusionDataset(cfg, split, train=False, infos=self.infos, load_all_sensors=True,
                                             with_targets=False)
        self.model: Optional[MultiSensorFusionNet] = None
        self.model_trained = False
        if ckpt is not None or allow_untrained_model:
            self.model = build_model(cfg).to(self.device).eval()
            if ckpt is not None:
                self.model.load_state_dict(ckpt["model"])
                self.model_trained = True
        self.tracker = MultiObjectTracker(self.classes)
        self._last_index: Optional[int] = None
        self._planners: dict[str, Any] = {}
        self.map_prior = MapPrior(Path(cfg.dataset.root), resolve_path(cfg.dataset.cache_dir))
        self.collision_labels = [self.classes.index(c) for c in cfg.planning.collision_classes]
        # Synthetic adverse weather (dashboard option). The dataset is shared between viewers, so the
        # condition is set per request under a lock instead of being a global switch.
        self._weather_lock = threading.Lock()
        self._weather_objs: dict[str, SyntheticWeather] = {}
        self._weather_cfg: Optional[dict[str, Any]] = None

    # --------------------------------------------------------------- weather
    def weather_conditions(self) -> dict[str, dict[str, Any]]:
        if self._weather_cfg is None:
            self._weather_cfg = load_weather_config()
        return dict(self._weather_cfg["conditions"])

    def _weather(self, name: Optional[str]) -> Optional[SyntheticWeather]:
        if not name or name == "clear":
            return None
        if name not in self._weather_objs:
            self.weather_conditions()
            self._weather_objs[name] = SyntheticWeather(synthetic_spec(name, self._weather_cfg))
        return self._weather_objs[name]

    def load_display_images(self, res: "FrameResult") -> dict[str, Optional[np.ndarray]]:
        """Camera images for the UI, with the frame's synthetic weather applied (if any)."""
        root = Path(self.cfg.dataset.root)
        images = {cam: load_image(root, rec) for cam, rec in res.info["cams"].items()}
        weather = self._weather(res.weather["name"]) if res.weather else None
        if weather is None:
            return images
        key = self.dataset.lidar_loader.load_keyframe(res.info)
        out = {}
        for k, (cam, img) in enumerate(images.items()):
            if img is None:
                out[cam] = None
                continue
            img = corrupt_display_image(img, res.info["cams"][cam], key, weather, res.index * 10 + k)
            # burn the label into the pixels so a cropped screenshot can never pass as a real recording
            text = f"SYNTHETIC {res.weather['label']}"
            cv2.rectangle(img, (0, 0), (30 + 19 * len(text), 64), (176, 40, 31), -1)
            cv2.putText(img, text, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            out[cam] = img
        return out

    # ------------------------------------------------------------------ meta
    def scenes(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, info in enumerate(self.infos):
            if not out or out[-1]["name"] != info["scene_name"]:
                out.append({"name": info["scene_name"], "description": info["scene_description"], "start": i, "count": 0,
                            "location": info["location"], "split": split_of_scene(info["scene_name"])})
            out[-1]["count"] += 1
        return out

    def planner(self, name: str):
        if name not in self._planners:
            self._planners[name] = build_planner(self.cfg, name)
        return self._planners[name]

    def map_patch(self, info: dict[str, Any], x_range, y_range, ppm: float) -> Optional[np.ndarray]:
        if not info.get("map_filename"):
            return None
        try:
            return self.map_prior.crop_to_ego(info["map_filename"], info["ref_ego2global"], tuple(x_range), tuple(y_range), ppm)
        except (OSError, MemoryError, ValueError) as exc:
            logger.warning("Map prior unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------- run
    def run(self, index: int, mode: str = "model", planner: str = "baseline", plan_on_gt: bool = True,
            score_threshold: Optional[float] = None, stage_callback: Optional[Callable[[str, float], None]] = None,
            weather: Optional[str] = None) -> FrameResult:
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        info = self.infos[index]
        t0 = time.perf_counter()
        with self._weather_lock:
            self.dataset.weather = self._weather(weather)
            try:
                item = self.dataset[index]
            finally:
                self.dataset.weather = None
        timings["data_loading"] = (time.perf_counter() - t0) * 1000
        if stage_callback:
            stage_callback("data_loading", timings["data_loading"])
        res = FrameResult(index, info, mode, self.model_trained, item.get("raw_lidar_points", np.zeros((0, 5), np.float32)),
                          item.get("raw_radar_points", np.zeros((0, 10), np.float32)))
        if weather and weather != "clear":
            res.weather = {"name": weather, "label": self.weather_conditions()[weather]["label"],
                           "stats": item["meta"].get("weather", {})}
        es = info["ego_state"]
        ego_state = np.array([es["speed"], es["yaw_rate"], es["accel"]], dtype=np.float64)
        sequential = self._last_index is not None and index == self._last_index + 1 and \
            self.infos[self._last_index]["scene_token"] == info["scene_token"]
        if not sequential:
            self.tracker.reset()
        self._last_index = index

        if mode == "model" and self.model is not None:
            self._run_model(res, item, timings, score_threshold, stage_callback)
        elif mode == "ground_truth":
            gt = info["gt"]
            keep = (gt["num_lidar_pts"] + gt["num_radar_pts"]) > 0
            res.detections = {"boxes": gt["boxes"][keep].astype(np.float64), "scores": np.ones(int(keep.sum())),
                              "labels": gt["labels"][keep], "names": [n for n, k in zip(gt["names"], keep) if k]}

        if res.detections is not None:
            t0 = time.perf_counter()
            d = res.detections
            # Only confident detections are tracked (the same threshold the planner uses); others get id 0.
            conf = d["scores"] >= float(self.cfg.planning.occupancy.min_score)
            ids = np.zeros(len(d["scores"]), dtype=np.int64)
            ids[conf] = self.tracker.update(d["boxes"][conf], d["labels"][conf], d["scores"][conf], info["ref_ego2global"],
                                            info["timestamp"])
            d["track_ids"] = ids
            d["histories"] = self.tracker.histories_ref(ids, info["ref_ego2global"])
            timings["tracking"] = (time.perf_counter() - t0) * 1000
            if stage_callback:
                stage_callback("tracking", timings["tracking"])

        # Planning.
        if res.detections is not None and (mode == "model" or plan_on_gt):
            t0 = time.perf_counter()
            name = planner
            learned = res.raw_outputs.get("learned_waypoints")
            if name in ("learned", "hybrid") and learned is None:
                name = "baseline"
            res.plan = self.planner(name).plan(ego_state, int(info["command"]), res.detections,
                                               lidar_points=res.lidar_points, learned_waypoints=learned)
            res.plan_source = res.plan.planner + (" on ground-truth objects" if mode == "ground_truth" else "")
            timings["planning"] = (time.perf_counter() - t0) * 1000
            if stage_callback:
                stage_callback("planning", timings["planning"])
        self._frame_metrics(res)
        timings["total"] = (time.perf_counter() - t_all) * 1000
        res.timings_ms = timings
        return res

    @torch.no_grad()
    def _run_model(self, res: FrameResult, item: dict[str, Any], timings: dict[str, float],
                   score_threshold: Optional[float], stage_callback) -> None:
        model = self.model
        batch = move_to_device(collate_batch([{k: v for k, v in item.items() if not k.startswith("raw_")}]), self.device)
        timer = StageTimer(callback=stage_callback)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            out = model(batch, timer=timer)
        with timer.stage("decode"):
            det = model.decode(out, score_threshold)[0]
        timings.update(timer.as_dict())
        timings["network"] = timer.total_ms
        boxes = det["boxes"].float().cpu().numpy().astype(np.float64)
        labels = det["labels"].cpu().numpy()
        res.detections = {"boxes": boxes, "scores": det["scores"].float().cpu().numpy(), "labels": labels,
                          "names": [self.classes[int(l)] for l in labels]}

        def energy(t: torch.Tensor) -> np.ndarray:
            return t[0].float().pow(2).sum(0).sqrt().cpu().numpy()

        maps = res.maps
        for m, feat in out["raw"].items():
            maps[f"{m}_raw_energy"] = energy(feat)
            maps[f"{m}_selected_energy"] = energy(out["selected"][m])
        maps["fused_energy"] = energy(out["fused"])
        maps["detection_heatmap"] = torch.sigmoid(out["det"]["heatmap"][0].float()).amax(0).cpu().numpy()
        if "modality_weights" in out["fusion_info"]:
            maps["modality_weights"] = out["fusion_info"]["modality_weights"][0].float().cpu().numpy()
            res.raw_outputs["modality_order"] = out["fusion_info"]["modality_order"]
        if "selection" in out:
            sel = out["selection"]
            for m, g in sel["spatial_gates"].items():
                maps[f"{m}_importance"] = g[0, 0].float().cpu().numpy()
            names = {"lidar": getattr(model, "lidar_encoder", None), "radar": getattr(model, "radar_encoder", None)}
            for m, g in sel["channel_gates"].items():
                enc = names[m]
                res.channel_gates[m] = {"names": list(enc.channel_names), "gates": g[0].float().cpu().numpy()}
            res.selection_stats = model.feature_selection.selection_stats(sel)
        if "plan" in out:
            res.raw_outputs["learned_waypoints"] = out["plan"]["waypoints"][0].float().cpu().numpy()
            res.maps["planning_attention"] = out["plan"]["attention"][0].float().cpu().numpy()

    def _frame_metrics(self, res: FrameResult) -> None:
        info = res.info
        gt = info["gt"]
        metrics: dict[str, Any] = {}
        if res.detections is not None and res.mode == "model":
            ev = DetectionEvaluator(self.classes)
            d = res.detections
            ev.add_sample(info["token"], d["boxes"], d["scores"], d["labels"], gt["boxes"], gt["labels"], gt["attributes"],
                          gt["num_lidar_pts"] + gt["num_radar_pts"], gt["velocity_valid"], bike_racks=gt.get("bike_racks"))
            r = ev.evaluate()
            metrics["detection"] = {k: r[k] for k in ("NDS", "mAP", "mAP_present_classes", "mATE", "mAVE")}
        fut = info["ego_future"]
        if res.plan is not None and fut["mask"].any():
            metrics["planning"] = sample_planning_metrics(res.plan.waypoints.astype(np.float64), fut["xy"], fut["mask"],
                                                          float(self.cfg.planning.dt), info["agent_future"],
                                                          self.collision_labels, self.cfg.planning.ego)
        res.frame_metrics = metrics

    def add_to_evaluator(self, evaluator: DetectionEvaluator, res: FrameResult) -> None:
        """Accumulate a processed frame into a running (scene-level) evaluator."""
        if res.detections is None or res.mode != "model":
            return
        gt, d = res.info["gt"], res.detections
        evaluator.add_sample(res.info["token"], d["boxes"], d["scores"], d["labels"], gt["boxes"], gt["labels"],
                             gt["attributes"], gt["num_lidar_pts"] + gt["num_radar_pts"], gt["velocity_valid"],
                             bike_racks=gt.get("bike_racks"))
