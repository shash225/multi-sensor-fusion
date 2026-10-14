"""PyTorch dataset for multi-sensor fusion, detection and planning on nuScenes.

One item = one key-frame, with every quantity in the reference frame *after* the BEV
augmentation matrix ``bda`` (which is also handed to the camera encoder so the lifted
camera features land in the same augmented frame):

=====================  ==========================  ======================================
key                    shape                       meaning
=====================  ==========================  ======================================
images                 (N_cam, 3, H, W)            normalised images
intrinsics             (N_cam, 3, 3)               K of the original image
post_rots/post_trans   (N_cam, 3, 3) / (N_cam, 3)  image augmentation
cam2ref                (N_cam, 4, 4)               camera -> reference ego frame
cam_valid              (N_cam,)                    image successfully loaded
depth_gt               (N_cam, H/16, W/16)         LiDAR depth for depth supervision
bda                    (4, 4)                      BEV augmentation matrix
lidar_points           (P, 5)                      x y z intensity time_lag
radar_points           (Q, 10)                     see RADAR_POINT_FEATURES
heatmap ... reg_target CenterPoint targets          see models/detection/targets.py
fg_heatmap             (1, ny, nx)                 class-agnostic, BEV resolution
gt_boxes / gt_labels   (M, 9) / (M,)               filtered GT (for evaluation)
ego_future(_mask)      (T, 2) / (T,)               logged ego trajectory
future_occ             (T, ny, nx) uint8           GT agent occupancy at future steps
ego_state              (3,)                        speed, yaw-rate, acceleration
command                ()                          0 right / 1 left / 2 straight
=====================  ==========================  ======================================
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.camera_loader import CameraLoader, lidar_depth_maps
from datasets.info_builder import driving_command, load_infos
from datasets.lidar_loader import LidarLoader
from datasets.radar_loader import RadarLoader
from datasets.weather import SyntheticWeather
from models.detection.targets import build_center_targets, foreground_heatmap
from utils.bev_grid import BEVGrid
from utils.config import Config
from utils.geometry import augment_boxes_bev, bev_rotation_matrix
from utils.raster import rasterize_boxes

logger = logging.getLogger(__name__)

POINT_KEYS = ("lidar_points", "radar_points")
LIST_KEYS = ("gt_boxes", "gt_labels", "meta", "raw_lidar_points", "raw_radar_points")


def enabled_modalities(cfg: Config) -> dict[str, bool]:
    f = cfg.fusion
    return {"camera": bool(f.use_camera_bev), "lidar": bool(f.use_lidar), "radar": bool(f.use_radar)}


class NuScenesFusionDataset(Dataset):
    """Multi-modal nuScenes key-frame dataset.

    Args:
        cfg: full configuration.
        split: ``mini_train`` / ``mini_val`` / ``train`` / ``val``.
        train: enables augmentation.
        infos: pre-loaded infos (otherwise loaded from the cache / built).
        load_all_sensors: always return raw LiDAR and radar points (dashboard visualisation),
            even for modalities the model does not use.
        with_targets: build detection / planning targets.
    """

    def __init__(self, cfg: Config, split: str, train: bool = False, infos: Optional[list[dict]] = None,
                 load_all_sensors: bool = False, with_targets: bool = True) -> None:
        self.cfg = cfg
        self.split = split
        self.train = train
        self.infos = infos if infos is not None else load_infos(cfg, split)
        self.load_all_sensors = load_all_sensors
        self.with_targets = with_targets
        self.modalities = enabled_modalities(cfg)
        self.classes = list(cfg.classes)
        self.grid = BEVGrid.from_config(cfg.bev)
        self.head_grid = self.grid.downsample(int(cfg.model.detection_head.out_stride))
        dcfg = cfg.dataset
        root = Path(dcfg.root)
        self.image_size = tuple(int(v) for v in dcfg.camera.image_size)
        self.feature_stride = int(cfg.model.camera_bev.feature_stride)
        self.camera_loader = CameraLoader(root, dcfg.cameras, self.image_size, dcfg.camera)
        self.lidar_loader = LidarLoader(root, self.grid, int(dcfg.lidar.sweeps), dcfg.lidar.max_points,
                                        float(dcfg.lidar.remove_close_radius))
        f = dcfg.radar.filters
        self.radar_loader = RadarLoader(root, self.grid, int(dcfg.radar.sweeps), f.invalid_states, f.dynprop_states,
                                        f.ambig_states)
        self.collision_labels = np.array([self.classes.index(c) for c in cfg.planning.collision_classes
                                          if c in self.classes])
        max_samples = cfg.train.get("max_train_samples") if train else cfg.train.get("max_val_samples")
        if max_samples:
            self.infos = self.infos[: int(max_samples)]
        # Synthetic adverse weather (evaluation only, set by evaluation/adverse_weather.py; None = unchanged data).
        wspec = cfg.get_path("weather.synthetic")
        self.weather = SyntheticWeather(wspec) if wspec else None

    def __len__(self) -> int:
        return len(self.infos)

    # ----------------------------------------------------------------- helpers
    def _sample_bda(self, rng: np.random.Generator) -> np.ndarray:
        aug = self.cfg.dataset.augmentation
        if not (self.train and aug.enabled):
            return np.eye(4)
        rot = np.deg2rad(rng.uniform(*aug.rot_range_deg))
        scale = rng.uniform(*aug.scale_range)
        return bev_rotation_matrix(rot, scale, flip_x=rng.random() < aug.flip_x_prob, flip_y=rng.random() < aug.flip_y_prob)

    @staticmethod
    def _apply_bda_points(points: np.ndarray, bda: np.ndarray, vel_cols: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
        if points.shape[0] == 0 or np.allclose(bda, np.eye(4)):
            return points
        out = points.copy()
        out[:, :3] = points[:, :3] @ bda[:3, :3].T + bda[:3, 3]
        for a, b in vel_cols:  # velocity vectors rotate/reflect/scale with the world
            out[:, [a, b]] = points[:, [a, b]] @ bda[:2, :2].T
        return out

    def _gt_mask(self, gt: dict[str, Any]) -> np.ndarray:
        boxes = gt["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        pts = gt["num_lidar_pts"] + gt["num_radar_pts"]
        return (pts >= int(self.cfg.dataset.gt_filter.min_points)) & self.grid.in_range(boxes[:, 0], boxes[:, 1])

    # ------------------------------------------------------------------- item
    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = self.infos[idx]
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1) if self.train else idx)
        bda = self._sample_bda(rng)
        mods = self.modalities
        item: dict[str, Any] = {"bda": bda.astype(np.float32)}

        need_depth = mods["camera"] and self.cfg.dataset.camera.depth_gt and self.with_targets
        weather_stats: dict[str, Any] = {}
        lidar = None
        if mods["lidar"] or self.load_all_sensors:
            lidar = self.lidar_loader.load(info, rng=rng)
            if self.weather is not None:
                lidar, s = self.weather.lidar(lidar, np.asarray(info["lidar"][0]["sensor2ref"])[:3, 3], idx)
                weather_stats.update(s)
        if mods["camera"]:
            cams = self.camera_loader.load(info, self.train, rng)
            key_pts = self.lidar_loader.load_keyframe(info) if (need_depth or self.weather is not None) else None
            if self.weather is not None:  # clean key-frame LiDAR only provides the scene geometry for the fog depth
                cams["images"], s = self.weather.camera(cams, key_pts, self.image_size, idx)
                weather_stats.update(s)
            item.update(cams)
            if need_depth:
                dbound = self.cfg.model.camera_bev.depth_bins
                item["depth_gt"] = lidar_depth_maps(key_pts, cams, self.image_size, self.feature_stride,
                                                    (float(dbound[0]), float(dbound[1])))
        if lidar is not None:
            lidar = self._apply_bda_points(lidar, bda)
            if self.load_all_sensors:
                item["raw_lidar_points"] = lidar
            if mods["lidar"]:
                item["lidar_points"] = lidar
        if mods["radar"] or self.load_all_sensors:
            radar = self.radar_loader.load(info)
            if self.weather is not None:
                radar, s = self.weather.radar(radar)
                weather_stats.update(s)
            radar = self._apply_bda_points(radar, bda, vel_cols=((4, 5), (6, 7)))
            if self.load_all_sensors:
                item["raw_radar_points"] = radar
            if mods["radar"]:
                item["radar_points"] = radar

        # Simulated sensor failure (training robustness).
        if self.train:
            drop = self.cfg.dataset.augmentation.modality_dropout
            if mods["lidar"] and rng.random() < float(drop.lidar):
                item["lidar_points"] = item["lidar_points"][:0]
            if mods["radar"] and rng.random() < float(drop.radar):
                item["radar_points"] = item["radar_points"][:0]
            if mods["camera"] and rng.random() < float(drop.camera):
                item["cam_valid"] = np.zeros_like(item["cam_valid"])

        gt = info["gt"]
        keep = self._gt_mask(gt)
        boxes = augment_boxes_bev(gt["boxes"][keep].astype(np.float64), bda).astype(np.float32)
        labels = gt["labels"][keep]
        item["gt_boxes"], item["gt_labels"] = boxes, labels
        mirrored = np.linalg.det(bda[:2, :2]) < 0

        # Planning inputs / targets.
        fut = info["ego_future"]
        fut_xy = (fut["xy"].astype(np.float64) @ bda[:2, :2].T + bda[:2, 3]).astype(np.float32)
        item["ego_future"] = fut_xy
        item["ego_future_mask"] = fut["mask"].astype(np.float32)
        item["command"] = np.int64(driving_command(fut_xy, fut["mask"], float(self.cfg.planning.command_threshold)))
        es = info["ego_state"]
        item["ego_state"] = np.array([es["speed"], -es["yaw_rate"] if mirrored else es["yaw_rate"], es["accel"]],
                                     dtype=np.float32)

        if self.with_targets:
            hcfg = self.cfg.model.detection_head
            targets = build_center_targets(boxes, labels, gt["velocity_valid"][keep], self.head_grid, len(self.classes),
                                           int(hcfg.max_objects), int(hcfg.min_radius), float(hcfg.gaussian_overlap))
            item.update(targets)
            item["fg_heatmap"] = foreground_heatmap(boxes, self.grid, int(hcfg.min_radius), float(hcfg.gaussian_overlap))
            occ = np.zeros((len(info["agent_future"]),) + self.grid.shape, dtype=np.uint8)
            for t, agents in enumerate(info["agent_future"]):
                sel = np.isin(agents["labels"], self.collision_labels)
                if sel.any():
                    rasterize_boxes(augment_boxes_bev(agents["boxes"][sel].astype(np.float64), bda), self.grid, out=occ[t])
            item["future_occ"] = occ

        item["meta"] = {
            "index": idx,
            "token": info["token"],
            "scene_name": info["scene_name"],
            "frame_idx": info.get("frame_idx", 0),
            "timestamp": info["timestamp"],
            "mirrored": bool(mirrored),
        }
        if self.weather is not None:
            item["meta"]["weather"] = {"condition": self.weather.name, "synthetic": True, **weather_stats}
        return item


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size arrays, concatenate point clouds with a leading batch-index column."""
    batch: dict[str, Any] = {}
    keys = set().union(*(s.keys() for s in samples))
    for key in keys:
        values = [s.get(key) for s in samples]
        if key in POINT_KEYS:
            width = max((v.shape[1] for v in values if v is not None), default=1)
            chunks = []
            for b, pts in enumerate(values):
                pts = pts if pts is not None else np.zeros((0, width), np.float32)
                chunks.append(np.concatenate([np.full((pts.shape[0], 1), b, np.float32), pts], 1))
            batch[key] = torch.from_numpy(np.concatenate(chunks, 0))
        elif key in LIST_KEYS:
            batch[key] = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values]
        else:
            batch[key] = torch.from_numpy(np.stack(values, 0))
    batch["batch_size"] = len(samples)
    return batch
