"""Pre-compute per-sample "infos" from the nuScenes tables.

An *info* is a self-contained dictionary with everything a training / inference
step needs for one key-frame: file paths of all sensors (+ sweeps), calibration and
ego-pose matrices, ground-truth boxes **already expressed in the reference frame**,
ego state, ego future/past trajectory and future agent boxes.

Building infos needs the full table set (slow for trainval); the result is cached as a
pickle, so the training dataset and the dashboard never touch the raw tables again.

Coordinate frames
-----------------
``sensor --(sensor2ego)--> ego(t_sensor) --(ego2global)--> global --(inv ref_ego2global)--> ref``

The reference frame ``ref`` is the ego frame at the LIDAR_TOP key-frame timestamp.
Cameras, radars and LiDAR sweeps are captured at slightly different times; going
through the global frame with each sensor's *own* ego pose compensates for ego motion
between the capture times (this is the time synchronisation).
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from datasets.nuscenes_db import NuScenesDB
from datasets.splits import get_split_scenes
from utils.config import Config, resolve_path
from utils.geometry import invert_transform, quat_to_rotmat, transform_boxes, transform_matrix, wrap_angle, yaw_from_rotmat

logger = logging.getLogger(__name__)

INFO_FORMAT_VERSION = 4

# Official nuScenes category -> detection class mapping (nuscenes.eval.detection.utils).
CATEGORY_TO_DETECTION = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

# Navigation command derived from the logged future (ST-P3 / UniAD convention).
COMMANDS = ("RIGHT", "LEFT", "STRAIGHT")


def _sd_matrices(db: NuScenesDB, sd: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    cs = db.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = db.get("ego_pose", sd["ego_pose_token"])
    return transform_matrix(cs["translation"], cs["rotation"]), transform_matrix(pose["translation"], pose["rotation"]), cs


def _sensor_record(db: NuScenesDB, sd: dict, ref_global2ego: np.ndarray, ref_timestamp: int) -> dict[str, Any]:
    sensor2ego, ego2global, cs = _sd_matrices(db, sd)
    rec = {
        "token": sd["token"],
        "path": sd["filename"],  # relative to dataset root (keeps the cache portable)
        "timestamp": sd["timestamp"],
        "time_lag": (ref_timestamp - sd["timestamp"]) * 1e-6,
        "sensor2ego": sensor2ego,
        "ego2global": ego2global,
        "sensor2ref": ref_global2ego @ ego2global @ sensor2ego,
    }
    if cs.get("camera_intrinsic"):
        rec["intrinsic"] = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
        rec["width"], rec["height"] = sd["width"], sd["height"]
    return rec


def annotation_velocity(db: NuScenesDB, ann: dict, max_time_diff: float = 1.5) -> np.ndarray:
    """Global-frame velocity of an annotation (identical to ``NuScenes.box_velocity``)."""
    has_prev, has_next = ann["prev"] != "", ann["next"] != ""
    if not has_prev and not has_next:
        return np.full(3, np.nan)
    first = db.get("sample_annotation", ann["prev"]) if has_prev else ann
    last = db.get("sample_annotation", ann["next"]) if has_next else ann
    pos_diff = np.asarray(last["translation"]) - np.asarray(first["translation"])
    t_last = 1e-6 * db.get("sample", last["sample_token"])["timestamp"]
    t_first = 1e-6 * db.get("sample", first["sample_token"])["timestamp"]
    time_diff = t_last - t_first
    if has_prev and has_next:
        max_time_diff *= 2
    if time_diff <= 0 or time_diff > max_time_diff:
        return np.full(3, np.nan)
    return pos_diff / time_diff


def _ann_box_global(ann: dict) -> np.ndarray:
    """nuScenes annotation -> ``[x, y, z, l, w, h, yaw]`` in the global frame (size is ``[w, l, h]``)."""
    w, l, h = ann["size"]
    yaw = yaw_from_rotmat(quat_to_rotmat(ann["rotation"]))
    return np.array([*ann["translation"], l, w, h, yaw], dtype=np.float64)


def _boxes_for_sample(db: NuScenesDB, sample: dict, ref_global2ego: np.ndarray, classes: list[str],
                      with_velocity: bool = True) -> dict[str, Any]:
    boxes, labels, names, lidar_pts, radar_pts, attrs, vel_valid, inst, tokens, vis = ([] for _ in range(10))
    for ann_token in sample["anns"]:
        ann = db.get("sample_annotation", ann_token)
        det_name = CATEGORY_TO_DETECTION.get(ann["category_name"])
        if det_name is None or det_name not in classes:
            continue
        box = np.zeros(9)
        box[:7] = _ann_box_global(ann)
        if with_velocity:
            vel = annotation_velocity(db, ann)
            valid = bool(np.all(np.isfinite(vel[:2])))
            box[7:9] = vel[:2] if valid else 0.0
        else:
            valid = False
        boxes.append(box)
        labels.append(classes.index(det_name))
        names.append(det_name)
        lidar_pts.append(ann["num_lidar_pts"])
        radar_pts.append(ann["num_radar_pts"])
        attrs.append(db.get("attribute", ann["attribute_tokens"][0])["name"] if ann["attribute_tokens"] else "")
        vel_valid.append(valid)
        inst.append(ann["instance_token"])
        tokens.append(ann["token"])
        vis.append(int(ann["visibility_token"]) if str(ann["visibility_token"]).isdigit() else 0)
    global_boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
    return {
        "boxes": transform_boxes(global_boxes, ref_global2ego).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "names": names,
        "num_lidar_pts": np.asarray(lidar_pts, dtype=np.int32),
        "num_radar_pts": np.asarray(radar_pts, dtype=np.int32),
        "attributes": attrs,
        "velocity_valid": np.asarray(vel_valid, dtype=bool),
        "instance_tokens": inst,
        "ann_tokens": tokens,
        "visibility": np.asarray(vis, dtype=np.int32),
    }


def _lidar_pose(db: NuScenesDB, sample: dict, lidar_channel: str) -> tuple[dict, np.ndarray]:
    sd = db.get("sample_data", sample["data"][lidar_channel])
    pose = db.get("ego_pose", sd["ego_pose_token"])
    return sd, transform_matrix(pose["translation"], pose["rotation"])


class CanBus:
    """Optional reader for the nuScenes CAN-bus expansion (``<root>/can_bus/<scene>_pose.json``)."""

    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / "can_bus"
        self.available = self.dir.is_dir()
        self._cache: dict[str, Optional[list[dict]]] = {}

    def pose_messages(self, scene_name: str) -> Optional[list[dict]]:
        if not self.available:
            return None
        if scene_name not in self._cache:
            path = self.dir / f"{scene_name}_pose.json"
            self._cache[scene_name] = json.loads(path.read_text()) if path.exists() else None
        return self._cache[scene_name]

    def ego_state(self, scene_name: str, timestamp: int) -> Optional[dict[str, float]]:
        msgs = self.pose_messages(scene_name)
        if not msgs:
            return None
        past = [m for m in msgs if m["utime"] <= timestamp]
        if not past:
            return None
        m = past[-1]
        return {
            "speed": float(np.hypot(m["vel"][0], m["vel"][1])),
            "yaw_rate": float(m["rotation_rate"][2]),
            "accel": float(m["accel"][0]),
            "source": "can_bus",
        }


def _ego_state_from_poses(db: NuScenesDB, lidar_sd: dict, window: float = 0.2, accel_window: float = 0.5) -> dict[str, float]:
    """Causal (backward-difference) ego speed, yaw rate and acceleration from 20 Hz LiDAR poses."""

    def pose_at(sd: dict) -> tuple[np.ndarray, float, float]:
        pose = db.get("ego_pose", sd["ego_pose_token"])
        rot = quat_to_rotmat(pose["rotation"])
        return np.asarray(pose["translation"][:2]), yaw_from_rotmat(rot), sd["timestamp"] * 1e-6

    def velocity(sd: dict) -> Optional[tuple[float, float, float]]:
        p0, yaw0, t0 = pose_at(sd)
        for prev in db.iter_sweeps(sd["prev"]) if sd["prev"] else []:
            p1, yaw1, t1 = pose_at(prev)
            if t0 - t1 >= window:
                dt = t0 - t1
                v_global = (p0 - p1) / dt
                # Speed along the current heading (signed: negative when reversing).
                speed = float(v_global @ np.array([np.cos(yaw0), np.sin(yaw0)]))
                return speed, float(wrap_angle(yaw0 - yaw1) / dt), t0
        return None

    current = velocity(lidar_sd)
    if current is None:  # first sweeps of a log: fall back to a forward difference
        nxt = db.get("sample_data", lidar_sd["next"]) if lidar_sd["next"] else None
        if nxt is None:
            return {"speed": 0.0, "yaw_rate": 0.0, "accel": 0.0, "source": "none"}
        p0, yaw0, t0 = pose_at(lidar_sd)
        p1, yaw1, t1 = pose_at(nxt)
        dt = max(t1 - t0, 1e-3)
        speed = float(((p1 - p0) / dt) @ np.array([np.cos(yaw0), np.sin(yaw0)]))
        return {"speed": speed, "yaw_rate": float(wrap_angle(yaw1 - yaw0) / dt), "accel": 0.0, "source": "ego_pose_fwd"}
    speed, yaw_rate, t0 = current
    accel = 0.0
    for prev in db.iter_sweeps(lidar_sd["prev"]) if lidar_sd["prev"] else []:
        if t0 - prev["timestamp"] * 1e-6 >= accel_window:
            past = velocity(prev)
            if past is not None:
                accel = (speed - past[0]) / (t0 - past[2])
            break
    return {"speed": speed, "yaw_rate": yaw_rate, "accel": float(accel), "source": "ego_pose"}


def driving_command(future_xy: np.ndarray, mask: np.ndarray, threshold: float) -> int:
    """0 = RIGHT, 1 = LEFT, 2 = STRAIGHT from the lateral offset of the last valid future point."""
    if not mask.any():
        return 2
    y = future_xy[np.nonzero(mask)[0][-1], 1]
    if y >= threshold:
        return 1
    if y <= -threshold:
        return 0
    return 2


def build_sample_info(db: NuScenesDB, sample_token: str, cfg: Config, can_bus: Optional[CanBus] = None) -> dict[str, Any]:
    """Build the info dictionary of one key-frame sample."""
    dcfg, pcfg = cfg.dataset, cfg.planning
    classes: list[str] = list(cfg.classes)
    sample = db.get("sample", sample_token)
    scene = db.get("scene", sample["scene_token"])
    log = db.get("log", scene["log_token"])
    map_rec = db.map_for_log(scene["log_token"])

    lidar_sd, ref_ego2global = _lidar_pose(db, sample, dcfg.lidar_channel)
    ref_global2ego = invert_transform(ref_ego2global)
    ref_ts = lidar_sd["timestamp"]

    info: dict[str, Any] = {
        "token": sample_token,
        "timestamp": sample["timestamp"],
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "scene_description": scene["description"],
        "location": log["location"],
        "log_token": log["token"],
        "map_filename": map_rec["filename"] if map_rec else None,
        "prev": sample["prev"],
        "next": sample["next"],
        "ref_ego2global": ref_ego2global,
    }

    # LiDAR key-frame + previous sweeps.
    sweeps = []
    for i, sd in enumerate(db.iter_sweeps(lidar_sd["token"])):
        if i >= int(dcfg.max_cached_lidar_sweeps):
            break
        sweeps.append(_sensor_record(db, sd, ref_global2ego, ref_ts))
    info["lidar"] = sweeps

    # Cameras (key-frame image of each camera).
    info["cams"] = {}
    for cam in dcfg.cameras:
        if cam in sample["data"]:
            info["cams"][cam] = _sensor_record(db, db.get("sample_data", sample["data"][cam]), ref_global2ego, ref_ts)

    # Radars: key-frame sweep + previous sweeps per radar.
    info["radars"] = {}
    for radar in dcfg.radars:
        if radar not in sample["data"]:
            continue
        recs = []
        for i, sd in enumerate(db.iter_sweeps(sample["data"][radar])):
            if i >= int(dcfg.max_cached_radar_sweeps):
                break
            recs.append(_sensor_record(db, sd, ref_global2ego, ref_ts))
        info["radars"][radar] = recs

    # Ground truth in the reference frame.
    info["gt"] = _boxes_for_sample(db, sample, ref_global2ego, classes)
    # Bicycle racks: the official evaluation ignores bicycles / motorcycles parked inside them.
    racks = [_ann_box_global(a) for a in (db.get("sample_annotation", t) for t in sample["anns"])
             if a["category_name"] == "static_object.bicycle_rack"]
    racks_global = np.zeros((len(racks), 9))
    if racks:
        racks_global[:, :7] = np.asarray(racks)
    info["gt"]["bike_racks"] = transform_boxes(racks_global, ref_global2ego).astype(np.float32)

    # Ego state (CAN bus when available, otherwise causal finite differences of ego poses).
    state = can_bus.ego_state(scene["name"], ref_ts) if can_bus is not None else None
    info["ego_state"] = state or _ego_state_from_poses(db, lidar_sd)

    # Ego future / past trajectory and future agent boxes (key-frames, 2 Hz).
    horizon = int(pcfg.horizon)
    fut_xy = np.zeros((horizon, 2), dtype=np.float32)
    fut_yaw = np.zeros(horizon, dtype=np.float32)
    fut_mask = np.zeros(horizon, dtype=bool)
    agent_future: list[dict[str, Any]] = []
    token = sample["next"]
    for k in range(horizon):
        if not token:
            agent_future.append({"boxes": np.zeros((0, 9), np.float32), "labels": np.zeros(0, np.int64), "instance_tokens": []})
            continue
        fsample = db.get("sample", token)
        _, f_ego2global = _lidar_pose(db, fsample, dcfg.lidar_channel)
        rel = ref_global2ego @ f_ego2global
        fut_xy[k] = rel[:2, 3]
        fut_yaw[k] = yaw_from_rotmat(rel[:3, :3])
        fut_mask[k] = True
        fboxes = _boxes_for_sample(db, fsample, ref_global2ego, classes, with_velocity=False)
        agent_future.append({k2: fboxes[k2] for k2 in ("boxes", "labels", "instance_tokens")})
        token = fsample["next"]
    info["ego_future"] = {"xy": fut_xy, "yaw": fut_yaw, "mask": fut_mask}
    info["agent_future"] = agent_future
    info["command"] = driving_command(fut_xy, fut_mask, float(pcfg.command_threshold))

    past = []
    token = sample["prev"]
    for _ in range(int(dcfg.history_frames)):
        if not token:
            break
        psample = db.get("sample", token)
        _, p_ego2global = _lidar_pose(db, psample, dcfg.lidar_channel)
        past.append((ref_global2ego @ p_ego2global)[:2, 3])
        token = psample["prev"]
    info["ego_past"] = np.asarray(past, dtype=np.float32).reshape(-1, 2)

    # Future trajectories of the current GT objects (for visualisation of object motion).
    n_obj = len(info["gt"]["instance_tokens"])
    inst_xy = np.zeros((n_obj, horizon, 2), dtype=np.float32)
    inst_mask = np.zeros((n_obj, horizon), dtype=bool)
    index = {tok: i for i, tok in enumerate(info["gt"]["instance_tokens"])}
    for k, fut in enumerate(agent_future):
        for j, tok in enumerate(fut["instance_tokens"]):
            if tok in index:
                inst_xy[index[tok], k] = fut["boxes"][j, :2]
                inst_mask[index[tok], k] = True
    info["gt"]["future_xy"] = inst_xy
    info["gt"]["future_mask"] = inst_mask
    return info


def build_infos(db: NuScenesDB, scene_names: list[str], cfg: Config) -> list[dict[str, Any]]:
    can_bus = CanBus(db.root)
    infos = []
    for name in scene_names:
        scene = db.scene_by_name(name)
        tokens = db.scene_sample_tokens(scene)
        for idx, token in enumerate(tokens):
            info = build_sample_info(db, token, cfg, can_bus)
            info["frame_idx"] = idx
            info["num_frames"] = len(tokens)
            infos.append(info)
    return infos


def _cache_key(cfg: Config, split: str) -> str:
    relevant = {
        "fmt": INFO_FORMAT_VERSION,
        "version": cfg.dataset.version,
        "split": split,
        "classes": list(cfg.classes),
        "cams": list(cfg.dataset.cameras),
        "radars": list(cfg.dataset.radars),
        "lidar_sweeps": cfg.dataset.max_cached_lidar_sweeps,
        "radar_sweeps": cfg.dataset.max_cached_radar_sweeps,
        "history": cfg.dataset.history_frames,
        "horizon": cfg.planning.horizon,
        "cmd": cfg.planning.command_threshold,
    }
    return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]


def load_infos(cfg: Config, split: str, db: Optional[NuScenesDB] = None, rebuild: bool = False) -> list[dict[str, Any]]:
    """Load cached infos for ``split`` or build (and cache) them from the tables."""
    cache_dir = resolve_path(cfg.dataset.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"infos_{cfg.dataset.version}_{split}_{_cache_key(cfg, split)}.pkl"
    if path.exists() and not rebuild:
        with open(path, "rb") as f:
            infos = pickle.load(f)
        logger.info("Loaded %d cached infos from %s", len(infos), path)
        return infos
    db = db or NuScenesDB(cfg.dataset.root, cfg.dataset.version)
    scenes = get_split_scenes(split, [s["name"] for s in db.scenes])
    logger.info("Building infos for split %s (%d scenes) ...", split, len(scenes))
    infos = build_infos(db, scenes, cfg)
    with open(path, "wb") as f:
        pickle.dump(infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached %d infos to %s", len(infos), path)
    return infos
