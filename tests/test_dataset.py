"""Real-data checks of the loader, calibration chain and camera geometry (skipped without nuScenes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from datasets.lidar_loader import LidarLoader
from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch
from models.camera_bev.lss import CameraBEVEncoder
from tests.conftest import requires_dataset
from utils.bev_grid import BEVGrid
from utils.geometry import points_in_boxes


@requires_dataset
def test_lidar_points_inside_gt_boxes_match_annotations(cfg, val_infos) -> None:
    """Any error in quaternion order, [w,l,h] size order or frame chaining breaks this agreement."""
    loader = LidarLoader(Path(cfg.dataset.root), BEVGrid.from_config(cfg.bev), 1, None, 0.0)
    ours, ref = [], []
    for info in val_infos[::8]:
        pts = loader.load_sweep({**info["lidar"][0], "time_lag": 0.0})
        counts = points_in_boxes(pts[:, :3].astype(np.float64), info["gt"]["boxes"].astype(np.float64)).sum(0)
        ours += counts.tolist()
        ref += info["gt"]["num_lidar_pts"].tolist()
    ours, ref = np.array(ours), np.array(ref)
    assert np.corrcoef(ours, ref)[0, 1] > 0.99
    assert np.mean(np.abs(ours - ref) <= np.maximum(0.1 * ref, 1)) > 0.9


@requires_dataset
def test_dataset_item_shapes(cfg) -> None:
    ds = NuScenesFusionDataset(cfg, cfg.dataset.val_split, train=True)
    batch = collate_batch([ds[0], ds[1]])
    h, w = cfg.dataset.camera.image_size
    assert batch["images"].shape == (2, 6, 3, h, w)
    assert batch["lidar_points"].shape[1] == 6 and batch["radar_points"].shape[1] == 11
    assert batch["heatmap"].shape == (2, len(cfg.classes), 100, 100)
    assert batch["future_occ"].shape == (2, int(cfg.planning.horizon), 200, 200)
    assert torch.isfinite(batch["lidar_points"]).all()


@requires_dataset
def test_lss_geometry_matches_projection(cfg, val_infos) -> None:
    """Lift a frustum point with the encoder and re-project it with the dataset matrices."""
    ds = NuScenesFusionDataset(cfg, cfg.dataset.val_split, train=True, infos=val_infos[:1])
    item = collate_batch([ds[0]])
    enc = CameraBEVEncoder(cfg.model.camera_bev.__class__({**cfg.model.camera_bev, "pretrained": False}),
                           BEVGrid.from_config(cfg.bev), tuple(cfg.dataset.camera.image_size))
    geom = enc.get_geometry(item["intrinsics"], item["post_rots"], item["post_trans"], item["cam2ref"], item["bda"])
    b, n, d, hh, ww = 0, 1, 10, 3, 7
    p_bev = geom[b, n, d, hh, ww].double().numpy()
    bda = item["bda"][0].double().numpy()
    p_ref = np.linalg.solve(bda[:3, :3], p_bev - bda[:3, 3])
    ref2cam = np.linalg.inv(item["cam2ref"][0, n].double().numpy())
    pc = ref2cam[:3, :3] @ p_ref + ref2cam[:3, 3]
    uv = item["intrinsics"][0, n].double().numpy() @ pc
    uv = uv[:2] / uv[2]
    uv_aug = item["post_rots"][0, n, :2, :2].double().numpy() @ uv + item["post_trans"][0, n, :2].double().numpy()
    expected = enc.frustum[d, hh, ww].double().numpy()
    assert np.allclose(uv_aug, expected[:2], atol=1e-3)
    assert np.isclose(pc[2], expected[2], atol=1e-3)
