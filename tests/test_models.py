"""Network shape / gradient tests for every fusion configuration (random tensors, CPU, tiny grid)."""
from __future__ import annotations

import pytest
import torch

from models.feature_selection.selection import CrossModalFeatureSelection
from models.fusion_model import ALL_LOSSES, build_model
from utils.config import Config, list_experiments, load_config

SMALL = ["bev.x_range=[-16,16]", "bev.y_range=[-16,16]", "dataset.camera.image_size=[64,176]",
         "model.camera_bev.pretrained=false", "model.camera_bev.depth_bins=[1.0,33.0,2.0]", "runtime.device=cpu"]


def _batch(cfg: Config, b: int = 2) -> dict:
    h, w = cfg.dataset.camera.image_size
    ny = nx = 64
    torch.manual_seed(0)
    lidar = torch.cat([torch.randint(0, b, (3000, 1)).float(), torch.rand(3000, 2) * 32 - 16, torch.rand(3000, 1) * 3 - 1,
                       torch.rand(3000, 1) * 100, torch.rand(3000, 1) * 0.5], 1)
    radar = torch.cat([torch.randint(0, b, (200, 1)).float(), torch.rand(200, 2) * 32 - 16, torch.randn(200, 8)], 1)
    k = torch.tensor([[400.0, 0, w / 2], [0, 400.0, h / 2], [0, 0, 1]])
    cam2ref = torch.eye(4).repeat(b, 6, 1, 1)
    cam2ref[..., :3, :3] = torch.tensor([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # camera looks along +x
    hm = torch.zeros(b, 10, ny // 2, nx // 2)
    hm[:, 0, 10, 10] = 1.0
    return {
        "batch_size": b, "images": torch.randn(b, 6, 3, h, w), "intrinsics": k.repeat(b, 6, 1, 1),
        "post_rots": torch.eye(3).repeat(b, 6, 1, 1), "post_trans": torch.zeros(b, 6, 3), "cam2ref": cam2ref,
        "bda": torch.eye(4).repeat(b, 1, 1), "cam_valid": torch.ones(b, 6, dtype=torch.bool),
        "depth_gt": torch.rand(b, 6, h // 16, w // 16) * 30, "lidar_points": lidar, "radar_points": radar,
        "heatmap": hm, "ind": torch.full((b, 500), 10 * 32 + 10), "mask": torch.cat([torch.ones(b, 1), torch.zeros(b, 499)], 1),
        "vel_mask": torch.cat([torch.ones(b, 1), torch.zeros(b, 499)], 1), "reg_target": torch.zeros(b, 500, 10),
        "fg_heatmap": torch.zeros(b, 1, ny, nx), "ego_future": torch.cumsum(torch.ones(b, 6, 2) * torch.tensor([2.0, 0.0]), 1),
        "ego_future_mask": torch.ones(b, 6), "future_occ": torch.zeros(b, 6, ny, nx, dtype=torch.uint8),
        "ego_state": torch.tensor([[4.0, 0.0, 0.0]] * b), "command": torch.full((b,), 2),
    }


@pytest.mark.parametrize("experiment", list(list_experiments()))
def test_forward_backward_every_experiment(experiment: str) -> None:
    cfg = load_config(experiment=experiment, overrides=SMALL, use_local=False)
    model = build_model(cfg).train()
    batch = _batch(cfg)
    out = model(batch)
    assert out["fused"].shape == (2, int(cfg.fusion.out_channels), 64, 64)
    assert out["det"]["heatmap"].shape == (2, 10, 32, 32)
    assert out["plan"]["waypoints"].shape == (2, 6, 2)
    losses = model.compute_losses(out, batch, cfg.loss, ALL_LOSSES)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.eval()
    dets = model.decode(model(batch), score_threshold=0.0)
    assert dets[0]["boxes"].shape[1] == 9


def test_feature_selection_gates_are_bounded_and_hard_mode_is_binary() -> None:
    cfg = load_config(use_local=False)
    sel_cfg = Config({**cfg.fusion.selection, "gate_mode": "hard_concrete"})
    fs = CrossModalFeatureSelection({"lidar": 16, "radar": 8}, 12, sel_cfg).eval()
    feats = {"lidar": torch.randn(2, 16, 20, 20), "radar": torch.randn(2, 8, 20, 20)}
    selected, info = fs(feats, torch.randn(2, 12, 20, 20))
    for m in feats:
        g = info["spatial_gates"][m]
        assert g.shape == (2, 1, 20, 20)
        assert set(torch.unique(g).tolist()) <= {0.0, 1.0}
        assert selected[m].shape == feats[m].shape
    stats = fs.selection_stats(info)
    assert 0.0 <= stats["lidar_effective_selected_ratio"] <= 1.0
