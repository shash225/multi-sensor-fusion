"""MultiSensorFusionNet — the complete network.

::

    Camera ─► CameraBEVEncoder (LSS) ─────────────┐ F_cam (B,64,200,200) ── guide ──┐
    LiDAR  ─► LidarEncoder (dynamic pillars) ─────┤ F_lidar (B,64,200,200) ─┐       │
    Radar  ─► RadarEncoder (velocity pillars) ────┤ F_radar (B,64,200,200) ─┴► CrossModalFeatureSelection
                                                  │                             │ F̃_lidar, F̃_radar
                                                  └──────────► Fusion (concat | attention_gated)
                                                                   │ fused BEV (B,128,200,200)
                                                              SECOND-FPN (B,256,100,100)
                                                         ┌─────────┴──────────┐
                                                    CenterHead           PlanningHead
                                                 3D boxes + velocity     ego waypoints (B,T,2)

Total loss (weights in configs/training.yaml)::

    L = λ1 L_detection + λ2 L_trajectory + λ3 L_feature_selection + λ4 L_fusion + λ5 L_depth
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch import nn

from models.camera_bev.lss import CameraBEVEncoder
from models.detection.center_head import CenterHead, aux_heatmap_loss, detection_loss
from models.detection.decoder import decode_detections
from models.feature_selection.losses import feature_selection_loss
from models.feature_selection.selection import CrossModalFeatureSelection
from models.fusion.bev_backbone import SECONDFPN
from models.fusion.registry import build_fusion
from models.lidar_encoder.pillar_encoder import LidarEncoder
from models.radar_encoder.radar_encoder import RadarEncoder
from models.trajectory.losses import trajectory_loss
from models.trajectory.planning_head import PlanningHead
from utils.bev_grid import BEVGrid
from utils.config import Config
from utils.timer import NullTimer, StageTimer

ALL_LOSSES = ("detection", "trajectory", "feature_selection", "fusion", "depth")


class MultiSensorFusionNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        f, m = cfg.fusion, cfg.model
        self.grid = BEVGrid.from_config(cfg.bev)
        self.classes = list(cfg.classes)
        self.use_camera, self.use_lidar, self.use_radar = bool(f.use_camera_bev), bool(f.use_lidar), bool(f.use_radar)
        if not (self.use_camera or self.use_lidar or self.use_radar):
            raise ValueError("At least one modality must be enabled in fusion.use_*")

        channels: dict[str, int] = {}
        if self.use_camera:
            self.camera_bev = CameraBEVEncoder(m.camera_bev, self.grid, tuple(cfg.dataset.camera.image_size))
            channels["camera"] = self.camera_bev.out_channels
        if self.use_lidar:
            self.lidar_encoder = LidarEncoder(m.lidar_encoder, self.grid)
            channels["lidar"] = self.lidar_encoder.out_channels
        if self.use_radar:
            self.radar_encoder = RadarEncoder(m.radar_encoder, self.grid)
            channels["radar"] = self.radar_encoder.out_channels
        self.modalities = list(channels)

        selectable = {k: c for k, c in channels.items() if k in ("lidar", "radar")}
        self.use_feature_selection = bool(f.feature_selection) and len(selectable) > 0
        if self.use_feature_selection:
            guide = channels.get("camera") if (self.use_camera and f.selection.camera_guided) else None
            self.feature_selection = CrossModalFeatureSelection(selectable, guide, f.selection)

        self.fusion_method = str(f.method)
        self.fusion = build_fusion(self.fusion_method, channels, int(f.out_channels))
        self.deep_supervision = bool(f.deep_supervision)
        if self.deep_supervision:
            heads = {k: nn.Conv2d(c, 1, 1) for k, c in channels.items()}
            heads["fused"] = nn.Conv2d(int(f.out_channels), 1, 1)
            self.aux_heads = nn.ModuleDict(heads)
            for h in self.aux_heads.values():
                nn.init.constant_(h.bias, -2.19)

        self.bev_backbone = SECONDFPN(int(f.out_channels), m.bev_backbone)
        hcfg = m.detection_head
        self.head_grid = self.grid.downsample(int(hcfg.out_stride))
        if self.bev_backbone.out_stride != int(hcfg.out_stride):
            raise ValueError("detection_head.out_stride must equal bev_backbone.layer_strides[0]")
        self.det_head = CenterHead(self.bev_backbone.out_channels, len(self.classes), int(hcfg.head_channels))
        self.use_planning_head = bool(m.trajectory_head.enabled)
        if self.use_planning_head:
            self.plan_head = PlanningHead(self.bev_backbone.out_channels, int(m.trajectory_head.hidden),
                                          int(cfg.planning.horizon), float(cfg.planning.dt), int(m.trajectory_head.num_heads))

    # ------------------------------------------------------------------ utils
    def set_grad_checkpoint(self, enabled: bool) -> None:
        if self.use_camera:
            self.camera_bev.backbone.grad_checkpoint = enabled
        self.bev_backbone.grad_checkpoint = enabled

    def module_groups(self) -> dict[str, nn.Module]:
        """Top-level sub-networks by name (used for staged freezing)."""
        return {name: mod for name, mod in self.named_children()}

    def freeze(self, names: Iterable[str]) -> list[str]:
        """Freeze parameters whose name starts with any of ``names`` (e.g. ``camera_bev.backbone``)."""
        prefixes = tuple(n.rstrip(".") + "." for n in names)
        frozen = []
        for pname, p in self.named_parameters():
            if pname.startswith(prefixes):
                p.requires_grad_(False)
                frozen.append(pname)
        return frozen

    # ---------------------------------------------------------------- forward
    def forward(self, batch: dict[str, Any], timer: Optional[StageTimer] = None) -> dict[str, Any]:
        timer = timer or NullTimer()
        bsz = int(batch["batch_size"])
        out: dict[str, Any] = {"raw": {}}
        feats: dict[str, torch.Tensor] = {}
        if self.use_camera:
            with timer.stage("camera_bev"):
                cam = self.camera_bev(batch["images"], batch["intrinsics"], batch["post_rots"], batch["post_trans"],
                                      batch["cam2ref"], batch["bda"], batch.get("cam_valid"))
            feats["camera"] = cam["bev"]
            out["depth_logits"], out["depth"] = cam["depth_logits"], cam["depth"]
        if self.use_lidar:
            with timer.stage("lidar_encoder"):
                lid = self.lidar_encoder(batch["lidar_points"], bsz)
            feats["lidar"] = lid["bev"]
            out["lidar_raster"] = lid["raster"]
        if self.use_radar:
            with timer.stage("radar_encoder"):
                rad = self.radar_encoder(batch["radar_points"], bsz)
            feats["radar"] = rad["bev"]
            out["radar_raster"] = rad["raster"]
        out["raw"] = dict(feats)

        selected = dict(feats)
        if self.use_feature_selection:
            with timer.stage("feature_selection"):
                sel, info = self.feature_selection({k: feats[k] for k in self.feature_selection.modalities},
                                                   feats.get("camera"))
            selected.update(sel)
            out["selection"] = info
        out["selected"] = selected

        with timer.stage("fusion"):
            fused, fusion_info = self.fusion(selected)
        out["fused"], out["fusion_info"] = fused, fusion_info
        if self.deep_supervision and self.training:
            out["aux"] = {k: self.aux_heads[k](selected[k]) for k in self.modalities}
            out["aux"]["fused"] = self.aux_heads["fused"](fused)

        with timer.stage("bev_backbone"):
            bev = self.bev_backbone(fused)
        out["bev_features"] = bev
        with timer.stage("detection_head"):
            out["det"] = self.det_head(bev)
        if self.use_planning_head and "ego_state" in batch:
            with timer.stage("planning_head"):
                out["plan"] = self.plan_head(bev, batch["ego_state"], batch["command"])
        return out

    # ------------------------------------------------------------------ losses
    def compute_losses(self, out: dict[str, Any], batch: dict[str, Any], loss_cfg: Config,
                       active: Iterable[str] = ALL_LOSSES) -> dict[str, torch.Tensor]:
        active = set(active)
        losses: dict[str, torch.Tensor] = {}
        total = out["fused"].new_zeros((), dtype=torch.float32)
        if "detection" in active:
            det = detection_loss(out["det"], batch, loss_cfg.detection, list(self.cfg.model.detection_head.code_weights))
            losses.update(det)
            total = total + float(loss_cfg.lambda_detection) * det["detection"]
        if "trajectory" in active and "plan" in out:
            traj = trajectory_loss(out["plan"]["waypoints"], batch, self.grid, self.cfg.planning.ego, loss_cfg.trajectory,
                                   float(self.cfg.planning.dt))
            losses.update(traj)
            total = total + float(loss_cfg.lambda_trajectory) * traj["trajectory"]
        if "feature_selection" in active and "selection" in out:
            fs = feature_selection_loss(out["selection"], batch["fg_heatmap"], loss_cfg.feature_selection,
                                        self.cfg.fusion.selection.target_keep_ratio)
            losses.update(fs)
            total = total + float(loss_cfg.lambda_feature_selection) * fs["feature_selection"]
        if "fusion" in active and "aux" in out:
            aux = [aux_heatmap_loss(logits, batch["fg_heatmap"]) for logits in out["aux"].values()]
            losses["fusion"] = torch.stack(aux).mean()
            total = total + float(loss_cfg.lambda_fusion) * losses["fusion"]
        if "depth" in active and self.use_camera and "depth_gt" in batch:
            losses["depth"] = self.camera_bev.depth_loss(out["depth_logits"], batch["depth_gt"])
            total = total + float(loss_cfg.lambda_depth) * losses["depth"]
        losses["total"] = total
        return losses

    # ----------------------------------------------------------------- decode
    def decode(self, out: dict[str, Any], score_threshold: Optional[float] = None) -> list[dict[str, torch.Tensor]]:
        h = self.cfg.model.detection_head
        thr = float(h.score_threshold if score_threshold is None else score_threshold)
        return decode_detections(out["det"], self.head_grid, self.classes, thr, int(h.max_detections), dict(h.nms_radius))


def build_model(cfg: Config) -> MultiSensorFusionNet:
    return MultiSensorFusionNet(cfg)
