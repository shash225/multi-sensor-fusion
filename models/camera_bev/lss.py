"""Camera -> BEV: Lift-Splat-Shoot with a BEVDepth-style camera-aware depth network.

Data flow (B batch, N=6 cameras, D depth bins, h×w = H/16 × W/16 feature cells)::

    images (B, N, 3, H, W)
      └─ ImageBackbone ............................ (B·N, 256, h, w)
          └─ DepthNet (conditioned on K, extrinsics)  depth (B·N, D, h, w) softmax
                                                      context (B·N, C, h, w)
              └─ LIFT: outer product ................ (B, N, D, h, w, C)
                 geometry (frustum -> ref frame) .... (B, N, D, h, w, 3)
                  └─ SPLAT: sum-pool into BEV cells . (B, C, ny, nx)
                      └─ BEV refinement convs ....... (B, C, ny, nx)

Geometry of frustum point ``(u, v, d)`` (augmented-image pixel, metric depth)::

    p_img = post_rot⁻¹ ((u, v, d) − post_tran)          # undo resize / crop / flip / rotate
    p_cam = K⁻¹ (p_img.u · d, p_img.v · d, d)            # back-project with the intrinsics
    p_ref = cam2ref · p_cam                              # extrinsics + ego motion to the LiDAR time
    p_bev = bda · p_ref                                  # BEV augmentation (identity at test time)
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.camera_bev.image_backbone import ImageBackbone
from models.common.blocks import ConvBNReLU, ResBlock
from utils.bev_grid import BEVGrid


class DepthNet(nn.Module):
    """Predicts a categorical depth distribution and context features per image cell.

    ``camera_aware``: an MLP over intrinsics/extrinsics produces SE-style gates for the
    depth and context branches (BEVDepth), so one network serves all six cameras.
    """

    def __init__(self, in_channels: int, mid_channels: int, depth_bins: int, context_channels: int,
                 camera_aware: bool = True, cam_param_dim: int = 16) -> None:
        super().__init__()
        self.reduce = ConvBNReLU(in_channels, mid_channels, 3)
        self.camera_aware = camera_aware
        if camera_aware:
            self.cam_bn = nn.BatchNorm1d(cam_param_dim)
            self.cam_mlp = nn.Sequential(nn.Linear(cam_param_dim, mid_channels), nn.ReLU(inplace=True),
                                         nn.Linear(mid_channels, 2 * mid_channels))
        self.context = nn.Sequential(ResBlock(mid_channels), nn.Conv2d(mid_channels, context_channels, 1))
        self.depth = nn.Sequential(ResBlock(mid_channels), ResBlock(mid_channels, dilation=2), nn.Conv2d(mid_channels, depth_bins, 1))

    def forward(self, x: torch.Tensor, cam_params: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.reduce(x)
        x_ctx, x_dep = x, x
        if self.camera_aware and cam_params is not None:
            gates = torch.sigmoid(self.cam_mlp(self.cam_bn(cam_params.float()))).to(x.dtype)
            g_ctx, g_dep = gates.chunk(2, dim=1)
            x_ctx = x * g_ctx[:, :, None, None]
            x_dep = x * g_dep[:, :, None, None]
        return self.depth(x_dep), self.context(x_ctx)


class CameraBEVEncoder(nn.Module):
    """Lift-Splat camera BEV encoder producing ``(B, C, ny, nx)`` on the shared grid."""

    def __init__(self, cfg: Any, grid: BEVGrid, image_size: tuple[int, int]) -> None:
        super().__init__()
        self.grid = grid
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.stride = int(cfg.feature_stride)
        dmin, dmax, dstep = (float(v) for v in cfg.depth_bins)
        self.dbound = (dmin, dmax, dstep)
        self.num_bins = int(round((dmax - dmin) / dstep))
        self.out_channels = int(cfg.out_channels)
        self.backbone = ImageBackbone(cfg.backbone, bool(cfg.pretrained), int(cfg.neck_channels),
                                      int(cfg.frozen_stages), bool(cfg.norm_eval))
        self.depth_net = DepthNet(int(cfg.neck_channels), int(cfg.neck_channels), self.num_bins, self.out_channels,
                                  bool(cfg.camera_aware_depth))
        self.bev_refine = nn.Sequential(ConvBNReLU(self.out_channels, self.out_channels, 3),
                                        *[ResBlock(self.out_channels) for _ in range(int(cfg.bev_refine_blocks))])
        self.register_buffer("frustum", self._create_frustum(), persistent=False)

    # ------------------------------------------------------------- geometry
    def _create_frustum(self) -> torch.Tensor:
        fh, fw = self.image_size
        h, w = fh // self.stride, fw // self.stride
        dmin, _, dstep = self.dbound
        depths = dmin + (torch.arange(self.num_bins, dtype=torch.float32) + 0.5) * dstep  # bin centres
        xs = (torch.arange(w, dtype=torch.float32) + 0.5) * self.stride - 0.5                # cell-centre pixels
        ys = (torch.arange(h, dtype=torch.float32) + 0.5) * self.stride - 0.5
        d = depths.view(-1, 1, 1).expand(-1, h, w)
        x = xs.view(1, 1, w).expand(self.num_bins, h, w)
        y = ys.view(1, h, 1).expand(self.num_bins, h, w)
        return torch.stack([x, y, d], -1)  # (D, h, w, 3)

    def get_geometry(self, intrinsics: torch.Tensor, post_rots: torch.Tensor, post_trans: torch.Tensor,
                     cam2ref: torch.Tensor, bda: torch.Tensor) -> torch.Tensor:
        """Reference-frame location of every frustum point: ``(B, N, D, h, w, 3)``."""
        b, n = intrinsics.shape[:2]
        pts = self.frustum.view(1, 1, *self.frustum.shape) - post_trans.view(b, n, 1, 1, 1, 3)
        pts = torch.linalg.inv(post_rots).view(b, n, 1, 1, 1, 3, 3).matmul(pts.unsqueeze(-1))
        pts = torch.cat([pts[..., :2, :] * pts[..., 2:3, :], pts[..., 2:3, :]], -2)
        combine = cam2ref[..., :3, :3].matmul(torch.linalg.inv(intrinsics))
        pts = combine.view(b, n, 1, 1, 1, 3, 3).matmul(pts).squeeze(-1) + cam2ref[..., :3, 3].view(b, n, 1, 1, 1, 3)
        pts = bda[:, :3, :3].view(b, 1, 1, 1, 1, 3, 3).matmul(pts.unsqueeze(-1)).squeeze(-1) + bda[:, :3, 3].view(b, 1, 1, 1, 1, 3)
        return pts

    def voxel_pool(self, geom: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Sum-pool lifted features ``(B, N, D, h, w, C)`` into BEV cells ``(B, C, ny, nx)`` (z collapsed)."""
        g = self.grid
        b, c = geom.shape[0], feats.shape[-1]
        geom = geom.reshape(b, -1, 3)
        feats = feats.reshape(b, -1, c).float()
        ix = torch.floor((geom[..., 0] - g.x_range[0]) / g.resolution).long()
        iy = torch.floor((geom[..., 1] - g.y_range[0]) / g.resolution).long()
        z = geom[..., 2]
        keep = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny) & (z >= g.z_range[0]) & (z < g.z_range[1])
        batch_idx = torch.arange(b, device=geom.device).view(b, 1).expand_as(ix)
        flat = (batch_idx * g.ny + iy) * g.nx + ix
        bev = torch.zeros(b * g.ny * g.nx, c, device=feats.device, dtype=torch.float32)
        bev.index_add_(0, flat[keep], feats[keep])
        return bev.view(b, g.ny, g.nx, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def camera_params(intrinsics: torch.Tensor, post_rots: torch.Tensor, cam2ref: torch.Tensor,
                      image_size: tuple[int, int]) -> torch.Tensor:
        """16-d camera descriptor: augmented focal lengths / principal point (normalised) + extrinsics."""
        k_aug = post_rots.matmul(intrinsics)
        h, w = image_size
        intr = torch.stack([k_aug[..., 0, 0] / w, k_aug[..., 1, 1] / h, k_aug[..., 0, 2] / w, k_aug[..., 1, 2] / h], -1)
        extr = cam2ref[..., :3, :4].reshape(*cam2ref.shape[:2], 12)
        return torch.cat([intr, extr], -1).flatten(0, 1)

    # -------------------------------------------------------------- forward
    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor, post_rots: torch.Tensor, post_trans: torch.Tensor,
                cam2ref: torch.Tensor, bda: torch.Tensor, cam_valid: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        b, n = images.shape[:2]
        feats = self.backbone(images.flatten(0, 1))
        cam_params = self.camera_params(intrinsics, post_rots, cam2ref, self.image_size)
        depth_logits, context = self.depth_net(feats, cam_params)
        depth = depth_logits.float().softmax(1)
        if cam_valid is not None:  # a missing / unreadable camera contributes nothing
            depth = depth * cam_valid.flatten().float().view(-1, 1, 1, 1)
        lifted = depth.unsqueeze(1) * context.float().unsqueeze(2)  # (BN, C, D, h, w)
        lifted = lifted.view(b, n, *lifted.shape[1:]).permute(0, 1, 3, 4, 5, 2)  # (B, N, D, h, w, C)
        geom = self.get_geometry(intrinsics.float(), post_rots.float(), post_trans.float(), cam2ref.float(), bda.float())
        bev = self.voxel_pool(geom, lifted)
        bev = self.bev_refine(bev)
        return {"bev": bev, "depth_logits": depth_logits, "depth": depth}

    def depth_loss(self, depth_logits: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
        """BEVDepth loss: BCE between the predicted distribution and the one-hot LiDAR depth bin."""
        dmin, dmax, dstep = self.dbound
        gt = depth_gt.reshape(-1)
        valid = (gt >= dmin) & (gt < dmax)
        if not valid.any():
            return depth_logits.sum() * 0.0
        probs = depth_logits.float().softmax(1).permute(0, 2, 3, 1).reshape(-1, self.num_bins)[valid]
        idx = ((gt[valid] - dmin) / dstep).floor().long().clamp(0, self.num_bins - 1)
        target = F.one_hot(idx, self.num_bins).float()
        with torch.autocast(device_type=probs.device.type, enabled=False):
            loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), target, reduction="none").sum()
        return loss / max(1.0, float(valid.sum()))


def depth_bin_centers(dbound: tuple[float, float, float]) -> list[float]:
    dmin, dmax, dstep = dbound
    return [dmin + (k + 0.5) * dstep for k in range(int(math.floor((dmax - dmin) / dstep)))]
