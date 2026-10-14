"""LiDAR encoder: dynamic PointPillars -> BEV feature map.

    lidar_points (P, 1+5) [batch, x, y, z, intensity, time_lag]   (reference frame)
      ├─ DynamicPillarVFE (PointNet per 0.5 m pillar) ........ (B, 64, ny, nx)
      ├─ explicit raster: density, max/mean height, intensity  (B, 4, ny, nx)
      └─ concat -> ConvBNReLU -> ResBlocks ................... (B, C-4, ny, nx)
    output = concat(learned, raster) ......................... (B, C, ny, nx)

Compute: one linear layer per point (~150k-250k points) + ~3 conv layers at 200x200.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.common.blocks import ConvBNReLU, ResBlock
from models.common.pillars import DynamicPillarVFE, PillarIndexer, bev_raster_stats
from utils.bev_grid import BEVGrid

LIDAR_RASTER_NAMES = ("lidar_density", "lidar_max_height", "lidar_mean_height", "lidar_intensity")


class LidarEncoder(nn.Module):
    in_features = 5

    def __init__(self, cfg: Any, grid: BEVGrid) -> None:
        super().__init__()
        self.grid = grid
        self.use_raster = bool(cfg.raster_channels)
        n_raster = len(LIDAR_RASTER_NAMES) if self.use_raster else 0
        self.out_channels = int(cfg.out_channels)
        learned = self.out_channels - n_raster
        self.vfe = DynamicPillarVFE(self.in_features, tuple(int(c) for c in cfg.pillar_channels), grid)
        self.conv = nn.Sequential(ConvBNReLU(self.vfe.out_channels + n_raster, learned, 3),
                                  *[ResBlock(learned) for _ in range(int(cfg.num_res_blocks))])
        zmax = max(abs(grid.z_range[0]), abs(grid.z_range[1]))
        self.register_buffer("normalizer", torch.tensor([grid.x_range[1], grid.y_range[1], zmax, 255.0, 0.5]), persistent=False)
        self.channel_names = [f"lidar_learned_{i}" for i in range(learned)] + list(LIDAR_RASTER_NAMES[:n_raster])

    def raster(self, points: torch.Tensor, batch_size: int) -> torch.Tensor:
        stats = bev_raster_stats(points, batch_size, self.grid, value_cols=[2, 3])  # density, mean z, mean intensity
        zmax = torch.zeros_like(stats[:, :1])
        if points.shape[0]:
            idx = PillarIndexer(self.grid)(points.float(), batch_size)
            if idx["flat"].numel():
                vals = torch.zeros(idx["flat"].numel(), device=points.device).scatter_reduce_(
                    0, idx["inverse"], idx["points"][:, 3], reduce="amax", include_self=False)
                flat = idx["flat"]
                b, rem = flat // (self.grid.ny * self.grid.nx), flat % (self.grid.ny * self.grid.nx)
                zmax[b, 0, rem // self.grid.nx, rem % self.grid.nx] = vals
        return torch.cat([stats[:, :1], zmax / 3.0, stats[:, 1:2] / 3.0, stats[:, 2:3] / 100.0], 1)

    def forward(self, points: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
        pillars = self.vfe(points, batch_size, self.normalizer)
        if self.use_raster:
            raster = self.raster(points, batch_size)
            feats = torch.cat([self.conv(torch.cat([pillars, raster], 1)), raster], 1)
        else:
            raster = pillars.new_zeros(batch_size, 0, *self.grid.shape)
            feats = self.conv(pillars)
        return {"bev": feats, "raster": raster, "pillars": pillars}
