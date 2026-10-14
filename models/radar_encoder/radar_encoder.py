"""Radar encoder: velocity-preserving radar pillars -> BEV feature map.

    radar_points (Q, 1+10) [batch, x, y, z, rcs, vx_c, vy_c, vx, vy, dyn_prop, time_lag]
      ├─ DynamicPillarVFE ............................................ (B, 64, ny, nx)
      ├─ explicit raster: density, vx_comp, vy_comp, rcs, time-lag .... (B, 5, ny, nx)
      └─ concat -> ConvBNReLU -> dilated ResBlocks (d = 1, 2, 4) ....... (B, C-5, ny, nx)
    output = concat(learned, raster) ................................. (B, C, ny, nx)

Radar returns are ~100x sparser than LiDAR, so dilated convolutions spread each return
over a larger receptive field. The motion-compensated velocity components are kept
*verbatim* in the output (raster channels), guaranteeing that velocity information
reaches fusion, detection (velocity regression) and planning.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.common.blocks import ConvBNReLU, ResBlock
from models.common.pillars import DynamicPillarVFE, bev_raster_stats
from utils.bev_grid import BEVGrid

RADAR_RASTER_NAMES = ("radar_density", "radar_vx", "radar_vy", "radar_rcs", "radar_time_lag")


class RadarEncoder(nn.Module):
    in_features = 10

    def __init__(self, cfg: Any, grid: BEVGrid) -> None:
        super().__init__()
        self.grid = grid
        self.use_raster = bool(cfg.raster_channels)
        n_raster = len(RADAR_RASTER_NAMES) if self.use_raster else 0
        self.out_channels = int(cfg.out_channels)
        learned = self.out_channels - n_raster
        self.vfe = DynamicPillarVFE(self.in_features, tuple(int(c) for c in cfg.pillar_channels), grid)
        self.conv = nn.Sequential(ConvBNReLU(self.vfe.out_channels + n_raster, learned, 3),
                                  *[ResBlock(learned, dilation=int(d)) for d in cfg.dilations])
        self.register_buffer("normalizer", torch.tensor([grid.x_range[1], grid.y_range[1], 5.0, 30.0, 10.0, 10.0,
                                                         10.0, 10.0, 7.0, 0.5]), persistent=False)
        self.register_buffer("raster_scale", torch.tensor([1.0, 10.0, 10.0, 30.0, 0.5]).view(1, -1, 1, 1), persistent=False)
        self.channel_names = [f"radar_learned_{i}" for i in range(learned)] + list(RADAR_RASTER_NAMES[:n_raster])

    def raster(self, points: torch.Tensor, batch_size: int) -> torch.Tensor:
        # value columns (0-based, without batch col): vx_comp=4, vy_comp=5, rcs=3, time_lag=9
        stats = bev_raster_stats(points, batch_size, self.grid, value_cols=[4, 5, 3, 9])
        return stats / self.raster_scale

    def forward(self, points: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
        pillars = self.vfe(points, batch_size, self.normalizer)
        if self.use_raster:
            raster = self.raster(points, batch_size)
            feats = torch.cat([self.conv(torch.cat([pillars, raster], 1)), raster], 1)
        else:
            raster = pillars.new_zeros(batch_size, 0, *self.grid.shape)
            feats = self.conv(pillars)
        return {"bev": feats, "raster": raster, "pillars": pillars}
