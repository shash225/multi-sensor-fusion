"""Dynamic pillar feature encoding (pure PyTorch, no custom CUDA ops).

"Dynamic voxelisation" (Zhou et al., MVF 2019) assigns *every* point to its pillar
without a max-points-per-pillar cap and aggregates with scatter-max. This keeps all
returns of the dense multi-sweep LiDAR and works identically for sparse radar.

Input:  ``points (P, 1 + F)`` with a leading batch-index column (see ``collate_batch``).
Output: BEV canvas ``(B, C, ny, nx)`` on the shared :class:`BEVGrid`.
"""
from __future__ import annotations

import torch
from torch import nn

from utils.bev_grid import BEVGrid


def scatter_mean(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, src.shape[1], device=src.device, dtype=src.dtype).index_add_(0, index, src)
    count = torch.zeros(size, device=src.device, dtype=src.dtype).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp(min=1).unsqueeze(1)


def scatter_max(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Per-group max for non-negative ``src`` (inputs are post-ReLU)."""
    out = torch.zeros(size, src.shape[1], device=src.device, dtype=src.dtype)
    return out.scatter_reduce_(0, index.unsqueeze(1).expand(-1, src.shape[1]), src, reduce="amax", include_self=True)


class PillarIndexer:
    """Computes pillar membership of points on a BEV grid."""

    def __init__(self, grid: BEVGrid) -> None:
        self.grid = grid

    def __call__(self, points: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
        g = self.grid
        b = points[:, 0].long()
        ix = torch.floor((points[:, 1] - g.x_range[0]) / g.resolution).long()
        iy = torch.floor((points[:, 2] - g.y_range[0]) / g.resolution).long()
        keep = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny) & (b >= 0) & (b < batch_size)
        b, ix, iy, pts = b[keep], ix[keep], iy[keep], points[keep]
        flat = (b * g.ny + iy) * g.nx + ix
        unique, inverse = torch.unique(flat, return_inverse=True)
        return {"points": pts, "ix": ix, "iy": iy, "flat": unique, "inverse": inverse}


class DynamicPillarVFE(nn.Module):
    """PointNet over pillars: per-point MLP layers with scatter-max aggregation.

    Point decoration (``in_features + 5`` dims): raw features, offset to the pillar's point
    mean (3), offset to the pillar centre in x/y (2).
    """

    def __init__(self, in_features: int, channels: tuple[int, ...], grid: BEVGrid) -> None:
        super().__init__()
        self.grid = grid
        self.indexer = PillarIndexer(grid)
        self.layers = nn.ModuleList()
        c_in = in_features + 5
        for i, c_out in enumerate(channels):
            last = i == len(channels) - 1
            # Every layer except the last concatenates the pillar-max feature (PFN in PointPillars).
            self.layers.append(nn.Sequential(nn.Linear(c_in, c_out if last else c_out // 2, bias=False),
                                             nn.BatchNorm1d(c_out if last else c_out // 2), nn.ReLU(inplace=True)))
            c_in = c_out
        self.out_channels = channels[-1]

    def forward(self, points: torch.Tensor, batch_size: int, normalizer: torch.Tensor | None = None) -> torch.Tensor:
        """Return the BEV canvas ``(B, C, ny, nx)``. ``normalizer`` rescales raw feature columns."""
        g = self.grid
        canvas = points.new_zeros(batch_size, self.out_channels, g.ny, g.nx, dtype=torch.float32)
        if points.shape[0] == 0:
            return canvas
        idx = self.indexer(points.float(), batch_size)
        pts, inv, n_pillars = idx["points"], idx["inverse"], idx["flat"].numel()
        if n_pillars == 0 or (self.training and pts.shape[0] < 2):  # BatchNorm1d needs >1 sample
            return canvas
        feats = pts[:, 1:]
        xyz = feats[:, :3]
        mean = scatter_mean(xyz, inv, n_pillars)[inv]
        centre_x = (idx["ix"].float() + 0.5) * g.resolution + g.x_range[0]
        centre_y = (idx["iy"].float() + 0.5) * g.resolution + g.y_range[0]
        raw = feats / normalizer if normalizer is not None else feats
        x = torch.cat([raw, xyz - mean, (xyz[:, 0] - centre_x).unsqueeze(1), (xyz[:, 1] - centre_y).unsqueeze(1)], 1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            pooled = scatter_max(x, inv, n_pillars)
            if i < len(self.layers) - 1:
                x = torch.cat([x, pooled[inv]], 1)
        flat = idx["flat"]
        b = flat // (g.ny * g.nx)
        rem = flat % (g.ny * g.nx)
        canvas[b, :, rem // g.nx, rem % g.nx] = pooled.to(canvas.dtype)
        return canvas


def bev_raster_stats(points: torch.Tensor, batch_size: int, grid: BEVGrid, value_cols: list[int],
                     count_scale: float = 3.0) -> torch.Tensor:
    """Hand-crafted BEV raster: ``log(1+count)/count_scale`` plus the mean of selected point columns.

    Returns ``(B, 1 + len(value_cols), ny, nx)``. These explicit channels keep physically
    meaningful quantities (density, height, intensity, radar velocity) visible to the
    feature-selection gates, which makes the learned importances interpretable.
    """
    g = grid
    out = points.new_zeros(batch_size, 1 + len(value_cols), g.ny, g.nx, dtype=torch.float32)
    if points.shape[0] == 0:
        return out
    idx = PillarIndexer(g)(points.float(), batch_size)
    pts, inv, flat = idx["points"], idx["inverse"], idx["flat"]
    if flat.numel() == 0:
        return out
    count = torch.zeros(flat.numel(), device=pts.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
    stats = [torch.log1p(count).unsqueeze(1) / count_scale]
    if value_cols:
        stats.append(scatter_mean(pts[:, [c + 1 for c in value_cols]], inv, flat.numel()))
    vals = torch.cat(stats, 1)
    b = flat // (g.ny * g.nx)
    rem = flat % (g.ny * g.nx)
    out[b, :, rem // g.nx, rem % g.nx] = vals
    return out
