"""BEV fusion methods.

All methods take ``{modality: (B, C_m, H, W)}`` and return ``(B, C_out, H, W)`` plus an
info dict, so they are interchangeable (see :mod:`models.fusion.registry`).

* :class:`ConcatFusion` — baseline: concatenate -> conv (Experiment C).
* :class:`AttentionGatedFusion` — per-cell softmax attention over modalities (which
  sensor to trust *where*), weighted sum + residual mixing + CBAM refinement
  (Experiments D-F). Combined with feature selection it forms the proposed method.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.common.blocks import CBAM, ConvBNReLU, ResBlock


class ConcatFusion(nn.Module):
    """``concat(F_m) -> 3x3 conv -> residual block``."""

    def __init__(self, in_channels: dict[str, int], out_channels: int) -> None:
        super().__init__()
        self.modalities = list(in_channels)
        self.fuse = nn.Sequential(ConvBNReLU(sum(in_channels.values()), out_channels, 3), ResBlock(out_channels))

    def forward(self, feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.fuse(torch.cat([feats[m] for m in self.modalities], 1)), {}


class AttentionGatedFusion(nn.Module):
    """Gated attention fusion.

    1. Project each modality to a common space: ``P_m = φ_m(F_m)``.
    2. Per-cell modality attention ``a = softmax_m(g([F_1..F_M]))`` ∈ Δ^M.
    3. ``F_w = Σ_m a_m · P_m`` (sensor reliability weighting).
    4. ``F = CBAM(ResBlock(conv1x1([F_w, P_1..P_M])))`` keeps complementary detail.
    """

    def __init__(self, in_channels: dict[str, int], out_channels: int) -> None:
        super().__init__()
        self.modalities = list(in_channels)
        m = len(self.modalities)
        self.proj = nn.ModuleDict({k: ConvBNReLU(c, out_channels, 1) for k, c in in_channels.items()})
        self.attention = nn.Sequential(ConvBNReLU(sum(in_channels.values()), 64, 3), nn.Conv2d(64, m, 1))
        self.mix = nn.Sequential(ConvBNReLU(out_channels * (m + 1), out_channels, 1), ResBlock(out_channels),
                                 CBAM(out_channels))

    def forward(self, feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        projected = [self.proj[k](feats[k]) for k in self.modalities]
        weights = torch.softmax(self.attention(torch.cat([feats[k] for k in self.modalities], 1)).float(), dim=1)
        weighted = sum(weights[:, i:i + 1].to(p.dtype) * p for i, p in enumerate(projected))
        fused = self.mix(torch.cat([weighted] + projected, 1))
        return fused, {"modality_weights": weights, "modality_order": self.modalities}
