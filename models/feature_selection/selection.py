"""Camera-Guided Cross-Modal Feature Selection (CG-CMFS) — the research component.

Problem
-------
Plain concatenation hands every LiDAR and radar channel at every BEV cell to the fusion
network, including redundant channels (e.g. LiDAR and radar both encoding "something is
here") and noisy ones (radar multipath ghosts, LiDAR returns from vegetation).

Mechanism
---------
For every selectable modality ``m ∈ {lidar, radar}`` the module predicts

* a **channel gate** ``c_m ∈ (0,1)^C`` — *which features* of the modality are useful for
  this scene (cross-modal squeeze-and-excitation: the gate of LiDAR depends on global
  statistics of LiDAR, radar *and* the camera BEV, so e.g. radar velocity channels can be
  up-weighted when LiDAR is sparse);
* a **spatial gate** ``s_m ∈ (0,1)^{H×W}`` — *where* the modality is informative
  (CBAM-style spatial attention over compressed features of all modalities, with the
  camera BEV as the spatial reference / guide).

Selected features: ``F̃_m = F_m ⊙ c_m ⊙ s_m``.

Why this design (vs. alternatives)
----------------------------------
* Channel + spatial gating (SE/CBAM) is cheap at 200×200 BEV resolution, unlike global
  cross-attention (40k tokens → O(1.6e9) attention entries) which does not fit a 4 GB GPU.
* Gates are *explicit* per-modality numbers in [0, 1] → directly interpretable as
  "feature importance" maps and channel importance bars (explainability requirement).
* ``hard_concrete`` mode turns the gates into stochastic binary masks with a
  straight-through estimator: genuine *selection* (0/1) instead of soft re-weighting.
* Supervision: a coverage loss (objects must stay selected by at least one sensor) and a
  sparsity loss (keep only a target ratio) — see ``losses.py``.

Shapes: inputs ``(B, C_m, H, W)`` per modality, guide ``(B, C_cam, H, W)`` or ``None``;
outputs the same shapes plus gate tensors.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from models.common.blocks import ConvBNReLU


class CrossModalFeatureSelection(nn.Module):
    """Learned channel + spatial feature selection for LiDAR / radar BEV features.

    Args:
        channels: channels per selectable modality, e.g. ``{"lidar": 64, "radar": 64}``.
        guide_channels: channels of the camera BEV guide (``None`` = no guide).
        cfg: ``fusion.selection`` config.
    """

    def __init__(self, channels: dict[str, int], guide_channels: Optional[int], cfg: Any) -> None:
        super().__init__()
        self.modalities = list(channels)
        self.channels = dict(channels)
        self.has_guide = guide_channels is not None
        self.gate_mode = str(cfg.gate_mode)
        self.temperature = float(cfg.temperature)
        self.threshold = float(cfg.threshold)
        init_open = float(cfg.init_open)

        all_channels = dict(channels)
        if self.has_guide:
            all_channels["camera"] = int(guide_channels)
        stat_dim = 2 * sum(all_channels.values())
        hidden = max(stat_dim // int(cfg.channel_reduction), 32)
        self.channel_mlp = nn.Sequential(nn.Linear(stat_dim, hidden), nn.ReLU(inplace=True),
                                         nn.Linear(hidden, sum(channels.values())))

        compress = 8
        self.compress = nn.ModuleDict({m: ConvBNReLU(c, compress, 1) for m, c in all_channels.items()})
        sp_hidden = int(cfg.spatial_hidden)
        k = int(cfg.spatial_kernel)
        self.spatial = nn.Sequential(
            ConvBNReLU((compress + 2) * len(all_channels), sp_hidden, k),
            ConvBNReLU(sp_hidden, sp_hidden, 3),
            nn.Conv2d(sp_hidden, len(self.modalities), 3, padding=1),
        )
        for head in (self.channel_mlp[-1], self.spatial[-1]):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, init_open)

    # ------------------------------------------------------------------ gates
    def _gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "soft":
            return torch.sigmoid(logits)
        if self.gate_mode == "hard_concrete":
            if self.training:
                u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
                soft = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / self.temperature)
                hard = (soft > 0.5).to(soft.dtype)
                return hard + soft - soft.detach()  # straight-through estimator
            return (logits > 0).to(logits.dtype)
        raise ValueError(f"Unknown gate_mode {self.gate_mode!r}")

    def forward(self, feats: dict[str, torch.Tensor], guide: Optional[torch.Tensor] = None
                ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        inputs = {m: feats[m] for m in self.modalities}
        if self.has_guide:
            if guide is None:
                raise ValueError("Camera-guided selection needs the camera BEV guide")
            inputs["camera"] = guide
        # Channel gates from global statistics of all modalities.
        stats = torch.cat([torch.cat([x.float().mean((2, 3)), x.float().amax((2, 3))], 1) for x in inputs.values()], 1)
        ch_logits = self.channel_mlp(stats)
        ch_logits_split = torch.split(ch_logits, [self.channels[m] for m in self.modalities], dim=1)
        # Spatial gates from compressed features + channel-pooled descriptors (CBAM).
        sp_in = torch.cat([torch.cat([self.compress[m](x), x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
                           for m, x in inputs.items()], 1)
        sp_logits = self.spatial(sp_in)

        selected, channel_gates, spatial_gates = {}, {}, {}
        for i, m in enumerate(self.modalities):
            c_gate = self._gate(ch_logits_split[i])            # (B, C_m)
            s_gate = self._gate(sp_logits[:, i:i + 1])          # (B, 1, H, W)
            channel_gates[m], spatial_gates[m] = c_gate, s_gate
            selected[m] = feats[m] * c_gate[:, :, None, None].to(feats[m].dtype) * s_gate.to(feats[m].dtype)
        info = {
            "channel_gates": channel_gates,
            "spatial_gates": spatial_gates,
            "channel_logits": dict(zip(self.modalities, ch_logits_split)),
            "spatial_logits": sp_logits,
        }
        return selected, info

    @torch.no_grad()
    def selection_stats(self, info: dict[str, Any]) -> dict[str, float]:
        """Selected-feature ratios (fraction of gates above threshold) and mean importances."""
        out: dict[str, float] = {}
        for m in self.modalities:
            s, c = info["spatial_gates"][m].float(), info["channel_gates"][m].float()
            out[f"{m}_spatial_selected_ratio"] = float((s > self.threshold).float().mean())
            out[f"{m}_channel_selected_ratio"] = float((c > self.threshold).float().mean())
            out[f"{m}_mean_spatial_importance"] = float(s.mean())
            out[f"{m}_mean_channel_importance"] = float(c.mean())
            out[f"{m}_effective_selected_ratio"] = float(((s > self.threshold).float().mean((1, 2, 3))
                                                          * (c > self.threshold).float().mean(1)).mean())
        return out
