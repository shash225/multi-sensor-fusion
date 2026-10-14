"""Feature-selection losses.

``L_feature_selection = w_cov · L_coverage + w_sp · L_sparsity + w_bin · L_binarization``

* **Coverage** (foreground guidance): at object centres (class-agnostic Gaussian
  heatmap ``y``) at least one sensor must keep its features:
  ``L_cov = Σ y · −log(max_m s_m) / Σ y``. Only the positive term is used, so the loss
  never forces background to zero — that is the job of the sparsity term.
* **Sparsity**: ``Σ_m |mean(s_m) − ρ_s| + |mean(c_m) − ρ_c|`` — keep only a target ratio of
  spatial cells / channels, forcing the network to choose.
* **Binarisation**: ``mean(4 g (1 − g))`` pushes gates towards 0/1 so the selection is
  crisp and interpretable.
"""
from __future__ import annotations

from typing import Any

import torch


def feature_selection_loss(info: dict[str, Any], fg_heatmap: torch.Tensor, cfg_weights: Any,
                           target_keep: Any) -> dict[str, torch.Tensor]:
    spatial = torch.cat([info["spatial_gates"][m].float() for m in info["spatial_gates"]], 1)  # (B, M, H, W)
    combined = spatial.amax(1, keepdim=True).clamp(1e-4, 1.0)
    y = fg_heatmap.float()
    coverage = (y * -torch.log(combined)).sum() / y.sum().clamp(min=1.0)

    rho_s, rho_c = float(target_keep.spatial), float(target_keep.channel)
    sparsity = spatial.new_zeros(())
    gates = []
    for m in info["spatial_gates"]:
        s, c = info["spatial_gates"][m].float(), info["channel_gates"][m].float()
        sparsity = sparsity + (s.mean() - rho_s).abs() + (c.mean() - rho_c).abs()
        gates += [s.flatten(), c.flatten()]
    all_gates = torch.cat(gates)
    binarization = (4.0 * all_gates * (1.0 - all_gates)).mean()
    total = (float(cfg_weights.coverage) * coverage + float(cfg_weights.sparsity) * sparsity
             + float(cfg_weights.binarization) * binarization)
    return {"fs_coverage": coverage, "fs_sparsity": sparsity, "fs_binarization": binarization, "feature_selection": total}
