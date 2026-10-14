"""CenterPoint-style 3D detection head on the BEV feature map.

Input ``(B, 256, 100, 100)`` -> per-cell predictions:

=========  ======  =====================================================
branch     ch      meaning
=========  ======  =====================================================
heatmap    10      class centre likelihood (logits)
reg        2       sub-cell centre offset (dx, dy)
height     1       box-centre z (m)
dim        3       log(l), log(w), log(h)
rot        2       sin(yaw), cos(yaw)
vel        2       vx, vy (m/s, reference frame)
=========  ======  =====================================================

Loss = Gaussian focal loss (classification) + L1 on centre / size / orientation /
velocity at GT centre cells.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.common.blocks import ConvBNReLU

HEAD_BRANCHES = {"reg": 2, "height": 1, "dim": 3, "rot": 2, "vel": 2}


class CenterHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, head_channels: int = 64) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.shared = ConvBNReLU(in_channels, head_channels, 3)
        self.heads = nn.ModuleDict()
        for name, out in {"heatmap": num_classes, **HEAD_BRANCHES}.items():
            self.heads[name] = nn.Sequential(ConvBNReLU(head_channels, head_channels, 3), nn.Conv2d(head_channels, out, 1))
        nn.init.constant_(self.heads["heatmap"][-1].bias, -2.19)  # prior probability 0.1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        return {name: head(x) for name, head in self.heads.items()}


def gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, gamma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """CornerNet focal loss on probabilities; returns (summed loss, number of positives)."""
    pred = pred.float().clamp(1e-4, 1 - 1e-4)
    target = target.float()
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * (1 - target).pow(gamma) * neg
    return pos_loss.sum() + neg_loss.sum(), pos.sum()


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """``feat (B, C, H, W)``, ``ind (B, K)`` flat indices -> ``(B, K, C)``."""
    b, c = feat.shape[:2]
    feat = feat.view(b, c, -1).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(-1).expand(-1, -1, c))


def detection_loss(preds: dict[str, torch.Tensor], batch: dict[str, Any], weights: Any,
                   code_weights: list[float]) -> dict[str, torch.Tensor]:
    """CenterPoint losses. ``weights`` holds heatmap/center/bbox/orientation/velocity factors."""
    hm_sum, num_pos = gaussian_focal_loss(torch.sigmoid(preds["heatmap"].float()), batch["heatmap"])
    loss_cls = hm_sum / num_pos.clamp(min=1.0)

    ind, mask, vel_mask = batch["ind"], batch["mask"].float(), batch["vel_mask"].float()
    target = batch["reg_target"].float()
    pred = torch.cat([_gather(preds[k].float(), ind) for k in ("reg", "height", "dim", "rot", "vel")], -1)  # (B, K, 10)
    cw = pred.new_tensor(code_weights)
    l1 = (pred - target).abs() * cw
    norm = mask.sum().clamp(min=1.0)
    m = mask.unsqueeze(-1)
    loss_center = (l1[..., 0:3] * m).sum() / norm
    loss_bbox = (l1[..., 3:6] * m).sum() / norm
    loss_rot = (l1[..., 6:8] * m).sum() / norm
    loss_vel = (l1[..., 8:10] * vel_mask.unsqueeze(-1)).sum() / vel_mask.sum().clamp(min=1.0)
    total = (float(weights.heatmap) * loss_cls + float(weights.center) * loss_center + float(weights.bbox) * loss_bbox
             + float(weights.orientation) * loss_rot + float(weights.velocity) * loss_vel)
    return {"det_cls": loss_cls, "det_center": loss_center, "det_bbox": loss_bbox, "det_orientation": loss_rot,
            "det_velocity": loss_vel, "detection": total}


def aux_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Class-agnostic deep-supervision heatmap loss (resizes the target if needed)."""
    if logits.shape[-2:] != target.shape[-2:]:
        target = F.adaptive_max_pool2d(target, logits.shape[-2:])
    loss, num_pos = gaussian_focal_loss(torch.sigmoid(logits.float()), target)
    return loss / num_pos.clamp(min=1.0)
