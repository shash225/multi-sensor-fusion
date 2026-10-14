"""Decode CenterHead outputs into 3D boxes ``[x, y, z, l, w, h, yaw, vx, vy]`` + score + label."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils.bev_grid import BEVGrid


def circle_nms(centers: torch.Tensor, scores: torch.Tensor, radius: float) -> torch.Tensor:
    """Greedy distance-based NMS (CenterPoint). Returns kept indices (sorted by score)."""
    order = scores.argsort(descending=True)
    if radius <= 0 or order.numel() <= 1:
        return order
    c = centers[order].detach().cpu().double()
    close = (torch.cdist(c, c) < radius).numpy()  # greedy loop on CPU: no per-step GPU sync
    suppressed = np.zeros(order.numel(), dtype=bool)
    keep = []
    for i in range(order.numel()):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= close[i]
    return order[torch.as_tensor(keep, device=order.device, dtype=torch.long)]


@torch.no_grad()
def decode_detections(preds: dict[str, torch.Tensor], grid: BEVGrid, classes: list[str], score_threshold: float = 0.1,
                      max_detections: int = 300, nms_radius: Optional[dict[str, float]] = None) -> list[dict[str, torch.Tensor]]:
    """Per-sample detections: ``{"boxes": (K, 9), "scores": (K,), "labels": (K,)}`` in the reference frame."""
    heat = torch.sigmoid(preds["heatmap"].float())
    b, c, h, w = heat.shape
    peak = F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (heat == peak)
    k = min(max_detections, c * h * w)
    scores, idx = heat.view(b, -1).topk(k)
    labels = idx // (h * w)
    pix = idx % (h * w)
    iy, ix = (pix // w).float(), (pix % w).float()

    def gather(name: str) -> torch.Tensor:
        t = preds[name].float()
        return t.view(b, t.shape[1], -1).gather(2, pix.unsqueeze(1).expand(-1, t.shape[1], -1)).permute(0, 2, 1)

    reg, height, dim, rot, vel = gather("reg"), gather("height"), gather("dim"), gather("rot"), gather("vel")
    x = (ix + reg[..., 0]) * grid.resolution + grid.x_range[0]
    y = (iy + reg[..., 1]) * grid.resolution + grid.y_range[0]
    dims = dim.clamp(max=4.0).exp()  # log-size clamp avoids exp overflow from an untrained head
    yaw = torch.atan2(rot[..., 0], rot[..., 1])
    boxes = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), height, dims, yaw.unsqueeze(-1), vel], -1)

    results = []
    for i in range(b):
        keep = scores[i] > score_threshold
        bx, sc, lb = boxes[i][keep], scores[i][keep], labels[i][keep]
        if nms_radius and bx.shape[0]:
            kept = []
            for cls_id in lb.unique():
                sel = torch.nonzero(lb == cls_id).squeeze(1)
                r = float(nms_radius.get(classes[int(cls_id)], 0.0))
                kept.append(sel[circle_nms(bx[sel, :2], sc[sel], r)])
            order = torch.cat(kept)
            order = order[sc[order].argsort(descending=True)]
            bx, sc, lb = bx[order], sc[order], lb[order]
        results.append({"boxes": bx, "scores": sc, "labels": lb})
    return results
