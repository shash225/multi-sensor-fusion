"""SECOND-FPN BEV backbone on the fused BEV map.

    fused (B, 128, 200, 200)
      ├─ block1: stride 2, 3 convs ....... (B, 128, 100, 100) ─ deblock ×1 ─┐
      └─ block2: stride 2, 5 convs ....... (B, 256,  50,  50) ─ deblock ×2 ─┤
                                                                      concat (B, 256, 100, 100)
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.common.blocks import ConvBNReLU


class SECONDFPN(nn.Module):
    def __init__(self, in_channels: int, cfg: Any) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        c_in = in_channels
        for n, stride, c_out, up, c_up in zip(cfg.layer_nums, cfg.layer_strides, cfg.num_filters,
                                              cfg.upsample_strides, cfg.num_upsample_filters):
            layers = [ConvBNReLU(c_in, c_out, 3, stride=stride)] + [ConvBNReLU(c_out, c_out, 3) for _ in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            if up == 1:
                self.deblocks.append(ConvBNReLU(c_out, c_up, 1))
            else:
                self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_out, c_up, up, stride=up, bias=False),
                                                   nn.BatchNorm2d(c_up), nn.ReLU(inplace=True)))
            c_in = c_out
        self.out_channels = int(sum(cfg.num_upsample_filters))
        self.out_stride = int(cfg.layer_strides[0])
        self.grad_checkpoint = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for block, deblock in zip(self.blocks, self.deblocks):
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            outs.append(deblock(x))
        return torch.cat(outs, 1)
