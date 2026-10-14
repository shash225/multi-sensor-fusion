"""Image backbone: torchvision ResNet + a light FPN neck at stride 16."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.common.blocks import ConvBNReLU

logger = logging.getLogger(__name__)
_WEIGHTS = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights", "resnet50": "ResNet50_Weights"}


class ImageBackbone(nn.Module):
    """ResNet trunk (layer1-4) whose stride-32 output is upsampled and merged with stride 16.

    Input ``(BN, 3, H, W)`` -> output ``(BN, out_channels, H/16, W/16)``.
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True, out_channels: int = 256,
                 frozen_stages: int = 1, norm_eval: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = getattr(torchvision.models, _WEIGHTS[name]).IMAGENET1K_V1
                net = getattr(torchvision.models, name)(weights=weights)
            except Exception as exc:  # offline / download failure: keep running with random init
                logger.warning("Could not load ImageNet weights for %s (%s); using random init.", name, exc)
                net = getattr(torchvision.models, name)(weights=None)
        else:
            net = getattr(torchvision.models, name)(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        c3 = self.layer3[-1].conv1.in_channels if name != "resnet50" else 1024
        c4 = self.layer4[-1].conv1.in_channels if name != "resnet50" else 2048
        self.neck = nn.Sequential(ConvBNReLU(c3 + c4, out_channels, 1), ConvBNReLU(out_channels, out_channels, 3))
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.grad_checkpoint = False
        self._freeze()

    def _freeze(self) -> None:
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages[: self.frozen_stages + 1 if self.frozen_stages > 0 else 0]:
            stage.eval()
            for p in stage.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True) -> "ImageBackbone":
        super().train(mode)
        self._freeze()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def _run(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(x))
        x = self._run(self.layer2, x)
        c3 = self._run(self.layer3, x)
        c4 = self._run(self.layer4, c3)
        c4 = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        return self.neck(torch.cat([c3, c4], 1))
