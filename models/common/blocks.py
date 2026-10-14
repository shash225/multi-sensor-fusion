"""Reusable convolutional / attention building blocks."""
from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dilation: int = 1, relu: bool = True) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding=dilation * (kernel // 2), dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResBlock(nn.Module):
    """Basic residual block (two 3x3 convs) with optional dilation."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels, 3, dilation=dilation)
        self.conv2 = ConvBNReLU(channels, channels, 3, dilation=dilation, relu=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class SqueezeExcitation(nn.Module):
    """Channel attention (SE): global pooling -> bottleneck MLP -> sigmoid channel weights."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Linear(2 * channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        return torch.sigmoid(self.mlp(stats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights(x)[:, :, None, None]


class SpatialAttention(nn.Module):
    """CBAM spatial attention: [mean_c, max_c] -> 7x7 conv -> sigmoid map."""

    def __init__(self, kernel: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel attention followed by spatial attention)."""

    def __init__(self, channels: int, reduction: int = 4, kernel: int = 7) -> None:
        super().__init__()
        self.channel = SqueezeExcitation(channels, reduction)
        self.spatial = SpatialAttention(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
