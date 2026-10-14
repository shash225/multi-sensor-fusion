"""Fusion-method registry: add a new fusion by decorating a class with ``@register_fusion("name")``.

A fusion class must accept ``(in_channels: dict[str, int], out_channels: int)`` and its
``forward(feats: dict[str, Tensor])`` must return ``(fused (B, C_out, H, W), info dict)``.
"""
from __future__ import annotations

from typing import Callable

from torch import nn

from models.fusion.fusion_methods import AttentionGatedFusion, ConcatFusion

FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "concat": ConcatFusion,
    "attention_gated": AttentionGatedFusion,
}


def register_fusion(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        FUSION_REGISTRY[name] = cls
        return cls
    return deco


def build_fusion(method: str, in_channels: dict[str, int], out_channels: int) -> nn.Module:
    if method not in FUSION_REGISTRY:
        raise KeyError(f"Unknown fusion method {method!r}; available: {sorted(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[method](in_channels, out_channels)
