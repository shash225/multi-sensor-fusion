"""Checkpoint saving/loading with the configuration embedded.

Every checkpoint stores the exact configuration used to build the model, so the
dashboard and evaluation scripts can rebuild the network without guessing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def save_checkpoint(path: Path, model: nn.Module, config: dict, *, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Any = None, scaler: Any = None, epoch: int = 0, step: int = 0,
                    metrics: Optional[dict] = None, extra: Optional[dict] = None) -> None:
    """Atomically write a checkpoint (written to ``*.tmp`` and renamed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model_weights(model: nn.Module, state_dict: dict, prefixes: Optional[Iterable[str]] = None) -> dict[str, list[str]]:
    """Partially load weights, skipping keys that are absent or have a different shape.

    Args:
        prefixes: only keys starting with one of these prefixes are loaded (``None`` = all).

    Returns:
        ``{"loaded": [...], "skipped_shape": [...], "missing": [...]}`` for logging.
    """
    prefixes = tuple(prefixes) if prefixes is not None else None
    own = model.state_dict()
    new_state, skipped = {}, []
    for key, value in state_dict.items():
        if prefixes is not None and not key.startswith(prefixes):
            continue
        if key not in own:
            continue
        if own[key].shape != value.shape:
            skipped.append(key)
            continue
        new_state[key] = value
    model.load_state_dict(new_state, strict=False)
    wanted = [k for k in own if prefixes is None or k.startswith(prefixes)]
    missing = [k for k in wanted if k not in new_state]
    if skipped:
        logger.warning("Skipped %d tensors with mismatching shapes (e.g. %s)", len(skipped), skipped[:3])
    return {"loaded": list(new_state), "skipped_shape": skipped, "missing": missing}


def find_checkpoints(root: Path) -> list[Path]:
    """All ``*.pt`` files under ``root``, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
