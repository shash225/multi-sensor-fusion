"""Checkpoint discovery and cached pipeline construction for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from utils.config import CONFIG_DIR, Config, load_config, read_yaml, resolve_path

STAGE_PREFERENCE = ("end_to_end", "joint", "detection", "fusion", "trajectory")
SHARED_FALLBACK = {"A_lidar_only": "lidar", "B_radar_only": "radar"}


def dashboard_methods() -> dict[str, str]:
    return dict(read_yaml(CONFIG_DIR / "ablation.yaml").get("dashboard_methods", {}))


def experiment_labels(short: bool = False) -> dict[str, str]:
    exps = read_yaml(CONFIG_DIR / "ablation.yaml").get("experiments", {})
    return {k: (v.get("short") if short else None) or v.get("label", k) for k, v in exps.items()}


def find_checkpoint(cfg: Config, experiment: str) -> Optional[Path]:
    root = resolve_path(cfg.runtime.checkpoint_dir)
    for stage in STAGE_PREFERENCE:
        for name in ("best.pt", "last.pt"):
            p = root / experiment / stage / name
            if p.exists():
                return p
    shared = SHARED_FALLBACK.get(experiment)
    if shared:
        for name in ("best.pt", "last.pt"):
            p = root / "_shared" / shared / name
            if p.exists():
                return p
    return None


def available_versions(cfg: Config) -> dict[str, bool]:
    root = Path(cfg.dataset.root)
    return {v: (root / v).is_dir() for v in ("v1.0-mini", "v1.0-trainval")}


def base_config(version: str, device: str) -> Config:
    overrides = [f"dataset.version={version}", f"runtime.device={device}"]
    if version == "v1.0-trainval":
        overrides += ["dataset.train_split=train", "dataset.val_split=val"]
    return load_config(overrides=overrides)


def load_ablation_results(cfg: Config) -> Optional[list[dict[str, Any]]]:
    path = resolve_path(cfg.runtime.output_dir) / "ablation" / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_history(cfg: Config, experiment: str, stage: str = "joint") -> list[dict[str, Any]]:
    path = resolve_path(cfg.runtime.output_dir) / "logs" / experiment / stage / "history.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
