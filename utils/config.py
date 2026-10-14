"""Layered YAML configuration.

Merge order (later wins):

1. ``configs/{dataset,model,fusion,planning,training}.yaml``  (committed defaults)
2. ``configs/hardware/<profile>.yaml``                         (GPU-specific sizes)
3. experiment overrides from ``configs/ablation.yaml``         (fusion ablations)
4. ``configs/local.yaml``                                      (machine-specific, git-ignored)
5. command-line ``--set a.b.c=value`` overrides

String values may reference environment variables with ``${env:NAME,default}``.
No filesystem path is hard-coded in Python code; everything flows from here.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_FILES = ("dataset.yaml", "model.yaml", "fusion.yaml", "planning.yaml", "training.yaml")
_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


class Config(dict):
    """A ``dict`` with attribute access, used for every configuration object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = _wrap(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # keep hasattr()/pickle semantics correct
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(copy.deepcopy(dict(self), memo))

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return ``cfg['a']['b']['c']`` for ``dotted='a.b.c'`` or ``default``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set a nested key, creating intermediate dictionaries as needed."""
        parts = dotted.split(".")
        node: MutableMapping = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], MutableMapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        """Plain-``dict`` deep copy (safe for YAML/JSON/pickle in checkpoints)."""
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: MutableMapping, update: Mapping) -> MutableMapping:
    """Recursively merge ``update`` into ``base`` (in place) and return ``base``."""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _interpolate_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_override(item: str) -> tuple[str, Any]:
    """Parse ``'a.b=3'`` into ``('a.b', 3)``; the value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"Override must look like key=value, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def list_experiments(config_dir: Path = CONFIG_DIR) -> dict[str, dict]:
    """Return the experiment table from ``configs/ablation.yaml``."""
    path = config_dir / "ablation.yaml"
    if not path.exists():
        return {}
    return read_yaml(path).get("experiments", {})


def load_config(
    experiment: Optional[str] = None,
    hardware: Optional[str] = None,
    overrides: Optional[Iterable[str]] = None,
    config_dir: Path = CONFIG_DIR,
    use_local: bool = True,
) -> Config:
    """Build the full configuration for a run.

    Args:
        experiment: name of an entry in ``configs/ablation.yaml`` (e.g. ``"F_cam_lidar_radar_fs"``).
            ``None`` uses ``runtime.default_experiment``.
        hardware: hardware profile name (``configs/hardware/<name>.yaml``). ``None`` uses
            ``runtime.hardware_profile``.
        overrides: ``key=value`` strings applied last.
        config_dir: directory containing the YAML files.
        use_local: merge ``configs/local.yaml`` when present.
    """
    merged: dict = {}
    for name in BASE_FILES:
        path = config_dir / name
        if path.exists():
            deep_merge(merged, read_yaml(path))

    local: dict = {}
    local_path = config_dir / "local.yaml"
    if use_local and local_path.exists():
        local = read_yaml(local_path)

    cli = [parse_override(o) for o in (overrides or [])]
    cli_dict = Config()
    for key, value in cli:
        cli_dict.set_path(key, value)

    hw_name = (
        hardware
        or Config(cli_dict).get_path("runtime.hardware_profile")
        or Config(local).get_path("runtime.hardware_profile")
        or Config(merged).get_path("runtime.hardware_profile")
    )
    if hw_name:
        hw_path = config_dir / "hardware" / f"{hw_name}.yaml"
        if not hw_path.exists():
            raise FileNotFoundError(f"Unknown hardware profile '{hw_name}' ({hw_path})")
        deep_merge(merged, read_yaml(hw_path))
        merged.setdefault("runtime", {})["hardware_profile"] = hw_name

    exp_name = experiment or Config(cli_dict).get_path("runtime.default_experiment") or Config(merged).get_path(
        "runtime.default_experiment"
    )
    if exp_name:
        experiments = list_experiments(config_dir)
        if exp_name not in experiments:
            raise KeyError(f"Unknown experiment '{exp_name}'. Available: {sorted(experiments)}")
        deep_merge(merged, experiments[exp_name].get("overrides", {}))
        merged.setdefault("runtime", {})["experiment"] = exp_name
        merged["runtime"]["experiment_description"] = experiments[exp_name].get("description", "")

    deep_merge(merged, local)
    deep_merge(merged, cli_dict.to_dict())
    return Config(_interpolate_env(merged))


def resolve_path(path_like: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones are anchored at the project root."""
    path = Path(os.path.expanduser(str(path_like)))
    return path if path.is_absolute() else (base / path)
