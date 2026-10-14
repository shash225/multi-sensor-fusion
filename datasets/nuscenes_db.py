"""Lightweight nuScenes table loader.

Reads the raw JSON tables of any nuScenes version (``v1.0-mini``, ``v1.0-trainval``)
and builds the same reverse indexes as the official devkit (``sample['data']``,
``sample['anns']``, ``sample_data['channel']``), without importing the devkit.
The devkit stays an *optional* dependency, used only for the official evaluation.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

TABLES = (
    "attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
    "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility",
)


class NuScenesDB:
    """In-memory nuScenes database with token lookup.

    Args:
        root: dataset root containing ``samples/``, ``sweeps/``, ``maps/`` and ``<version>/``.
        version: table folder name, e.g. ``"v1.0-mini"``.
    """

    def __init__(self, root: str | Path, version: str = "v1.0-mini", verbose: bool = True) -> None:
        self.root = Path(root)
        self.version = version
        table_dir = self.root / version
        if not table_dir.is_dir():
            raise FileNotFoundError(
                f"nuScenes tables not found at '{table_dir}'. Set dataset.root / dataset.version "
                f"(configs/local.yaml) to the folder that contains '{version}/'."
            )
        start = time.time()
        self.tables: dict[str, list[dict]] = {}
        self._index: dict[str, dict[str, dict]] = {}
        for name in TABLES:
            path = table_dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing nuScenes table {path}")
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            self.tables[name] = records
            self._index[name] = {r["token"]: r for r in records}
        self._build_reverse_index()
        if verbose:
            logger.info(
                "Loaded nuScenes %s from %s: %d scenes, %d samples, %d annotations (%.1fs)",
                version, self.root, len(self.tables["scene"]), len(self.tables["sample"]),
                len(self.tables["sample_annotation"]), time.time() - start,
            )

    # ------------------------------------------------------------------ index
    def _build_reverse_index(self) -> None:
        for rec in self.tables["sample"]:
            rec["data"] = {}
            rec["anns"] = []
        for sd in self.tables["sample_data"]:
            cs = self.get("calibrated_sensor", sd["calibrated_sensor_token"])
            sensor = self.get("sensor", cs["sensor_token"])
            sd["channel"] = sensor["channel"]
            sd["sensor_modality"] = sensor["modality"]
            if sd["is_key_frame"]:
                self.get("sample", sd["sample_token"])["data"][sensor["channel"]] = sd["token"]
        for ann in self.tables["sample_annotation"]:
            self.get("sample", ann["sample_token"])["anns"].append(ann["token"])
            inst = self.get("instance", ann["instance_token"])
            ann["category_name"] = self.get("category", inst["category_token"])["name"]
        self._scene_by_name = {s["name"]: s for s in self.tables["scene"]}
        self._log_to_map: dict[str, dict] = {}
        for m in self.tables["map"]:
            for log_token in m["log_tokens"]:
                self._log_to_map[log_token] = m

    # ------------------------------------------------------------------ access
    def get(self, table: str, token: str) -> dict[str, Any]:
        try:
            return self._index[table][token]
        except KeyError as exc:
            raise KeyError(f"Token {token!r} not found in table {table!r}") from exc

    def has(self, table: str, token: str) -> bool:
        return token in self._index[table]

    @property
    def scenes(self) -> list[dict]:
        return self.tables["scene"]

    def scene_by_name(self, name: str) -> dict:
        return self._scene_by_name[name]

    def scene_sample_tokens(self, scene: dict) -> list[str]:
        """Key-frame sample tokens of a scene in temporal order."""
        tokens, token = [], scene["first_sample_token"]
        while token:
            tokens.append(token)
            token = self.get("sample", token)["next"]
        return tokens

    def iter_sweeps(self, sd_token: str, direction: str = "prev") -> Iterator[dict]:
        """Walk the ``prev``/``next`` chain of a sample_data record (including itself)."""
        token = sd_token
        while token:
            rec = self.get("sample_data", token)
            yield rec
            token = rec[direction]

    def map_for_log(self, log_token: str) -> dict | None:
        return self._log_to_map.get(log_token)

    def file_path(self, sample_data: dict) -> Path:
        return self.root / sample_data["filename"]
