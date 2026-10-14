"""Official nuScenes scene splits.

The mini splits are embedded (they are tiny). The full ``train``/``val`` splits come
from the official devkit (``nuscenes.utils.splits``) so they always match the benchmark.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MINI_TRAIN = ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100"]
MINI_VAL = ["scene-0103", "scene-0916"]


def get_split_scenes(split: str, available_scenes: list[str]) -> list[str]:
    """Scene names of ``split`` that exist in the loaded version.

    Supported splits: ``mini_train``, ``mini_val``, ``train``, ``val``, ``test`` and ``all``.
    """
    if split == "all":
        return sorted(available_scenes)
    if split == "mini_train":
        names = MINI_TRAIN
    elif split == "mini_val":
        names = MINI_VAL
    else:
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                f"Split '{split}' needs the official devkit: pip install nuscenes-devkit --no-deps"
            ) from exc
        names = create_splits_scenes()[split]
    available = set(available_scenes)
    selected = [n for n in names if n in available]
    if not selected:
        raise ValueError(f"No scene of split '{split}' is present in this nuScenes version.")
    if len(selected) < len(names):
        logger.warning("Split %s: %d/%d scenes present", split, len(selected), len(names))
    return selected
