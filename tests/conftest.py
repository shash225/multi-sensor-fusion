"""Shared pytest fixtures. Tests needing nuScenes are skipped when the dataset is not configured."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config  # noqa: E402


def _dataset_available() -> bool:
    try:
        cfg = load_config()
        return (Path(cfg.dataset.root) / cfg.dataset.version).is_dir()
    except Exception:
        return False


requires_dataset = pytest.mark.skipif(not _dataset_available(), reason="nuScenes not found (set dataset.root)")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def val_infos(cfg):
    from datasets.info_builder import load_infos

    return load_infos(cfg, cfg.dataset.val_split)
