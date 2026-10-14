"""Pre-compute the per-sample info caches (and optionally the reduced map rasters).

    python scripts/preprocess.py                 # train / val / all splits of dataset.version
    python scripts/preprocess.py --maps          # also cache the 0.5 m/px map priors (~1 min, ~1.5 GB RAM peak)
    python scripts/preprocess.py --set dataset.version=v1.0-trainval dataset.train_split=train dataset.val_split=val
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.info_builder import load_infos  # noqa: E402
from datasets.map_prior import MapPrior  # noqa: E402
from datasets.nuscenes_db import NuScenesDB  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", action="store_true", help="cache reduced map priors")
    parser.add_argument("--rebuild", action="store_true", help="ignore existing caches")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(overrides=args.set)
    t0 = time.time()
    db = NuScenesDB(cfg.dataset.root, cfg.dataset.version)
    for split in (cfg.dataset.train_split, cfg.dataset.val_split, "all"):
        infos = load_infos(cfg, split, db=db, rebuild=args.rebuild)
        n_boxes = sum(len(i["gt"]["labels"]) for i in infos)
        print(f"{split:<12} {len(infos):6d} samples  {n_boxes:7d} GT boxes")
    if args.maps:
        prior = MapPrior(Path(cfg.dataset.root), resolve_path(cfg.dataset.cache_dir))
        for rec in db.tables["map"]:
            arr = prior.load(rec["filename"])
            print(f"map {rec['filename']}: {None if arr is None else arr.shape}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
