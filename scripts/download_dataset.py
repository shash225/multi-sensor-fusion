"""Help obtaining nuScenes.

nuScenes is distributed under the nuScenes Terms of Use (CC BY-NC-SA 4.0, non-commercial).
The full dataset requires a free account at https://www.nuscenes.org/nuscenes#download;
the mini split (~4 GB) is also served from a public URL. This script never accepts the
terms on your behalf: pass ``--i-accept-the-nuscenes-terms`` after reading them.

    python scripts/download_dataset.py                       # print instructions + verify an existing copy
    python scripts/download_dataset.py --download-mini --dest D:/datasets/nuscenes --i-accept-the-nuscenes-terms
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config  # noqa: E402

MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
REQUIRED = ("samples", "sweeps", "maps")


def verify(root: Path, version: str) -> bool:
    ok = (root / version).is_dir() and all((root / d).is_dir() for d in REQUIRED)
    print(f"{root}: {'OK' if ok else 'incomplete'}")
    for d in (*REQUIRED, version, "can_bus"):
        print(f"  {'✓' if (root / d).exists() else '✗'} {d}{'  (optional CAN-bus expansion)' if d == 'can_bus' else ''}")
    return ok


def download_mini(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "v1.0-mini.tgz"
    if not archive.exists():
        print(f"Downloading {MINI_URL} -> {archive} (~4 GB)")

        def hook(blocks: int, size: int, total: int) -> None:
            done = blocks * size
            if total > 0 and blocks % 2000 == 0:
                print(f"  {done / 2**30:.2f} / {total / 2**30:.2f} GB", flush=True)

        urllib.request.urlretrieve(MINI_URL, archive, hook)
    print("Extracting ...")
    with tarfile.open(archive) as tar:
        tar.extractall(dest)
    print("Done. Set dataset.root in configs/local.yaml to", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-mini", action="store_true")
    parser.add_argument("--dest", default=None)
    parser.add_argument("--i-accept-the-nuscenes-terms", dest="accept", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if args.download_mini:
        if not args.accept:
            raise SystemExit("Read https://www.nuscenes.org/terms-of-use and re-run with --i-accept-the-nuscenes-terms")
        download_mini(Path(args.dest or cfg.dataset.root))
        return
    print(__doc__)
    verify(Path(cfg.dataset.root), cfg.dataset.version)


if __name__ == "__main__":
    main()
