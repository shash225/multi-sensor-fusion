"""Stage 4 - feature selection + fusion.

    python training/train_fusion.py --experiment F_cam_lidar_radar_fs [--epochs N] [--set key=value ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.cli import run  # noqa: E402

if __name__ == "__main__":
    run(default_stage="fusion")
