"""Stage 1 - camera -> BEV encoder (LSS + depth supervision).

    python training/train_bev.py --experiment F_cam_lidar_radar_fs [--epochs N] [--set key=value ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.cli import run  # noqa: E402

if __name__ == "__main__":
    run(default_stage="camera_bev")
