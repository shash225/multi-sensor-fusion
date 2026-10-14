"""Single entry point for the whole project.

    python main.py preprocess [--maps]                     build sample-info caches (+ map caches)
    python main.py validate                                Phase 2: sensor sync / calibration report + images
    python main.py smoke [--experiments all]               shapes / losses / memory / latency on real data
    python main.py train --stage joint --experiment F_cam_lidar_radar_fs [--epochs 20]
    python main.py train --stage curriculum                the 7-stage curriculum (stages 1-7 in order)
    python main.py evaluate --checkpoint checkpoints/F_cam_lidar_radar_fs/joint/best.pt [--official]
    python main.py ablation [--train] [--epochs 20]        train/evaluate experiments A-F + plots
    python main.py infer --scene scene-0103 [--video]      render a scene replay to images / MP4
    python main.py bev --scene scene-0103 --frame 12       every BEV of one frame in a single image
    python main.py planner --planner baseline              evaluate planners with GT perception
    python main.py weather --weather all|fog|rain          adverse-weather test of trained models (read-only)
    python main.py benchmark                               latency / memory per experiment
    python main.py app                                     launch the Streamlit dashboard
    python main.py test                                    run the test-suite
"""
from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "preprocess": "scripts/preprocess.py",
    "validate": "scripts/validate_sensors.py",
    "smoke": "scripts/smoke_test.py",
    "train": "training/cli.py",
    "evaluate": "scripts/evaluate.py",
    "ablation": "evaluation/ablation.py",
    "infer": "scripts/inference.py",
    "bev": "scripts/render_bev.py",
    "planner": "scripts/eval_planner.py",
    "benchmark": "scripts/benchmark.py",
    "download": "scripts/download_dataset.py",
    "weather": "evaluation/adverse_weather.py",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "app":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), *rest]))
    if cmd == "test":
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", *rest], cwd=ROOT))
    if cmd not in COMMANDS:
        print(f"Unknown command {cmd!r}.\n{__doc__}")
        sys.exit(2)
    script = ROOT / COMMANDS[cmd]
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
