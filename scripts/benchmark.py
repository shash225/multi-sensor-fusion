"""Latency / memory benchmark of every experiment configuration (inference, batch 1, real nuScenes frames).

Weights do not affect speed, so trained checkpoints are not required.

    python scripts/benchmark.py
    python scripts/benchmark.py --set dataset.camera.image_size=[256,704]     # what-if for bigger GPUs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch  # noqa: E402
from models.fusion_model import build_model  # noqa: E402
from utils.config import list_experiments, load_config  # noqa: E402
from utils.device import amp_enabled, get_device, gpu_memory_report, move_to_device  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from utils.timer import StageTimer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    names = args.experiments or list(list_experiments())
    rows = []
    for name in names:
        cfg = load_config(experiment=name, overrides=args.set)
        device = get_device(cfg.runtime.device)
        use_amp = amp_enabled(device, bool(cfg.train.amp))
        ds = NuScenesFusionDataset(cfg, cfg.dataset.val_split, train=False, with_targets=False)
        model = build_model(cfg).to(device).eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        times, stage_times = [], {}
        with torch.no_grad():
            for i in range(args.frames + 2):
                batch = move_to_device(collate_batch([ds[i % len(ds)]]), device)
                timer = StageTimer()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    model(batch, timer=timer)
                if i >= 2:  # skip cuDNN warm-up
                    times.append(timer.total_ms)
                    for k, v in timer.as_dict().items():
                        stage_times.setdefault(k, []).append(v)
        mem = gpu_memory_report(device)
        rows.append({"experiment": name, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                     "latency_ms_median": round(float(np.median(times)), 1), "fps": round(1000 / float(np.median(times)), 1),
                     "peak_memory_MB": round(mem.get("peak_allocated_mb", 0.0)), "amp": use_amp,
                     "stages_ms": {k: round(float(np.median(v)), 1) for k, v in stage_times.items()}})
        print(json.dumps(rows[-1]))
        del model
    print(f"\n{'experiment':<30}{'params(M)':>10}{'ms':>8}{'FPS':>7}{'peak MB':>9}")
    for r in rows:
        print(f"{r['experiment']:<30}{r['params_M']:>10}{r['latency_ms_median']:>8}{r['fps']:>7}{r['peak_memory_MB']:>9}")


if __name__ == "__main__":
    main()
