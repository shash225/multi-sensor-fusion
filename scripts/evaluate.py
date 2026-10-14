"""Evaluate a checkpoint on the validation split (all metric families) and save a JSON report.

    python scripts/evaluate.py --checkpoint checkpoints/F_cam_lidar_radar_fs/joint/best.pt
    python scripts/evaluate.py --checkpoint ... --official      # also run the official nuScenes devkit evaluation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch  # noqa: E402
from evaluation.evaluator import evaluate_model, summary_line  # noqa: E402
from evaluation.official_eval import run_official_eval, to_submission  # noqa: E402
from models.fusion_model import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import Config, load_config, resolve_path  # noqa: E402
from utils.device import amp_enabled, get_device, move_to_device  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

MACHINE_KEYS = ("dataset.root", "dataset.version", "dataset.cache_dir", "runtime.device", "runtime.output_dir")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--official", action="store_true", help="also run nuscenes-devkit DetectionEval")
    parser.add_argument("--no-baseline-planner", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    setup_logging()
    local = load_config(overrides=args.set)
    ckpt = load_checkpoint(Path(args.checkpoint))
    cfg = Config(ckpt["config"])
    for key in MACHINE_KEYS:
        cfg.set_path(key, local.get_path(key))
    device = get_device(cfg.runtime.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    ds = NuScenesFusionDataset(cfg, cfg.dataset.val_split, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=int(local.dataset.num_workers))
    use_amp = amp_enabled(device, bool(cfg.train.amp))
    res = evaluate_model(model, loader, cfg, device, use_amp, run_baseline_planner=not args.no_baseline_planner)
    print(summary_line(res))
    out_dir = resolve_path(local.runtime.output_dir) / "evaluation" / Path(args.checkpoint).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.official:
        preds = {}
        model.eval()
        with torch.no_grad():
            for batch in loader:
                meta = batch["meta"]
                batch = move_to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(batch)
                for b, det in enumerate(model.decode(out, score_threshold=0.05)):
                    preds[meta[b]["token"]] = {k: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy()
                                               for k, v in det.items()}
        infos = {i["token"]: i for i in ds.infos}
        mods = {"camera": model.use_camera, "lidar": model.use_lidar, "radar": model.use_radar}
        res["official_devkit"] = run_official_eval(to_submission(preds, infos, list(cfg.classes), mods), cfg.dataset.root,
                                                   cfg.dataset.version, cfg.dataset.val_split, out_dir)
        o = res["official_devkit"]
        print(f"official devkit: NDS {o['nd_score']:.4f} mAP {o['mean_ap']:.4f}")
    (out_dir / "report.json").write_text(json.dumps(res, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x)))
    print(f"report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
