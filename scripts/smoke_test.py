"""End-to-end smoke test on real nuScenes data: shapes, losses, backward, memory, latency.

For each experiment it runs one training step (forward + loss + backward, AMP) and one
inference pass, printing every intermediate tensor shape so dimension bugs surface
immediately.

    python scripts/smoke_test.py                         # proposed model (F)
    python scripts/smoke_test.py --experiments all       # every ablation configuration
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch  # noqa: E402
from models.fusion_model import ALL_LOSSES, build_model  # noqa: E402
from utils.config import list_experiments, load_config  # noqa: E402
from utils.device import amp_enabled, get_device, gpu_memory_report, move_to_device  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from utils.timer import StageTimer  # noqa: E402


def describe(obj, prefix: str = "") -> list[str]:
    lines = []
    if torch.is_tensor(obj):
        lines.append(f"{prefix:<44s} {tuple(obj.shape)} {str(obj.dtype).replace('torch.', '')}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            lines += describe(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)) and obj and torch.is_tensor(obj[0]):
        lines.append(f"{prefix:<44s} list[{len(obj)}] first={tuple(obj[0].shape)}")
    return lines


def run(experiment: str, overrides: list[str], verbose: bool) -> dict:
    cfg = load_config(experiment=experiment, overrides=overrides)
    seed_everything(int(cfg.runtime.seed))
    device = get_device(cfg.runtime.device)
    use_amp = amp_enabled(device, cfg.train.amp)
    ds = NuScenesFusionDataset(cfg, cfg.dataset.train_split, train=True)
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True, collate_fn=collate_batch, num_workers=0)
    t0 = time.perf_counter()
    batch = next(iter(loader))
    load_ms = (time.perf_counter() - t0) * 1000
    model = build_model(cfg).to(device)
    model.set_grad_checkpoint(bool(cfg.train.grad_checkpoint))
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print("\n=== batch ===")
        print("\n".join(describe({k: v for k, v in batch.items() if k not in ("meta",)})))
    batch = move_to_device(batch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out = model(batch)
        losses = model.compute_losses(out, batch, cfg.loss, ALL_LOSSES)
    scaler.scale(losses["total"]).backward()
    scaler.step(opt)
    scaler.update()
    train_mem = gpu_memory_report(device).get("peak_allocated_mb", 0.0)
    if verbose:
        print("\n=== model outputs ===")
        print("\n".join(describe({k: v for k, v in out.items() if k not in ("fusion_info",)})))
        print("\n=== losses ===")
        for k, v in losses.items():
            print(f"{k:<24s} {float(v):.4f}")
    # Inference pass with per-stage timing.
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timer = StageTimer()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        model(batch)  # warm-up (cuDNN autotune)
        out = model(batch, timer=timer)
        dets = model.decode(out)
    infer_mem = gpu_memory_report(device).get("peak_allocated_mb", 0.0)
    finite = all(bool(torch.isfinite(v.detach()).all()) for v in losses.values())
    return {
        "experiment": experiment,
        "params_M": round(n_params / 1e6, 2),
        "batch_size": int(cfg.train.batch_size),
        "image_size": list(cfg.dataset.camera.image_size),
        "data_load_ms": round(load_ms, 1),
        "train_peak_mem_MB": round(train_mem, 0),
        "infer_peak_mem_MB": round(infer_mem, 0),
        "infer_stage_ms": {k: round(v, 1) for k, v in timer.as_dict().items()},
        "infer_total_ms": round(timer.total_ms, 1),
        "losses": {k: round(float(v.detach()), 4) for k, v in losses.items()},
        "losses_finite": bool(finite),
        "num_detections_b0": int(dets[0]["boxes"].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", default=["F_cam_lidar_radar_fs"], help="names or 'all'")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    setup_logging()
    names = list(list_experiments()) if args.experiments == ["all"] else args.experiments
    results = [run(n, args.set, verbose=not args.quiet and len(names) == 1) for n in names]
    if args.quiet:
        print(f"\n{'experiment':<30}{'params(M)':>10}{'train MB':>10}{'infer MB':>10}{'infer ms':>10}  finite{'L_total':>10}{'dets':>6}")
        for r in results:
            print(f"{r['experiment']:<30}{r['params_M']:>10}{r['train_peak_mem_MB']:>10.0f}{r['infer_peak_mem_MB']:>10.0f}"
                  f"{r['infer_total_ms']:>10.1f}  {str(r['losses_finite']):<6}{r['losses']['total']:>10.3f}{r['num_detections_b0']:>6}")
    else:
        print(json.dumps(results, indent=2))
    if not all(r["losses_finite"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
