"""Ablation study runner: train (optional) + evaluate every experiment with one recipe, then plot.

Hypothesis: learned feature selection before LiDAR-Radar fusion improves perception and
planning. Every experiment uses the identical recipe (stage ``joint``, same seed, epochs,
data); only the sensor set and fusion / feature-selection design change.

Outputs (``outputs/ablation/``): ``results.json``, ``results.csv``, ``results.md`` (table +
hypothesis deltas) and ``ablation_metrics.png`` (one panel per metric).

    python evaluation/ablation.py --train --epochs 20                 # train + evaluate default set
    python evaluation/ablation.py                                      # evaluate existing checkpoints
    python evaluation/ablation.py --experiments E_lidar_radar_fs C_lidar_radar_concat
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch  # noqa: E402
from evaluation.evaluator import evaluate_model, summary_line  # noqa: E402
from models.fusion_model import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import CONFIG_DIR, Config, read_yaml, resolve_path  # noqa: E402
from utils.device import amp_enabled, get_device, gpu_memory_report  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("ablation")

# (key, label, higher_is_better)
PLOT_METRICS = [
    ("NDS", "NDS", True), ("mAP", "mAP", True), ("mAP_present", "mAP, present classes", True),
    ("mAVE", "mAVE, m/s", False), ("ADE", "Planning ADE, m", False),
    ("FDE", "Planning FDE, m", False), ("collision_rate", "Collision rate", False),
    ("latency_ms", "Latency, ms", False),
]
ACCENT, MUTED_BAR = "#2a78d6", "#c3c2b7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def checkpoint_for(cfg: Config, experiment: str, stage: str = "joint", which: str = "last") -> Optional[Path]:
    """``which='last'`` (default): final weights after the full, identical training budget — no model
    selection on the evaluation split. ``which='best'``: the epoch with the best validation NDS
    (optimistic, because it is selected on the same split it is evaluated on)."""
    d = resolve_path(cfg.runtime.checkpoint_dir) / experiment / stage
    order = ("last.pt", "best.pt") if which == "last" else ("best.pt", "last.pt")
    for name in order:
        if (d / name).exists():
            return d / name
    return None


def evaluate_checkpoint(path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    ckpt = load_checkpoint(path)
    cfg = Config(ckpt["config"])
    for k, v in (overrides or {}).items():
        cfg.set_path(k, v)
    device = get_device(cfg.runtime.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    ds = NuScenesFusionDataset(cfg, cfg.dataset.val_split, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=int(cfg.dataset.num_workers),
                        persistent_workers=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    res = evaluate_model(model, loader, cfg, device, amp_enabled(device, bool(cfg.train.amp)), compute_losses=False,
                         run_baseline_planner=True)
    res["peak_memory_mb"] = gpu_memory_report(device).get("peak_allocated_mb", 0.0)
    res["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    res["epoch"] = ckpt.get("epoch")
    return res


def flatten(name: str, label: str, res: dict[str, Any]) -> dict[str, Any]:
    det = res["detection"]
    plan = res.get("planning_learned", {})
    base = res.get("planning_baseline", {})
    fs = res.get("feature_selection", {})
    return {
        "experiment": name, "label": label, "epoch": res.get("epoch"),
        "NDS": det["NDS"], "mAP": det["mAP"], "mAP_present": det["mAP_present_classes"],
        "mATE": det["mATE"], "mASE": det["mASE"], "mAOE": det["mAOE"], "mAVE": det["mAVE"], "mAAE": det["mAAE"],
        "ADE": plan.get("ade"), "FDE": plan.get("fde"), "L2_3s": plan.get("l2_3s"),
        "collision_rate": plan.get("collision_rate"),
        "baseline_planner_ADE": base.get("ade"), "baseline_planner_collision": base.get("collision_rate"),
        "latency_ms": res["latency_ms"].get("total"), "params_M": res["params_M"], "peak_memory_mb": res["peak_memory_mb"],
        "lidar_selected_ratio": fs.get("lidar_effective_selected_ratio"),
        "radar_selected_ratio": fs.get("radar_effective_selected_ratio"),
        "lidar_mean_importance": fs.get("lidar_mean_spatial_importance"),
        "radar_mean_importance": fs.get("radar_mean_spatial_importance"),
    }


def plot(rows: list[dict[str, Any]], out_path: Path, proposed: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in PLOT_METRICS if any(r.get(m[0]) is not None for r in rows)]
    cols = 4
    nrows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.3 * cols + 2.6, 0.42 * len(rows) * nrows + 1.6 * nrows),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    labels = [r["label"] for r in rows]
    for i, (ax, (key, title, higher)) in enumerate(zip(axes.flat, metrics)):
        vals = [r.get(key) if r.get(key) is not None else 0.0 for r in rows]
        colors = [ACCENT if r["experiment"] == proposed else MUTED_BAR for r in rows]
        y = list(range(len(rows)))[::-1]
        ax.barh(y, vals, color=colors, height=0.62, edgecolor="#fcfcfb", linewidth=2)
        ax.set_yticks(y)
        if i % cols == 0:  # one shared label column per row of panels
            ax.set_yticklabels(labels, fontsize=8, color=INK2)
        ax.set_title(f"{title}  ({'higher' if higher else 'lower'} is better)", fontsize=9, color=INK, loc="left")
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(axis="x", colors="#898781", labelsize=7.5)
        ax.tick_params(axis="y", length=0)
        vmax = max(vals) if max(vals) > 0 else 1.0
        ax.set_xlim(0, vmax * 1.22)
        for yi, v, r in zip(y, vals, rows):
            if r.get(key) is not None:
                ax.text(v + vmax * 0.02, yi, f"{v:.3f}" if v < 10 else f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Ablation on nuScenes-mini val (81 frames, final epoch, 1 seed) - proposed model in blue", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def hypothesis_table(rows: list[dict[str, Any]]) -> list[str]:
    by = {r["experiment"]: r for r in rows}
    pairs = [("E_lidar_radar_fs", "C_lidar_radar_concat", "Feature selection vs concatenation (LiDAR+Radar)"),
             ("E_lidar_radar_fs", "D_lidar_radar_attention", "Feature selection vs attention-only (LiDAR+Radar)"),
             ("F_cam_lidar_radar_fs", "I_cam_lidar_radar_attention", "Feature selection vs attention-only (Cam+LiDAR+Radar)"),
             ("F_cam_lidar_radar_fs", "E_lidar_radar_fs", "Adding the camera BEV"),
             ("C_lidar_radar_concat", "A_lidar_only", "Adding radar to LiDAR")]
    lines = ["| Comparison | ΔNDS | ΔmAP | ΔmAVE (↓) | ΔADE (↓) | ΔCollision (↓) | ΔLatency ms |", "|---|---|---|---|---|---|---|"]
    for a, b, desc in pairs:
        if a in by and b in by:
            ra, rb = by[a], by[b]

            def d(k: str) -> str:
                if ra.get(k) is None or rb.get(k) is None:
                    return "n/a"
                return f"{ra[k] - rb[k]:+.4f}"
            lines.append(f"| {desc} ({a[:1]} vs {b[:1]}) | {d('NDS')} | {d('mAP')} | {d('mAVE')} | {d('ADE')} | "
                         f"{d('collision_rate')} | {d('latency_ms')} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", default=None, help="default: ablation.yaml default_run")
    parser.add_argument("--train", action="store_true", help="train experiments that have no checkpoint yet")
    parser.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], help="training config overrides key=value")
    parser.add_argument("--proposed", default="F_cam_lidar_radar_fs")
    parser.add_argument("--which", default="last", choices=["last", "best"],
                        help="evaluate the final weights (fair, default) or the best-validation-NDS epoch")
    parser.add_argument("--plot-only", action="store_true", help="redraw table/plot from results.json")
    args = parser.parse_args()
    setup_logging()
    table = read_yaml(CONFIG_DIR / "ablation.yaml")
    names = args.experiments or table["default_run"]
    from utils.config import load_config
    base_cfg = load_config(overrides=args.set)
    out_dir = resolve_path(base_cfg.runtime.output_dir) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = json.loads((out_dir / "results.json").read_text())
        write_reports(rows, out_dir, base_cfg, args.proposed, args.which)
        return
    rows = []
    for name in names:
        path = checkpoint_for(base_cfg, name, which=args.which)
        if args.retrain or (args.train and path is None):
            from training.trainer import Trainer

            t0 = time.time()
            Trainer(name, "joint", args.set, args.epochs).fit()
            logger.info("Trained %s in %.1f min", name, (time.time() - t0) / 60)
            torch.cuda.empty_cache()
            path = checkpoint_for(base_cfg, name, which=args.which)
        if path is None:
            logger.warning("No checkpoint for %s - skipped (run with --train)", name)
            continue
        res = evaluate_checkpoint(path)
        logger.info("%s | %s", name, summary_line(res))
        rows.append(flatten(name, table["experiments"][name].get("label", name), res))
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
        torch.cuda.empty_cache()
    if not rows:
        return
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_reports(rows, out_dir, base_cfg, args.proposed, args.which)


def write_reports(rows: list[dict[str, Any]], out_dir: Path, cfg: Config, proposed: str, which: str) -> None:
    cols = ["label", "epoch", "NDS", "mAP", "mAP_present", "mAVE", "ADE", "FDE", "collision_rate",
            "baseline_planner_ADE", "baseline_planner_collision", "latency_ms"]
    md = [f"# Ablation results (nuScenes {cfg.dataset.version}, {cfg.dataset.val_split}, {which} checkpoint)", "",
          "ADE / FDE / collision_rate: learned planning head. baseline_planner_*: classical planner (A* + iLQR) fed with "
          "the model's own detections.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(r["label"] if c == "label" else "n/a" if v is None else str(v) if c == "epoch" else f"{v:.4f}")
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "## Hypothesis tests (differences, first minus second)", ""] + hypothesis_table(rows)
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    plot(rows, out_dir / "ablation_metrics.png", proposed)
    print("\n".join(md))


if __name__ == "__main__":
    main()
