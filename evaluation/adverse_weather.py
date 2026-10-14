"""Adverse-weather testing of the trained fusion models (evaluation only — weights are never written).

    python main.py weather --weather all                     # synthetic fog + rain, experiments A-F
    python main.py weather --weather fog --experiments F_cam_lidar_radar_fs
    python main.py weather --source real                     # nuScenes scenes tagged from their descriptions
    python main.py weather --report-only                     # rebuild tables / plots from saved runs

Two strictly separated data sources (never mixed in one table):

* ``synthetic`` — clear-weather mini_val frames corrupted by :mod:`datasets.weather`
  (camera, LiDAR and radar consistently). Every output is labelled SYNTHETIC.
* ``real`` — nuScenes scenes of the evaluation split whose *original* description mentions
  rain / fog / night. Scenes that belong to the checkpoint's training split are excluded
  (data leakage) and listed in the manifest.

Outputs: ``outputs/adverse_weather/<source>/`` — ``runs/<condition>__<experiment>.json``
(one per evaluation, reused unless ``--force``), ``results.{json,csv,md}``, ``plots/``,
``samples/``, ``manifest.json`` (data, parameters, checkpoint hashes, commands).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.info_builder import load_infos  # noqa: E402
from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch  # noqa: E402
from datasets.splits import get_split_scenes  # noqa: E402
from datasets.weather import (expand_conditions, load_weather_config, real_weather_manifest,  # noqa: E402
                              synthetic_spec)
from evaluation.ablation import checkpoint_for  # noqa: E402
from evaluation.evaluator import evaluate_model  # noqa: E402
from models.fusion_model import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import Config, list_experiments, load_config, resolve_path  # noqa: E402
from utils.device import amp_enabled, get_device  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("adverse_weather")
DEFAULT_EXPERIMENTS = ["A_lidar_only", "B_radar_only", "C_lidar_radar_concat", "D_lidar_radar_attention",
                       "E_lidar_radar_fs", "F_cam_lidar_radar_fs"]
PROPOSED = "F_cam_lidar_radar_fs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ one evaluation
@torch.no_grad()
def evaluate_condition(ckpt_path: Path, weather_spec: Optional[dict[str, Any]] = None,
                       infos: Optional[list[dict]] = None, split: Optional[str] = None,
                       max_samples: Optional[int] = None) -> dict[str, Any]:
    """Evaluate one checkpoint (read-only) on clean, synthetic-weather or real-weather data."""
    ckpt = load_checkpoint(ckpt_path)
    cfg = Config(ckpt["config"])
    if weather_spec is not None:
        cfg.set_path("weather.synthetic", weather_spec)
    if max_samples:
        cfg.set_path("train.max_val_samples", int(max_samples))
    device = get_device(cfg.runtime.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    ds = NuScenesFusionDataset(cfg, split or cfg.dataset.val_split, train=False, infos=infos)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch,
                        num_workers=int(cfg.dataset.num_workers), persistent_workers=False)
    t0 = time.perf_counter()
    res = evaluate_model(model, loader, cfg, device, amp_enabled(device, bool(cfg.train.amp)), compute_losses=False,
                         run_baseline_planner=False)
    res["wall_time_s"] = time.perf_counter() - t0
    res["num_frames"] = len(ds)
    res["epoch"] = ckpt.get("epoch")
    res["scenes"] = sorted({i["scene_name"] for i in ds.infos})
    return res


def flat_row(source: str, condition: str, label: str, experiment: str, exp_label: str, res: dict[str, Any]) -> dict:
    det = res["detection"]
    pr = det["PR@2m"]
    lat = res["latency_ms"].get("total", float("nan"))
    row = {"source": source, "condition": condition, "condition_label": label, "experiment": experiment,
           "experiment_label": exp_label, "frames": res["num_frames"], "scenes": " ".join(res["scenes"]),
           "NDS": det["NDS"], "mAP": det["mAP"], "mAP_present": det["mAP_present_classes"], "mATE": det["mATE"],
           "mAVE": det["mAVE"], "precision": pr["precision"], "recall": pr["recall"], "TP": pr["tp"], "FP": pr["fp"],
           "FN": pr["fn"], "center_error_m": pr["mean_center_error_m"], "latency_ms": lat,
           "FPS": 1000.0 / lat if lat and lat > 0 else float("nan"),
           "AP_car": det["per_class"]["car"]["AP"], "AP_pedestrian": det["per_class"]["pedestrian"]["AP"]}
    plan = res.get("planning_learned", {})
    row.update({"ADE": plan.get("ade", float("nan")), "collision_rate": plan.get("collision_rate", float("nan"))})
    for k, v in (res.get("weather_stats") or {}).items():
        row[f"ws_{k}"] = v
    for k, v in (res.get("fusion_weights") or {}).items():
        row[f"trust_{k}"] = v
    fs = res.get("feature_selection") or {}
    for m in ("lidar", "radar"):
        if f"{m}_effective_selected_ratio" in fs:
            row[f"fs_{m}_kept"] = fs[f"{m}_effective_selected_ratio"]
            row[f"fs_{m}_spatial_importance"] = fs[f"{m}_mean_spatial_importance"]
    return row


def add_relative(rows: list[dict]) -> None:
    """Relative change vs the clear-weather run of the same experiment and source."""
    base = {(r["source"], r["experiment"]): r for r in rows if r["condition"] in ("clear",)}
    for r in rows:
        b = base.get((r["source"], r["experiment"]))
        for k in ("NDS", "mAP", "precision", "recall", "FPS"):
            r[f"rel_{k}"] = (r[k] - b[k]) / b[k] if b is not None and b[k] else float("nan")


# ------------------------------------------------------------------ synthetic study
def run_synthetic(args: argparse.Namespace, base_cfg: Config, wcfg: dict, out_dir: Path) -> tuple[list[dict], dict]:
    conditions = expand_conditions(args.weather, wcfg)
    exps = list_experiments()
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows, errors, hashes = [], [], {}
    for exp in args.experiments:
        path = checkpoint_for(base_cfg, exp, "joint", args.which)
        if path is None:
            errors.append({"experiment": exp, "error": "no trained checkpoint (train it with main.py ablation --train)"})
            logger.error("No checkpoint for %s - skipped and reported", exp)
            continue
        before = sha256(path)
        for cond in conditions:
            run_file = runs_dir / f"{cond}__{exp}.json"
            label = wcfg["conditions"][cond]["label"]
            if run_file.exists() and not args.force:
                res = json.loads(run_file.read_text(encoding="utf-8"))
                logger.info("reuse %s", run_file.name)
            else:
                logger.info("evaluate %s | %s ...", exp, cond)
                try:
                    spec = None if cond == "clear" else synthetic_spec(cond, wcfg)
                    res = evaluate_condition(path, spec, max_samples=args.max_samples)
                except Exception as exc:  # report, never hide
                    errors.append({"experiment": exp, "condition": cond, "error": repr(exc),
                                   "traceback": traceback.format_exc()})
                    logger.exception("FAILED %s | %s", exp, cond)
                    continue
                res["checkpoint"] = str(path.relative_to(ROOT))
                res["weather_spec"] = None if cond == "clear" else synthetic_spec(cond, wcfg)
                run_file.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
            row = flat_row("synthetic", cond, label, exp, exps.get(exp, {}).get("label", exp), res)
            rows.append(row)
            logger.info("  %-14s NDS %.4f mAP %.4f P %.3f R %.3f %.1f FPS", cond, row["NDS"], row["mAP"],
                        row["precision"], row["recall"], row["FPS"])
        after = sha256(path)
        hashes[exp] = {"checkpoint": str(path.relative_to(ROOT)), "sha256_before": before, "sha256_after": after,
                       "unchanged": before == after}
        if before != after:  # pragma: no cover - must never happen
            raise RuntimeError(f"Checkpoint {path} changed during weather testing!")
    add_relative(rows)
    meta = {"conditions": {c: wcfg["conditions"][c] for c in conditions}, "checkpoints": hashes, "errors": errors,
            "parameters": {k: wcfg[k] for k in ("camera", "lidar", "radar", "seed")},
            "data": {"version": base_cfg.dataset.version, "split": base_cfg.dataset.val_split,
                     "scenes": sorted({s for r in rows for s in r["scenes"].split()}),
                     "note": "SYNTHETIC weather applied to clear-weather validation frames (no retraining)"}}
    return rows, meta


# ------------------------------------------------------------------ real-weather study
def run_real(args: argparse.Namespace, base_cfg: Config, wcfg: dict, out_dir: Path) -> tuple[list[dict], dict]:
    from datasets.nuscenes_db import NuScenesDB
    db = NuScenesDB(base_cfg.dataset.root, base_cfg.dataset.version, verbose=False)
    split = args.split or base_cfg.dataset.val_split
    split_scenes = get_split_scenes(split, [s["name"] for s in db.scenes])
    exps = list_experiments()
    rows, errors, hashes, manifests = [], [], {}, {}
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    infos_all = load_infos(base_cfg, split, db=db)
    for exp in args.experiments:
        path = checkpoint_for(base_cfg, exp, "joint", args.which)
        if path is None:
            errors.append({"experiment": exp, "error": "no trained checkpoint"})
            continue
        ck_cfg = Config(load_checkpoint(path)["config"])
        train_split = ck_cfg.dataset.train_split
        train_scenes = get_split_scenes(train_split, [s["name"] for s in db.scenes])
        manifest = real_weather_manifest(db, split_scenes, train_scenes, wcfg["real_weather"])
        manifests[exp] = {"train_split": train_split, "eval_split": split, "scenes": manifest}
        before = sha256(path)
        by_cond: dict[str, list[str]] = {}
        for rec in manifest:
            if rec["in_training_split"]:
                continue  # leakage guard
            by_cond.setdefault(rec["condition"], []).append(rec["scene"])
        for cond, scenes in sorted(by_cond.items(), key=lambda kv: kv[0] != "clear"):
            infos = [i for i in infos_all if i["scene_name"] in scenes]
            run_file = runs_dir / f"{cond}__{exp}.json"
            if run_file.exists() and not args.force:
                res = json.loads(run_file.read_text(encoding="utf-8"))
            else:
                try:
                    res = evaluate_condition(path, None, infos=infos, split=split, max_samples=args.max_samples)
                except Exception as exc:
                    errors.append({"experiment": exp, "condition": cond, "error": repr(exc),
                                   "traceback": traceback.format_exc()})
                    logger.exception("FAILED %s | %s", exp, cond)
                    continue
                res["checkpoint"] = str(path.relative_to(ROOT))
                run_file.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
            rows.append(flat_row("real", cond, f"Real nuScenes: {cond}", exp, exps.get(exp, {}).get("label", exp), res))
        after = sha256(path)
        hashes[exp] = {"checkpoint": str(path.relative_to(ROOT)), "sha256_before": before, "sha256_after": after,
                       "unchanged": before == after}
    add_relative(rows)
    any_manifest = next(iter(manifests.values()), {"scenes": []})
    missing = [c for c in ("rain", "fog") if not any(r["condition"] == c and not r["in_training_split"]
                                                     for r in any_manifest["scenes"])]
    meta = {"manifests": manifests, "checkpoints": hashes, "errors": errors,
            "not_testable": {c: f"no '{c}' scene in the evaluation split '{split}' outside the training split"
                             for c in missing},
            "data": {"version": base_cfg.dataset.version, "split": split}}
    return rows, meta


# ------------------------------------------------------------------ reports
def fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{nd}f}"
    return str(v)


def write_reports(rows: list[dict], meta: dict, out_dir: Path, source: str, command: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"rows": rows, **meta}, indent=1, default=float), encoding="utf-8")
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    tag = "SYNTHETIC weather (physically-motivated corruptions of clear-weather nuScenes-mini val frames)" \
        if source == "synthetic" else "REAL weather (nuScenes scenes tagged from their original descriptions)"
    lines = [f"# Adverse-weather results — {tag}", "",
             f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} with `{command}`. Weights loaded read-only "
             "(SHA-256 identical before/after: " + ", ".join(f"{k}: {'yes' if v['unchanged'] else 'NO'}"
                                                             for k, v in meta.get("checkpoints", {}).items()) + ").", "",
             "Precision / recall: detections with score ≥ 0.3 matched to GT of the same class within 2 m centre "
             "distance. mAP: nuScenes centre-distance mAP (0.5/1/2/4 m). mAP50 (2-D IoU ≥ 0.5) is not defined for "
             "this 3-D BEV detector. FPS = 1000 / median network latency (data loading and weather simulation "
             "excluded).", ""]
    if not rows:
        lines += ["**No evaluation could be run.**", ""]
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], []).append(r)
    for exp, rs in by_exp.items():
        lines += [f"## {rs[0]['experiment_label']} (`{exp}`)", "",
                  "| Weather | Data | Frames | Precision | Recall | NDS | mAP | Missed (FN) | False pos. (FP) | "
                  "Centre err. m | FPS | ΔNDS vs clear | ΔRecall vs clear |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            rel = "baseline" if r["condition"] == "clear" else fmt(100 * r["rel_NDS"], 1) + " %"
            relr = "baseline" if r["condition"] == "clear" else fmt(100 * r["rel_recall"], 1) + " %"
            data = "synthetic" if source == "synthetic" else "real"
            lines.append(f"| {r['condition_label']} | {data}: {r['scenes']} | {r['frames']} | {fmt(r['precision'])} | "
                         f"{fmt(r['recall'])} | {fmt(r['NDS'], 4)} | {fmt(r['mAP'], 4)} | {r['FN']} | {r['FP']} | "
                         f"{fmt(r['center_error_m'], 2)} | {fmt(r['FPS'], 1)} | {rel} | {relr} |")
        lines.append("")
    if source == "synthetic" and rows:
        conds = list(dict.fromkeys(r["condition"] for r in rows))
        lines += ["## Fusion vs single sensors — NDS by condition", "",
                  "| Experiment | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for exp, rs in by_exp.items():
            m = {r["condition"]: r for r in rs}
            lines.append(f"| {rs[0]['experiment_label']} | " + " | ".join(fmt(m[c]['NDS'], 4) if c in m else "–" for c in conds) + " |")
        lines += ["", "## Relative NDS change vs clear (%)", "", "| Experiment | " + " | ".join(conds[1:]) + " |",
                  "|---|" + "---|" * (len(conds) - 1)]
        for exp, rs in by_exp.items():
            m = {r["condition"]: r for r in rs}
            lines.append(f"| {rs[0]['experiment_label']} | " + " | ".join(
                fmt(100 * m[c]['rel_NDS'], 1) if c in m else "–" for c in conds[1:]) + " |")
        lines += ["", "## Simulated sensor degradation (mean over frames)", "",
                  "| Condition | Camera transmission | Camera contrast ratio | LiDAR points kept | LiDAR backscatter pts | "
                  "Radar points kept | Radar loss dB |", "|---|---|---|---|---|---|---|"]
        f_rows = {r["condition"]: r for r in by_exp.get(PROPOSED, next(iter(by_exp.values())))}
        for c in conds:
            r = f_rows.get(c, {})
            lines.append(f"| {c} | {fmt(r.get('ws_camera_mean_transmission', 1.0 if c == 'clear' else float('nan')))} | "
                         f"{fmt(r.get('ws_camera_rms_contrast_ratio', 1.0 if c == 'clear' else float('nan')))} | "
                         f"{fmt(r.get('ws_lidar_retained_ratio', float('nan')))} | "
                         f"{fmt(r.get('ws_lidar_backscatter_points', float('nan')), 0)} | "
                         f"{fmt(r.get('ws_radar_retained_ratio', float('nan')))} | {fmt(r.get('ws_radar_mean_loss_db', float('nan')), 2)} |")
        lines += ["", "## Reliability mechanisms (attention trust and feature-selection gates)", "",
                  "| Experiment | Condition | Trust camera | Trust LiDAR | Trust radar | FS LiDAR kept | FS radar kept |",
                  "|---|---|---|---|---|---|---|"]
        for exp, rs in by_exp.items():
            for r in rs:
                if any(k.startswith("trust_") for k in r):
                    lines.append(f"| {r['experiment_label']} | {r['condition']} | {fmt(r.get('trust_camera', float('nan')))} | "
                                 f"{fmt(r.get('trust_lidar', float('nan')))} | {fmt(r.get('trust_radar', float('nan')))} | "
                                 f"{fmt(r.get('fs_lidar_kept', float('nan')))} | {fmt(r.get('fs_radar_kept', float('nan')))} |")
    if meta.get("not_testable"):
        lines += ["", "## Not testable with the available data", ""] + [f"* **{k}**: {v}" for k, v in meta["not_testable"].items()]
    if meta.get("errors"):
        lines += ["", "## Errors (runs that failed — reported, not hidden)", ""] + \
                 [f"* {e.get('experiment')} / {e.get('condition', '-')}: {e['error']}" for e in meta["errors"]]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weather", default="all", help="all | fog | rain | comma list of conditions (configs/weather.yaml)")
    p.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    p.add_argument("--which", choices=["last", "best"], default="last", help="checkpoint (last = fair protocol)")
    p.add_argument("--split", default=None, help="evaluation split for --source real (default: config val split)")
    p.add_argument("--max-samples", type=int, default=None, help="limit frames (quick check)")
    p.add_argument("--force", action="store_true", help="re-run evaluations that already have a saved result")
    p.add_argument("--report-only", action="store_true", help="only rebuild reports/plots from saved runs")
    p.add_argument("--no-visuals", action="store_true")
    p.add_argument("--scene", default="scene-0916", help="frame shown in the visual comparisons")
    p.add_argument("--frame", type=int, default=20)
    p.add_argument("--out", default="outputs/adverse_weather")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. runtime.device=cpu")
    args = p.parse_args()
    setup_logging()
    if args.experiments == ["all"]:
        args.experiments = DEFAULT_EXPERIMENTS
    base_cfg = load_config(overrides=args.set)
    wcfg = load_weather_config()
    out_dir = resolve_path(args.out) / args.source
    command = "python main.py weather " + " ".join(sys.argv[1:])
    if args.report_only:
        args.force = False
    rows, meta = (run_synthetic if args.source == "synthetic" else run_real)(args, base_cfg, wcfg, out_dir)
    meta["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                           "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                           "generated": dt.datetime.now().isoformat(timespec="seconds"), "command": command}
    write_reports(rows, meta, out_dir, args.source, command)
    (out_dir / "manifest.json").write_text(json.dumps({k: v for k, v in meta.items()}, indent=1, default=str),
                                           encoding="utf-8")
    if rows and not args.no_visuals:
        from visualization.weather_view import report_plots, sample_figures
        report_plots(rows, out_dir / "plots", args.source)
        if args.source == "synthetic":
            sample_figures(base_cfg, wcfg, expand_conditions(args.weather, wcfg), args.scene, args.frame,
                           out_dir / "samples", [e for e in (PROPOSED, "A_lidar_only") if e in args.experiments])
    logger.info("Reports written to %s", out_dir)


if __name__ == "__main__":
    main()
