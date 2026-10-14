"""Figures for the adverse-weather study.

* :func:`report_plots` — metric charts from the result rows (one PNG per question).
* :func:`sample_figures` — one real frame under every condition: camera / LiDAR / radar as the
  network receives them, and detections vs ground truth (TP / false positive / missed) for the
  fused model and a single-sensor model.

Every synthetic panel carries a "SYNTHETIC" stamp; only the clear row is real sensor data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

logger = logging.getLogger(__name__)

# Fixed categorical order (dataviz reference palette, slots 1-6) — colour follows the experiment.
EXP_COLORS = {"A_lidar_only": "#2a78d6", "B_radar_only": "#eb6834", "C_lidar_radar_concat": "#1baf7a",
              "D_lidar_radar_attention": "#eda100", "E_lidar_radar_fs": "#e87ba4", "F_cam_lidar_radar_fs": "#008300"}
MOD_COLORS = {"camera": "#4a3aa7", "lidar": "#2a78d6", "radar": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TP_C, FP_C, FN_C, GT_C = "#eb6834", "#e34948", "#e34948", "#1baf7a"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b2")
    ax.tick_params(colors=INK2, labelsize=8)


def _stamp(fig, source: str) -> None:
    text = ("SYNTHETIC WEATHER — simulated corruptions of clear-weather nuScenes-mini validation frames"
            if source == "synthetic" else "REAL WEATHER — nuScenes scenes tagged from their original descriptions")
    fig.text(0.01, 0.005, text, fontsize=7.5, color="#b0281f" if source == "synthetic" else INK2, weight="bold")


# ------------------------------------------------------------------ report charts
def report_plots(rows: list[dict[str, Any]], out_dir: Path, source: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    exps = list(dict.fromkeys(r["experiment"] for r in rows))
    labels = {r["experiment"]: r["experiment_label"] for r in rows}
    get = {(r["experiment"], r["condition"]): r for r in rows}
    x = np.arange(len(conds))
    paths = []

    def lines(ax, key, title, pct=False):
        for e in exps:
            ys = [get[(e, c)][key] * (100 if pct else 1) if (e, c) in get else np.nan for c in conds]
            prop = e == "F_cam_lidar_radar_fs"
            ax.plot(x, ys, color=EXP_COLORS.get(e, INK2), lw=2.4 if prop else 1.4, marker="o", ms=5 if prop else 3.5,
                    label=labels[e] + ("  (proposed)" if prop else ""), zorder=3 if prop else 2)
        ax.set_xticks(x, [c.replace("_", "\n") for c in conds])
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _style(ax)

    # 1. detection performance by condition
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    lines(axs[0], "NDS", "NDS by weather (higher is better)")
    lines(axs[1], "mAP", "nuScenes mAP by weather (higher is better)")
    axs[1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _stamp(fig, source)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths.append(out_dir / "detection_by_weather.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    # 2. precision / recall
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    lines(axs[0], "precision", "Precision @ 2 m, score ≥ 0.3")
    lines(axs[1], "recall", "Recall @ 2 m, score ≥ 0.3")
    axs[1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _stamp(fig, source)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths.append(out_dir / "precision_recall_by_weather.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    # 3. relative degradation heatmap (diverging, neutral grey at 0)
    adv = [c for c in conds if c != "clear"]
    if adv:
        mat = np.array([[get[(e, c)]["rel_NDS"] * 100 if (e, c) in get else np.nan for c in adv] for e in exps])
        lim = max(10.0, float(np.nanmax(np.abs(mat)))) if np.isfinite(mat).any() else 10.0
        fig, ax = plt.subplots(figsize=(1.6 * len(adv) + 3.5, 0.55 * len(exps) + 1.6))
        im = ax.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.0f} %", ha="center", va="center", fontsize=8,
                            color="white" if abs(mat[i, j]) > 0.6 * lim else INK)
        ax.set_xticks(range(len(adv)), [c.replace("_", " ") for c in adv], fontsize=8)
        ax.set_yticks(range(len(exps)), [labels[e] for e in exps], fontsize=8)
        ax.set_title("NDS change relative to clear weather (same model)", fontsize=9.5, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
        _stamp(fig, source)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        paths.append(out_dir / "relative_nds_heatmap.png")
        fig.savefig(paths[-1], dpi=160)
        plt.close(fig)

    # 4. latency / FPS
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / max(len(exps), 1)
    for k, e in enumerate(exps):
        ys = [get[(e, c)]["FPS"] if (e, c) in get else np.nan for c in conds]
        ax.bar(x - 0.4 + w * (k + 0.5), ys, w * 0.9, color=EXP_COLORS.get(e, INK2), label=labels[e])
    ax.set_xticks(x, [c.replace("_", "\n") for c in conds])
    ax.set_title("Network throughput (FPS = 1000 / median latency, GPU; data loading excluded)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    _style(ax)
    _stamp(fig, source)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths.append(out_dir / "fps_by_weather.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    # 5. simulated sensor degradation
    if source == "synthetic":
        ref = "F_cam_lidar_radar_fs" if any(e == "F_cam_lidar_radar_fs" for e in exps) else exps[0]
        series = [("ws_camera_mean_transmission", "Camera: mean fog/rain transmission", MOD_COLORS["camera"]),
                  ("ws_lidar_retained_ratio", "LiDAR: share of points kept", MOD_COLORS["lidar"]),
                  ("ws_radar_retained_ratio", "Radar: share of points kept", MOD_COLORS["radar"])]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        for ax, (key, title, col) in zip(axs, series):
            ys = [get.get((ref, c), {}).get(key, 1.0 if c == "clear" else np.nan) for c in conds]
            ax.bar(x, ys, 0.6, color=col)
            for xi, y in zip(x, ys):
                if np.isfinite(y):
                    ax.text(xi, y + 0.02, f"{y:.2f}", ha="center", fontsize=7.5, color=INK2)
            ax.set_xticks(x, [c.replace("_", "\n") for c in conds], fontsize=7)
            ax.set_ylim(0, 1.12)
            ax.set_title(title, fontsize=9, loc="left")
            _style(ax)
        _stamp(fig, source)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        paths.append(out_dir / "sensor_degradation.png")
        fig.savefig(paths[-1], dpi=160)
        plt.close(fig)

    # 6. attention-fusion trust per sensor
    att = [e for e in exps if any(k.startswith("trust_") for k in get.get((e, conds[0]), {}))]
    if att:
        fig, axs = plt.subplots(1, len(att), figsize=(4 * len(att), 3.4), sharey=True, squeeze=False)
        for ax, e in zip(axs[0], att):
            bottom = np.zeros(len(conds))
            for m in ("camera", "lidar", "radar"):
                ys = np.array([get.get((e, c), {}).get(f"trust_{m}", 0.0) for c in conds])
                if ys.any():
                    ax.bar(x, ys, 0.6, bottom=bottom, color=MOD_COLORS[m], label=m, edgecolor="white", lw=1)
                    bottom += ys
            ax.set_xticks(x, [c.replace("_", "\n") for c in conds], fontsize=7)
            ax.set_title(f"{labels[e]}\nmean attention weight per sensor", fontsize=8.5, loc="left")
            _style(ax)
        axs[0][-1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        _stamp(fig, source)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        paths.append(out_dir / "fusion_trust_by_weather.png")
        fig.savefig(paths[-1], dpi=160)
        plt.close(fig)
    return paths


# ------------------------------------------------------------------ sample frames
def _box_poly(box: np.ndarray) -> np.ndarray:
    from utils.geometry import box_corners_bev
    return box_corners_bev(box[None].astype(np.float64))[0]


def _match(pred_boxes, pred_labels, gt_boxes, gt_labels, dist=2.0):
    """Greedy (score-sorted input) class-aware centre matching -> TP mask for preds, matched mask for GT."""
    tp = np.zeros(len(pred_boxes), bool)
    taken = np.zeros(len(gt_boxes), bool)
    for i in range(len(pred_boxes)):
        best, best_d = -1, dist
        for j in range(len(gt_boxes)):
            if not taken[j] and gt_labels[j] == pred_labels[i]:
                d = float(np.hypot(*(gt_boxes[j, :2] - pred_boxes[i, :2])))
                if d < best_d:
                    best, best_d = j, d
        if best >= 0:
            tp[i], taken[best] = True, True
    return tp, taken


def _bev_panel(ax, lidar, radar, gt_boxes, gt_labels, det, classes, title, score_th=0.3, extent=50.0) -> str:
    ax.set_facecolor("#111317")
    if lidar is not None and len(lidar):
        ax.scatter(lidar[:, 1], lidar[:, 0], s=0.15, c=np.clip(lidar[:, 2], -2, 3), cmap="Greys_r", vmin=-2.5, vmax=3,
                   lw=0, rasterized=True)
    if radar is not None and len(radar):
        ax.scatter(radar[:, 1], radar[:, 0], s=4, c="#e87ba4", lw=0)
    counts = ""
    if det is not None:
        keep = det["scores"] >= score_th
        order = np.argsort(-det["scores"][keep])
        pb, ps, pl = det["boxes"][keep][order], det["scores"][keep][order], det["labels"][keep][order]
        tp, matched = _match(pb, pl, gt_boxes, gt_labels)
        for j, b in enumerate(gt_boxes):
            poly = _box_poly(b)
            ax.fill(poly[:, 1], poly[:, 0], fill=False, ec=GT_C if matched[j] else FN_C, lw=1.0 if matched[j] else 1.6,
                    ls="-" if matched[j] else (0, (2, 1)))
        for i, b in enumerate(pb):
            poly = _box_poly(b)
            ax.fill(poly[:, 1], poly[:, 0], fill=False, ec=TP_C if tp[i] else FP_C, lw=1.2)
            if abs(b[0]) < extent and abs(b[1]) < extent:
                ax.text(b[1], b[0] + 1.6, f"{classes[int(pl[i])][:4]} {ps[i]:.2f}", fontsize=4.2,
                        color=TP_C if tp[i] else FP_C, ha="center")
        counts = f"TP {int(tp.sum())} · FP {int((~tp).sum())} · missed {int((~matched).sum())}"
    else:
        for b in gt_boxes:
            poly = _box_poly(b)
            ax.fill(poly[:, 1], poly[:, 0], fill=False, ec=GT_C, lw=1.0)
    ax.add_patch(plt.Rectangle((-0.925, -1.0), 1.85, 4.08, color="#1baf7a"))
    ax.set_xlim(extent, -extent)  # y (left) to the left of the plot
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title + (f"\n{counts}" if counts else ""), fontsize=7.5, color=INK, loc="left")
    return counts


def _full_res_camera(info: dict, root: Path, spec: dict | None, key_lidar: np.ndarray, index: int, cam: str) -> np.ndarray:
    """CAM image at full resolution with the same weather model (display only)."""
    from datasets.camera_loader import IMAGENET_MEAN, IMAGENET_STD
    from datasets.weather import SyntheticWeather
    rec = info["cams"][cam]
    with Image.open(root / rec["path"]) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    if spec is None:
        return rgb
    h, w = rgb.shape[:2]
    cams = {"images": ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32),
            "intrinsics": np.asarray(rec["intrinsic"], np.float32)[None], "post_rots": np.eye(3, dtype=np.float32)[None],
            "post_trans": np.zeros((1, 3), np.float32), "cam2ref": np.asarray(rec["sensor2ref"], np.float32)[None],
            "cam_valid": np.array([True])}
    imgs, _ = SyntheticWeather(spec).camera(cams, key_lidar, (h, w), index)
    return np.clip(imgs[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def sample_figures(base_cfg, wcfg: dict, conditions: list[str], scene: str, frame: int, out_dir: Path,
                   experiments: list[str]) -> list[Path]:
    import torch

    from datasets.info_builder import load_infos
    from datasets.lidar_loader import LidarLoader
    from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch
    from datasets.weather import synthetic_spec
    from evaluation.ablation import checkpoint_for
    from models.fusion_model import build_model
    from utils.bev_grid import BEVGrid
    from utils.checkpoint import load_checkpoint
    from utils.config import Config
    from utils.device import get_device, move_to_device

    out_dir.mkdir(parents=True, exist_ok=True)
    split = base_cfg.dataset.val_split
    infos = load_infos(base_cfg, split)
    sel = [k for k, i in enumerate(infos) if i["scene_name"] == scene]
    if not sel:
        logger.warning("Scene %s is not in the evaluation split %s - no sample figures", scene, split)
        return []
    index = sel[min(frame, len(sel) - 1)]
    info = infos[index]
    root = Path(base_cfg.dataset.root)
    key_lidar = LidarLoader(root, BEVGrid.from_config(base_cfg.bev)).load_keyframe(info)
    specs = {c: (None if c == "clear" else synthetic_spec(c, wcfg)) for c in conditions}
    tag = f"{scene}_frame{frame:03d}"
    paths = []

    # --- per-condition sensor data + model predictions
    models = {}
    for exp in experiments:
        path = checkpoint_for(base_cfg, exp, "joint", "last")
        if path is not None:
            ck = load_checkpoint(path)
            models[exp] = ck
    data, preds = {}, {e: {} for e in models}
    for c in conditions:
        cfg = Config(base_cfg.to_dict())  # all sensors, as configured for this machine (display data)
        if specs[c]:
            cfg.set_path("weather.synthetic", specs[c])
        data[c] = NuScenesFusionDataset(cfg, split, train=False, infos=infos, load_all_sensors=True)[index]
        for exp, ck in models.items():
            cfg = Config(ck["config"])
            if specs[c]:
                cfg.set_path("weather.synthetic", specs[c])
            item = NuScenesFusionDataset(cfg, split, train=False, infos=infos, load_all_sensors=True)[index]
            device = get_device(cfg.runtime.device)
            model = build_model(cfg).to(device).eval()
            model.load_state_dict(ck["model"])
            with torch.no_grad():
                out = model(move_to_device(collate_batch([item]), device))
                det = model.decode(out, score_threshold=0.1)[0]
            preds[exp][c] = {k: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for k, v in det.items()}
            del model
    classes = list(base_cfg.classes)

    # --- figure 1: what each sensor delivers
    n = len(conditions)
    fig, axs = plt.subplots(n, 3, figsize=(15, 3.1 * n), gridspec_kw={"width_ratios": [1.9, 1, 1]})
    axs = np.atleast_2d(axs)
    for r, c in enumerate(conditions):
        label = wcfg["conditions"][c]["label"]
        stamp = "REAL sensor data (clear)" if c == "clear" else "SYNTHETIC"
        img = _full_res_camera(info, root, specs[c], key_lidar, index, "CAM_FRONT")
        axs[r, 0].imshow(img)
        axs[r, 0].set_axis_off()
        axs[r, 0].set_title(f"{label} — CAM_FRONT", fontsize=8.5, loc="left")
        axs[r, 0].text(12, 40, stamp, color="white", fontsize=8, weight="bold",
                       bbox=dict(fc="#b0281f" if c != "clear" else "#2a78d6", ec="none", pad=2))
        it = data[c]
        ws = it["meta"].get("weather", {})
        lid, rad = it.get("raw_lidar_points"), it.get("raw_radar_points")
        _bev_panel(axs[r, 1], lid, None, it["gt_boxes"], it["gt_labels"], None, classes,
                   f"LiDAR BEV ({0 if lid is None else len(lid):,} pts"
                   + (f", {100 * ws.get('lidar_retained_ratio', 1):.0f} % kept)" if c != "clear" else ")"))
        _bev_panel(axs[r, 2], None, rad, it["gt_boxes"], it["gt_labels"], None, classes,
                   f"Radar BEV ({0 if rad is None else len(rad)} pts"
                   + (f", −{ws.get('radar_mean_loss_db', 0):.2f} dB rain loss)" if "rain" in c
                      else (", unchanged: fog ≪ 1 dB at 77 GHz)" if "fog" in c else ")")))
    fig.suptitle(f"Sensor inputs under weather — {scene}, key-frame {frame} (validation scene, model never trained on it)",
                 fontsize=10, x=0.01, ha="left")
    _stamp(fig, "synthetic")
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    paths.append(out_dir / f"sensors_{tag}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    # --- figure 2: detections vs GT, fused vs single sensor
    if models:
        exps = list(models)
        fig, axs = plt.subplots(len(exps), n, figsize=(3.3 * n, 3.9 * len(exps)), squeeze=False)
        for i, exp in enumerate(exps):
            for j, c in enumerate(conditions):
                it = data[c]
                lid = it.get("raw_lidar_points")
                _bev_panel(axs[i, j], lid, it.get("raw_radar_points"), it["gt_boxes"], it["gt_labels"], preds[exp][c],
                           classes, f"{exp.split('_')[0]} ({'fused' if exp.startswith(('C', 'D', 'E', 'F')) else 'single sensor'}) · "
                           f"{c}{'' if c == 'clear' else ' (SYNTHETIC)'}")
        fig.suptitle(f"Detections (score ≥ 0.3) vs ground truth — {scene}, key-frame {frame}\n"
                     "orange = correct detection · red solid = false positive · red dashed = missed object · "
                     "green = detected object (GT) · grey = LiDAR · pink = radar",
                     fontsize=9, x=0.01, ha="left")
        _stamp(fig, "synthetic")
        fig.tight_layout(rect=(0, 0.02, 1, 0.94), h_pad=2.5)
        paths.append(out_dir / f"detections_{tag}.png")
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)
    return paths
