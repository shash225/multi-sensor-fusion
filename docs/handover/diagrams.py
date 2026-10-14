"""Diagrams for the handover document (matplotlib → PNG). Names and shapes mirror the code.

Layouts are sized so that, once scaled to the A4 text width, box titles print at ≈ 7 pt and box text at ≈ 5.5 pt or more.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "build"
INK, INK2 = "#1b1f27", "#4a5160"
FS = 1.3  # global font scale
PAL = {  # (fill, edge)
    "data": ("#eef1f6", "#8a94a6"), "load": ("#e6eefb", "#5b86c5"), "enc": ("#dfeafc", "#2a78d6"),
    "fs": ("#fde9dd", "#e0702e"), "fuse": ("#dcf3ec", "#1b9a74"), "head": ("#ece8fb", "#6a55c9"),
    "plan": ("#e3f4df", "#3f8f2f"), "app": ("#f1f1ee", "#7d7b72"), "loss": ("#fbe7ea", "#c23b52"),
    "start": ("#1b1f27", "#1b1f27"), "dec": ("#fff4d6", "#c89212"),
}
plt.rcParams["font.family"] = "DejaVu Sans"


def _ax(w: float, h: float):
    fig = plt.figure(figsize=(w * 0.62, h * 0.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size, **kw):
    kw.setdefault("color", INK2)
    ax.text(x, y, s, fontsize=size * FS, **kw)


def box(ax, x, y, w, h, title, sub="", kind="enc", ts=8.6, ss=6.6, round_=0.12):
    fc, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec=ec, lw=1.3))
    color = "white" if kind == "start" else INK
    if sub:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=ts * FS, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=ss * FS,
                color=INK2 if kind != "start" else "white", linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts * FS, weight="bold", color=color)
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, label, ts=7.2):
    fc, ec = PAL["dec"]
    ax.add_patch(plt.Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], fc=fc, ec=ec, lw=1.3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=ts * FS, color=INK, linespacing=1.2)


def arrow(ax, p, q, label="", color="#5d6574", ts=6.4, rad=0.0, lw=1.2, offset=(0.08, 0.0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1]
        ax.text(mx, my, label, fontsize=ts * FS, color=INK2, ha="left", va="center",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def elbow(ax, pts, color="#5d6574", lw=1.2):
    """Orthogonal connector through `pts`; arrow head on the last segment (keeps loops clear of boxes)."""
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 shrinkA=0, shrinkB=1))


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def left(b):
    return (b[0], b[1] + b[3] / 2)


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ architecture
def architecture() -> Path:
    fig, ax = _ax(17, 17.0)
    ds = box(ax, 2.0, 15.5, 13, 1.1, "nuScenes dataset  (dataset.root, dataset.version)",
             "nuscenes_db.NuScenesDB → info_builder.load_infos() → cached per-sample infos", "data")
    cam = box(ax, 0.3, 13.2, 5.2, 1.6, "CameraLoader", "camera_loader.py · 6 images\nK, post_rot/tran, cam2ref", "load")
    lid = box(ax, 5.9, 13.2, 5.2, 1.6, "LidarLoader", "lidar_loader.py · sweeps\nsensor→ref, time-lag → (P, 5)", "load")
    rad = box(ax, 11.5, 13.2, 5.2, 1.6, "RadarLoader", "radar_loader.py · PCD, filters\npos + velocity → ref → (Q, 10)", "load")
    dset = box(ax, 0.3, 11.6, 16.4, 1.1, "NuScenesFusionDataset.__getitem__  +  collate_batch",
               "BEV augmentation (bda) · depth GT · CenterPoint targets · fg heatmap · ego future · future occupancy", "data")
    for b in (cam, lid, rad):
        arrow(ax, (b[0] + b[2] / 2, ds[1]), top(b))
        arrow(ax, bottom(b), (b[0] + b[2] / 2, dset[1] + dset[3]))
    e_cam = box(ax, 0.3, 9.3, 5.2, 1.7, "CameraBEVEncoder", "camera_bev/lss.py · LSS\nResNet18 → depth → lift → splat\nF_cam (B, 64, 200, 200)", "enc")
    e_lid = box(ax, 5.9, 9.3, 5.2, 1.7, "LidarEncoder", "lidar_encoder/pillar_encoder.py\npillar VFE + raster(4)\nF_lidar (B, 64, 200, 200)", "enc")
    e_rad = box(ax, 11.5, 9.3, 5.2, 1.7, "RadarEncoder", "radar_encoder/radar_encoder.py\npillar VFE + raster(5), dilated\nF_radar (B, 64, 200, 200)", "enc")
    for b in (e_cam, e_lid, e_rad):
        arrow(ax, (b[0] + b[2] / 2, dset[1]), top(b))
    fs = box(ax, 5.9, 7.0, 10.8, 1.6, "CrossModalFeatureSelection  (research component)",
             "feature_selection/selection.py · channel gates (B, 64) + spatial gates (B, 1, 200, 200)\n"
             "F̃_m = F_m ⊙ c_m ⊙ s_m,  m ∈ {lidar, radar}", "fs")
    arrow(ax, bottom(e_lid), (e_lid[0] + e_lid[2] / 2, fs[1] + fs[3]))
    arrow(ax, bottom(e_rad), (e_rad[0] + e_rad[2] / 2, fs[1] + fs[3]))
    arrow(ax, (e_cam[0] + e_cam[2] * 0.8, e_cam[1]), left(fs), "guide", rad=0.15, offset=(-0.7, -0.2))
    fu = box(ax, 0.3, 4.9, 16.4, 1.4, "Fusion  (registry: concat | attention_gated)",
             "fusion/fusion_methods.py · per-cell softmax over sensors, Σ a_m·P_m ⊕ P_1..P_M → ResBlock → CBAM → (B, 128, 200, 200)", "fuse")
    arrow(ax, bottom(e_cam), (e_cam[0] + e_cam[2] / 2, fu[1] + fu[3]), "F_cam")
    arrow(ax, bottom(fs), (fs[0] + fs[2] / 2, fu[1] + fu[3]), "F̃_lidar, F̃_radar")
    bb = box(ax, 3.5, 3.3, 10.0, 1.1, "SECONDFPN BEV backbone", "fusion/bev_backbone.py · → (B, 256, 100, 100)", "fuse")
    arrow(ax, bottom(fu), top(bb))
    det = box(ax, 0.3, 1.5, 7.6, 1.3, "CenterHead + decode_detections", "detection/ · heatmap, reg, size, yaw, velocity\n→ boxes (K, 9)", "head")
    ph = box(ax, 9.1, 1.5, 7.6, 1.3, "PlanningHead", "trajectory/planning_head.py\nego query × BEV tokens → GRU → (B, 6, 2)", "head")
    arrow(ax, (bb[0] + 2.5, bb[1]), top(det))
    arrow(ax, (bb[0] + 7.5, bb[1]), top(ph))
    app = box(ax, 0.3, 0.1, 16.4, 1.0, "Tracking · Planning · Visualization · Application",
              "tracking.py → trajectory_planner.py → inference/pipeline.py → app/streamlit_app.py, scripts/", "plan")
    arrow(ax, bottom(det), (det[0] + det[2] / 2, app[1] + app[3]))
    arrow(ax, bottom(ph), (ph[0] + ph[2] / 2, app[1] + app[3]))
    return save(fig, "architecture.png")


# ------------------------------------------------------------------------------ data flow
def dataflow() -> Path:
    steps = [
        ("data", "Index → info", "FusionPipeline.run(index) · infos from load_infos()", "paths, sensor2ref, GT, ego state"),
        ("data", "BEV augmentation", "NuScenesFusionDataset._sample_bda() (identity at inference)", "bda (4, 4)"),
        ("load", "Cameras", "CameraLoader.load(), image_aug_matrices()", "images (6, 3, 128, 352), K, post_rot, cam2ref"),
        ("load", "LiDAR", "LidarLoader.load(): sweeps → ref, time-lag, filter", "points (P, 5)"),
        ("load", "Radar", "RadarLoader.load(): parse, filter, rotate velocities", "points (Q, 10)"),
        ("data", "Batching", "collate_batch(): stack arrays, batch-index column", "batch dict (B = 1)"),
        ("enc", "Camera → BEV", "CameraBEVEncoder.forward()", "F_cam (1, 64, 200, 200)"),
        ("enc", "LiDAR / radar", "LidarEncoder.forward(), RadarEncoder.forward()", "F_lidar, F_radar (1, 64, 200, 200)"),
        ("fs", "Feature selection", "CrossModalFeatureSelection.forward()", "F̃_lidar, F̃_radar + gates"),
        ("fuse", "Fusion + backbone", "AttentionGatedFusion → SECONDFPN", "(1, 128, 200, 200) → (1, 256, 100, 100)"),
        ("head", "Detection", "CenterHead → decode_detections()", "boxes (K, 9), scores, labels"),
        ("head", "Learned trajectory", "PlanningHead.forward()", "waypoints (1, 6, 2)"),
        ("plan", "Tracking", "MultiObjectTracker.update() (score ≥ 0.3)", "track ids, histories"),
        ("plan", "Planning", "BaselinePlanner.plan() (or learned / hybrid)", "waypoints (6, 2), collision flag"),
        ("loss", "Frame metrics", "DetectionEvaluator, sample_planning_metrics", "NDS, mAP, ADE, FDE, collision"),
        ("app", "Visualization", "app/ui/panels.py → visualization/*", "images, tables, KPIs"),
    ]
    h = 0.82
    W, H = 15, len(steps) * (h + 0.2) + 0.4
    fig, ax = _ax(W, H)
    y = H - 0.3 - h
    prev = None
    for kind, title, sub, shape in steps:
        b = box(ax, 0.2, y, 3.4, h, title, "", kind, ts=7.0)
        text(ax, 3.9, y + h * 0.66, sub, 6.6, color=INK, va="center")
        text(ax, 3.9, y + h * 0.24, "→ " + shape, 6.4, color="#1c5cab", va="center", family="DejaVu Sans Mono")
        if prev is not None:
            arrow(ax, bottom(prev), top(b))
        prev = b
        y -= h + 0.2
    return save(fig, "dataflow.png")


# ------------------------------------------------------------------------------ dashboard execution flow
def exec_dashboard() -> Path:
    fig, ax = _ax(15, 18.5)
    cx, w = 4.6, 8.6

    def node(y, title, sub="", kind="app", hh=1.0):
        return box(ax, cx - w / 2, y, w, hh, title, sub, kind, ts=7.4, ss=6.0)

    n0 = box(ax, cx - 1.5, 17.5, 3.0, 0.7, "START", "", "start", ts=7.6)
    n1 = node(16.1, "python app/streamlit_app.py", "not inside Streamlit → re-launch via streamlit.web.cli")
    n2 = node(14.7, "Page setup + session state", "set_page_config · CSS · ss_default(...)")
    n3 = node(13.3, "Sidebar controls", "version · device · mode · method · planner · speed")
    n4 = node(11.9, "get_pipeline()  [cache_resource]", "FusionPipeline: infos · dataset · model", "data")
    d1y = 10.4
    diamond(ax, cx, d1y, 4.6, 1.3, "mode 3 and\ncheckpoint?")
    n5 = box(ax, 9.4, d1y - 0.5, 4.8, 1.0, "eff_mode = model", "banner MODEL PREDICTION", "head", ts=7.2, ss=6.0)
    n6 = node(8.6, "eff_mode = ground_truth / dataset", "banner GROUND TRUTH / DATASET", "dec")
    n7 = node(7.2, "Scene + key-frame selection", "callbacks cb_prev / next / play / pause / reset")
    d2y = 5.7
    diamond(ax, cx, d2y, 4.6, 1.3, "frame result\ncached?")
    n8 = box(ax, 9.4, d2y - 0.5, 4.8, 1.0, "pipe.run(index, …)", "stage callbacks tick the strip", "enc", ts=7.2, ss=6.0)
    n9 = node(3.9, "Render panels", "camera · fusion tiles · BEV · trajectory · tables")
    n10 = node(2.5, "Optional sections", "explain FS · compare · ablation · curves")
    d3y = 1.0
    diamond(ax, cx, d3y, 3.6, 1.1, "playing?")
    n11 = box(ax, 9.4, 0.5, 4.8, 1.0, "sleep → _advance → st.rerun()", "", "app", ts=7.0)
    arrow(ax, bottom(n0), top(n1))
    for a, b in ((n1, n2), (n2, n3), (n3, n4)):
        arrow(ax, bottom(a), top(b))
    arrow(ax, bottom(n4), (cx, d1y + 0.65))
    arrow(ax, (cx + 2.3, d1y), left(n5), "yes", offset=(-0.4, 0.25))
    arrow(ax, (cx, d1y - 0.65), top(n6), "no", offset=(0.15, 0))
    arrow(ax, bottom(n6), top(n7))
    xr = n7[0] + n7[2]
    elbow(ax, [bottom(n5), (bottom(n5)[0], 7.95), (xr, 7.95)])
    arrow(ax, bottom(n7), (cx, d2y + 0.65))
    arrow(ax, (cx + 2.3, d2y), left(n8), "no", offset=(-0.4, 0.25))
    arrow(ax, (cx, d2y - 0.65), top(n9), "yes", offset=(0.15, 0))
    elbow(ax, [bottom(n8), (bottom(n8)[0], right(n9)[1]), right(n9)])
    arrow(ax, bottom(n9), top(n10))
    arrow(ax, bottom(n10), (cx, d3y + 0.55))
    arrow(ax, (cx + 1.8, d3y), left(n11), "yes", offset=(-0.3, 0.25))
    text(ax, cx - 2.0, d3y - 0.75, "no → wait for user interaction", 6.0, ha="center", va="center")
    # loop back through the right margin, clear of the right-hand column
    xm = n11[0] + n11[2] + 0.4
    elbow(ax, [right(n11), (xm, right(n11)[1]), (xm, 7.45), (xr, 7.45)])
    text(ax, xm - 0.15, 3.2, "next frame", 6.2, ha="right", va="center")
    return save(fig, "exec_dashboard.png")


# ------------------------------------------------------------------------------ training flow
def exec_training() -> Path:
    fig, ax = _ax(15, 15.0)
    cx, w = 5.4, 10.4

    def node(y, title, sub="", kind="app", hh=1.0):
        return box(ax, cx - w / 2, y, w, hh, title, sub, kind, ts=7.4, ss=6.0)

    n0 = box(ax, cx - 1.5, 14.0, 3.0, 0.7, "START", "", "start", ts=7.6)
    n1 = node(12.6, "python main.py train --stage S --experiment E", "main.py → runpy → training/cli.py: run()")
    n2 = node(11.2, "build_stage_config(E, S, overrides)", "load_config (base → hardware → experiment → local → CLI) + stage", "data")
    n3 = node(9.8, "Trainer.__init__", "logging · seed · device · AMP self-test · DataLoaders", "data")
    n4 = node(8.4, "build_model(cfg)", "grad checkpoint · init_from earlier stages · freeze", "enc")
    n5 = node(7.0, "AdamW + warm-up/cosine + GradScaler", "lr = train.lr × stage.lr_scale", "app")
    n6 = node(5.3, "train_epoch()", "forward → compute_losses(active) → backward ÷ accum\nunscale → clip → step → scheduler", "loss", hh=1.3)
    n7 = node(3.8, "evaluate_model(val)", "NDS / mAP · ADE / FDE / collision · FS stats · latency", "head")
    n8 = node(2.4, "save best.pt (if improved) + last.pt", "config embedded · history.jsonl", "plan")
    dy = 1.35
    diamond(ax, cx, dy, 3.8, 1.0, "more epochs?")
    n9 = box(ax, cx - 1.5, 0.08, 3.0, 0.5, "END", "", "start", ts=7.6)
    arrow(ax, bottom(n0), top(n1))
    for a, b in ((n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6), (n6, n7), (n7, n8)):
        arrow(ax, bottom(a), top(b))
    arrow(ax, bottom(n8), (cx, dy + 0.5))
    # epoch loop through the right margin so it does not run along the box edges
    xm = cx + w / 2 + 0.6
    elbow(ax, [(cx + 1.9, dy), (xm, dy), (xm, right(n6)[1]), right(n6)])
    text(ax, cx + 2.1, dy + 0.12, "yes", 6.4)
    arrow(ax, (cx, dy - 0.5), top(n9))
    text(ax, cx + 0.15, dy - 0.68, "no", 6.4, va="center")
    text(ax, 11.0, 13.1, "--stage curriculum:\ncamera_bev → lidar → radar\n→ fusion → detection\n→ trajectory → end_to_end\n(one Trainer.fit() each)",
         6.4, va="top")
    return save(fig, "exec_training.png")


# ------------------------------------------------------------------------------ coordinate frames
def frames() -> Path:
    fig, ax = _ax(12, 11.6)
    text(ax, 0.1, 11.15, "sensor2ref = inv(ref_ego2global) · ego2global(t_sensor) · sensor2ego", 6.6, color=INK,
         family="DejaVu Sans Mono")
    text(ax, 0.1, 10.75, "computed per sweep in datasets/info_builder.py: _sensor_record", 6.0)
    r1 = [box(ax, 0.1 + i * 4.2, 8.4, 3.1, 1.9, t, s, k, ts=7.4, ss=6.0) for i, (t, s, k) in enumerate([
        ("Sensor frame", "LIDAR_TOP / RADAR_* / CAM_*\ncamera: x right, y down,\nz forward", "load"),
        ("Ego frame @ t_sensor", "x forward · y left · z up\norigin: rear-axle centre", "data"),
        ("Global frame", "metric map frame\nper location", "data")])]
    r2 = [box(ax, 0.1 + i * 4.2, 5.6, 3.1, 1.9, t, s, k, ts=7.4, ss=6.0) for i, (t, s, k) in enumerate([
        ("Reference ego frame", "ego @ LIDAR_TOP key-frame\nmodel inputs + GT live here", "enc"),
        ("Augmented BEV frame", "bda: flip / rotate / scale\n(training; identity\notherwise)", "fs"),
        ("BEV grid cell (iy, ix)", "ix = ⌊(x − x_min) / r⌋\niy = ⌊(y − y_min) / r⌋\n200 × 200, r = 0.5 m", "fuse")])]

    def gap_label(row, i, label):
        a, b = row[i], row[i + 1]
        cx = (a[0] + a[2] + b[0]) / 2
        text(ax, cx, a[1] + a[3] / 2 + 0.12, label, 5.6, ha="center", va="bottom")

    arrow(ax, right(r1[0]), left(r1[1]))
    arrow(ax, right(r1[1]), left(r1[2]))
    gap_label(r1, 0, "sensor\n2ego")
    gap_label(r1, 1, "ego2global\n(t_s)")
    arrow(ax, bottom(r1[2]), (r1[2][0] + r1[2][2] / 2, r1[2][1] - 0.4))
    ax.add_patch(FancyArrowPatch((r1[2][0] + r1[2][2] / 2, r1[2][1] - 0.4), top(r2[0]), arrowstyle="-|>", mutation_scale=11,
                                 color="#5d6574", lw=1.2, connectionstyle="arc3,rad=0.0"))
    text(ax, 6.0, 8.12, "inv(ego2global(t_ref))", 5.8, ha="center", va="center")
    arrow(ax, right(r2[0]), left(r2[1]))
    arrow(ax, right(r2[1]), left(r2[2]))
    gap_label(r2, 0, "bda")
    gap_label(r2, 1, "quantise")
    text(ax, 0.1, 4.8, "Camera projection (camera_view.project_points, camera_loader.lidar_depth_maps):", 6.4)
    r3 = [box(ax, 0.1 + i * 3.0, 2.8, 2.7, 1.6, t, s, k, ts=7.0, ss=5.8) for i, (t, s, k) in enumerate([
        ("Camera point", "p_c = inv(cam2ref)·p_ref", "load"), ("Pixel", "(u, v) = (K p_c)_xy / z", "data"),
        ("Augmented pixel", "post_rot·(u, v, 1)\n+ post_tran", "fs"), ("Feature cell", "⌊u/16⌋, ⌊v/16⌋", "fuse")])]
    for a, b in zip(r3[:-1], r3[1:]):
        arrow(ax, right(a), left(b))
    text(ax, 0.1, 2.1, "LSS lift (lss.get_geometry) runs it backwards:", 6.4)
    text(ax, 0.1, 1.55, "(u', v', d) = post_rot⁻¹((u, v, d) − post_tran)", 6.4, color=INK, family="DejaVu Sans Mono")
    text(ax, 0.1, 1.05, "p_ref = R_cam2ref · K⁻¹ · [u'd, v'd, d]ᵀ + t_cam2ref ;  p_bev = bda · p_ref", 6.4, color=INK,
         family="DejaVu Sans Mono")
    # ego axes sketch (bottom right, clear of the boxes and formulas)
    ox, oy = 11.4, 0.35
    ax.add_patch(FancyBboxPatch((ox - 0.25, oy - 0.1), 0.5, 1.1, boxstyle="round,pad=0,rounding_size=0.1", fc="#dcf3ec", ec="#1b9a74"))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox, oy + 1.7), arrowstyle="-|>", mutation_scale=11, color="#c23b52", lw=1.5))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox - 1.1, oy), arrowstyle="-|>", mutation_scale=11, color="#2a78d6", lw=1.5))
    text(ax, ox - 1.0, oy + 1.95, "x forward", 5.8, color="#c23b52")
    text(ax, ox - 1.7, oy + 0.15, "y left", 5.8, color="#2a78d6")
    text(ax, 8.9, 0.1, "ego origin: rear axle · z up", 5.4)
    return save(fig, "frames.png")


# ------------------------------------------------------------------------------ LSS
def lss() -> Path:
    fig, ax = _ax(12, 8.9)
    row1 = [("Images", "(B, 6, 3, 128, 352)\nnormalised, augmented", "load"),
            ("ImageBackbone", "ResNet18 L3 ⊕ up(L4)\n→ (B·6, 256, 8, 22)", "enc"),
            ("DepthNet", "camera-aware gates\ndepth (B·6, 59, 8, 22)\ncontext (B·6, 64, 8, 22)", "enc"),
            ("Lift", "softmax(depth) ⊗ context\n(B, 6, 59, 8, 22, 64)", "fs")]
    row2 = [("Geometry", "frustum → ref frame\n(B, 6, 59, 8, 22, 3)", "data"),
            ("Splat", "index_add_ into cells\n(B, 64, 200, 200)", "fuse"),
            ("BEV refine", "ConvBNReLU + 2 ResBlocks\nF_cam (B, 64, 200, 200)", "fuse")]
    # three bands: supervision (top) · image path (middle) · geometry + splat (bottom); every arrow runs in a gap
    text(ax, 0.1, 8.35, "Camera → BEV (models/camera_bev/lss.py)", 7.4, color=INK, weight="bold")
    text(ax, 0.1, 7.85, "B = batch · 6 cameras · D = 59 depth bins", 6.0)
    text(ax, 0.1, 7.4, "feature stride 16 → 8 × 22 cells per image", 6.0)
    aux = box(ax, 6.1, 6.7, 5.7, 1.9, "Depth supervision (training)", "lidar_depth_maps(): nearest LiDAR point per feature cell\ndepth_loss(): BCE(softmax, one-hot bin)", "loss", ts=7.2, ss=5.8)
    b1 = [box(ax, 0.1 + i * 3.0, 3.7, 2.7, 2.0, t, s, k, ts=7.4, ss=5.8) for i, (t, s, k) in enumerate(row1)]
    cal = box(ax, 0.1, 0.3, 2.7, 2.0, "Calibration", "K, post_rot, post_tran\ncam2ref (extr. + ego)\nbda", "data", ts=7.4, ss=5.8)
    b2 = [box(ax, 3.1 + i * 3.0, 0.3, 2.7, 2.0, t, s, k, ts=7.4, ss=5.8) for i, (t, s, k) in enumerate(row2)]
    for a, b in zip(b1[:-1], b1[1:]):
        arrow(ax, right(a), left(b))
    arrow(ax, top(b1[2]), (top(b1[2])[0], aux[1]))
    text(ax, top(b1[2])[0] + 0.12, 6.1, "depth logits", 5.6)
    arrow(ax, right(cal), left(b2[0]))
    for a, b in zip(b2[:-1], b2[1:]):
        arrow(ax, right(a), left(b))
    arrow(ax, (cal[0] + 1.9, cal[1] + cal[3]), (b1[2][0] + 0.4, b1[2][1]))
    text(ax, 4.3, 2.62, "camera_params() (16-d)", 5.6)
    arrow(ax, (b1[3][0] + 0.5, b1[3][1]), (b2[1][0] + b2[1][2] - 0.5, b2[1][1] + b2[1][3]))
    text(ax, 9.5, 3.05, "features", 5.6)
    return save(fig, "lss.png")


# ------------------------------------------------------------------------------ feature selection
def feature_selection() -> Path:
    fig, ax = _ax(15, 11.0)
    fl = box(ax, 0.2, 8.6, 2.8, 1.3, "F_lidar", "(B, 64, H, W)", "enc")
    fr = box(ax, 0.2, 6.7, 2.8, 1.3, "F_radar", "(B, 64, H, W)", "enc")
    fc = box(ax, 0.2, 4.8, 2.8, 1.3, "F_cam (guide)", "(B, 64, H, W)", "enc")
    ch = box(ax, 4.0, 7.5, 4.6, 2.2, "Channel branch", "[mean, max] pooling → (B, 384)\nLinear(384→96) · ReLU\nLinear(96→128) → (B, 64) per sensor", "fs", ts=7.4, ss=5.9)
    sp = box(ax, 4.0, 4.0, 4.6, 2.8, "Spatial branch (CBAM)", "per input: 1×1 → 8 ch\n+ channel mean + max = 10 ch\nconcat (B, 30, H, W)\n7×7 CBR(32) → 3×3 CBR(32)\n→ 3×3 conv → (B, 2, H, W)", "fs", ts=7.4, ss=5.9)
    for b in (fl, fr, fc):
        arrow(ax, right(b), left(ch))
        arrow(ax, right(b), left(sp))
    g = box(ax, 9.5, 5.6, 2.8, 2.8, "_gate()", "soft: σ(logit)\nhard_concrete:\nlogistic noise / τ,\nstraight-through 0/1\n(eval: logit > 0)", "dec", ts=7.4, ss=5.8)
    arrow(ax, right(ch), (g[0], g[1] + 2.1))
    arrow(ax, right(sp), (g[0], g[1] + 0.7))
    out = box(ax, 12.7, 5.6, 2.1, 2.8, "Selected", "F̃_m = F_m\n⊙ c_m ⊙ s_m\n(B, 64, H, W)\n+ gates", "fuse", ts=7.4, ss=5.9)
    arrow(ax, right(g), left(out))
    loss = box(ax, 8.4, 0.5, 6.4, 3.0, "feature_selection_loss", "coverage = Σ y·(−log max_m s_m) / Σ y\nsparsity = Σ_m |s̄_m − 0.35| + |c̄_m − 0.75|\nbinarization = mean 4g(1−g)\nL_fs = cov + 0.5·sp + 0.1·bin  (× λ3 = 0.5)", "loss", ts=7.4, ss=5.9)
    arrow(ax, bottom(g), top(loss))
    text(ax, 0.2, 3.3, "Sizes for model F (C = 64, H = W = 200).\nWithout camera (E): pooled 256, spatial 20 ch.\nInit: last layers N(0, 0.01), bias 2.0\n→ all gates start at σ(2) ≈ 0.88.", 6.0,
         va="top")
    return save(fig, "feature_selection.png")


# ------------------------------------------------------------------------------ planner
def planner() -> Path:
    fig, ax = _ax(12, 9.4)
    row1 = [("Detections", "score ≥ 0.3\nboxes (K, 9) with velocity", "head"),
            ("build_scene()", "static C-space (≤ 1 m/s,\ninflated) · dynamic\noccupancy (6, 120, 140)\nclearance + LiDAR costs", "plan"),
            ("Goal", "_goal(speed, command)\nlookahead 6–45 m\n±35° for LEFT / RIGHT", "dec"),
            ("A*", "8-connected, Euclidean\nheuristic → smooth,\nresample 0.25 m", "plan")]
    row2 = [("Speed profile", "9 constant accelerations\ncollision vs dynamic\noccupancy + progress", "plan"),
            ("iLQR MPC", "BicycleILQR.solve()\n10 Hz · 30 steps · 12 it\nstate [x, y, ψ, v, δ]", "fuse"),
            ("PlanResult", "waypoints (6, 2) @ 0.5 s\nyaws · speeds\ncollision flag", "head")]
    b1 = [box(ax, 0.1 + i * 3.0, 6.5, 2.7, 2.4, t, s, k, ts=7.4, ss=5.8) for i, (t, s, k) in enumerate(row1)]
    # snake layout: row 2 runs right-to-left under A*, so no arrow crosses a box
    b2 = [box(ax, 9.1 - i * 3.0, 3.4, 2.7, 2.4, t, s, k, ts=7.4, ss=5.8) for i, (t, s, k) in enumerate(row2)]
    for a, b in zip(b1[:-1], b1[1:]):
        arrow(ax, right(a), left(b))
    arrow(ax, bottom(b1[3]), top(b2[0]))
    for a, b in zip(b2[:-1], b2[1:]):
        arrow(ax, left(a), right(b))
    text(ax, 0.1, 5.0, "BaselinePlanner.plan()", 7.0, color=INK, weight="bold")
    text(ax, 0.1, 4.55, "planning/trajectory_planner.py", 5.6)
    text(ax, 0.1, 2.75, "Learned / hybrid modes:", 6.4, color=INK)
    a1 = box(ax, 0.1, 0.4, 5.4, 1.8, "LearnedPlanner", "PlanningHead waypoints +\ncollision check (build_scene)", "head", ts=7.2, ss=5.9)
    a2 = box(ax, 6.4, 0.4, 5.5, 1.8, "HybridPlanner", "learned trajectory; if it collides\n→ BaselinePlanner fallback", "dec", ts=7.2, ss=5.9)
    arrow(ax, right(a1), left(a2))
    return save(fig, "planner.png")


def all_diagrams() -> dict[str, Path]:
    return {"architecture": architecture(), "dataflow": dataflow(), "exec_dashboard": exec_dashboard(),
            "exec_training": exec_training(), "frames": frames(), "lss": lss(), "feature_selection": feature_selection(),
            "planner": planner()}


if __name__ == "__main__":
    for k, v in all_diagrams().items():
        print(k, v)
