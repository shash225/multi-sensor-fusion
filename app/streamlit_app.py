"""Multi-Sensor Autonomous Driving System — interactive nuScenes replay & faculty demo.

Launch (either works):

    streamlit run app/streamlit_app.py
    python app/streamlit_app.py

Modes
-----
1. Dataset Viewer            - raw camera / LiDAR / radar of real nuScenes key-frames.
2. Ground Truth Simulation   - nuScenes annotations + logged ego trajectory replayed in BEV
                               (the classical planner can run on the GT objects, labelled as such).
3. AI Autonomous Driving     - the trained network: camera BEV -> LiDAR/radar -> feature selection
                               -> fusion -> detection -> tracking -> trajectory planning.

Nothing is simulated or faked: without a trained checkpoint, mode 3 falls back to
"GROUND TRUTH / DATASET VISUALIZATION" (or, if explicitly allowed, shows the untrained
network with a red UNTRAINED banner).
"""
from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _inside_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:  # pragma: no cover
        return False


if __name__ == "__main__" and not _inside_streamlit():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--theme.base", "dark", *sys.argv[1:]]
    sys.exit(stcli.main())

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import panels  # noqa: E402
from app.ui.resources import (available_versions, base_config, dashboard_methods, experiment_labels, find_checkpoint,  # noqa: E402
                              load_ablation_results, load_history)
from app.ui.styles import CSS, FLOW_STAGES, banner_html, chip, flow_html, kpis_html, panel_title, stage_key_for  # noqa: E402
from evaluation.detection_metrics import DetectionEvaluator  # noqa: E402
from evaluation.trajectory_metrics import aggregate  # noqa: E402
from inference.pipeline import FrameResult, FusionPipeline  # noqa: E402
from datasets.weather import load_weather_config, scene_weather_tags  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="Multi-Sensor Fusion · nuScenes Demo", page_icon="🚗", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0}
MODES = {"1 · Dataset Viewer": "dataset", "2 · Ground Truth Simulation": "ground_truth", "3 · AI Autonomous Driving": "model"}
PROPOSED = "F_cam_lidar_radar_fs"
FS_EXPLANATION = ("The feature-selection network learns which LiDAR and Radar features are most informative for "
                  "perception and suppresses redundant or noisy features before fusion.")
SYSTEM_STORY = ("This system takes synchronized Camera, LiDAR, and Radar data from the nuScenes autonomous-driving dataset. "
                "The camera information is transformed into a Bird's-Eye View representation. LiDAR and Radar are "
                "independently encoded into BEV features. A deep-learning feature-selection network learns which LiDAR "
                "and Radar features are important. These selected features are fused with the BEV representation, after "
                "which the system performs 3D object detection and trajectory planning. The resulting autonomous-driving "
                "scene is replayed and visualized using real nuScenes data.")


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading nuScenes data and model ...", max_entries=12)
def get_pipeline(version: str, device: str, checkpoint: str | None, experiment: str | None,
                 allow_untrained: bool) -> FusionPipeline:
    overrides = [f"dataset.version={version}", f"runtime.device={device}"]
    if experiment:
        cfg = load_config(experiment=experiment, overrides=overrides)
    else:
        cfg = base_config(version, device)
        if checkpoint is None:  # data-only pipeline: raw sensors are loaded separately, skip network camera preprocessing
            cfg.set_path("fusion.use_camera_bev", False)
    return FusionPipeline(cfg, Path(checkpoint) if checkpoint else None, split="all", allow_untrained_model=allow_untrained)


def ss_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


for k, v in {"frame_slider": 0, "playing": False, "scene": None, "results": OrderedDict(), "explain": False,
             "scene_eval": None, "scene_eval_key": None, "scene_plan": [], "seen": set(), "compare": None}.items():
    ss_default(k, v)
ss = st.session_state

# -------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🚗 Control panel")
faculty = st.sidebar.toggle("🎓 Faculty Demo", value=True, help="Simplified, presentation-friendly layout")
probe_cfg = load_config()
versions = available_versions(probe_cfg)
version_opts = [v for v, ok in versions.items() if ok] or ["v1.0-mini"]
version = st.sidebar.selectbox("nuScenes version", version_opts,
                               help="Only versions whose table folder exists under dataset.root are listed.")
missing = [v for v, ok in versions.items() if not ok]
if missing:
    st.sidebar.caption(f"Not found under `{probe_cfg.dataset.root}`: {', '.join(missing)}")
_cli_device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv[:-1] else "auto"  # streamlit run app.py -- --device cpu
device = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=["auto", "cuda", "cpu"].index(_cli_device))
mode_label = st.sidebar.radio("Demo mode", list(MODES), index=2)
requested_mode = MODES[mode_label]

methods = dashboard_methods()
labels_by_exp = experiment_labels()
method_status = {m: find_checkpoint(probe_cfg, exp) for m, exp in methods.items()}
method_label = st.sidebar.selectbox("Fusion method", list(methods), index=len(methods) - 1,
                                    format_func=lambda m: f"{'✅' if method_status[m] else '⬜'} {m}")
experiment = methods[method_label]
checkpoint = method_status[method_label]
planner_name = st.sidebar.selectbox("Trajectory planner", ["baseline", "learned", "hybrid"],
                                    help="baseline: A* + speed profile + iLQR MPC · learned: planning head · "
                                         "hybrid: learned with classical safety fallback")
allow_untrained = st.sidebar.checkbox("Allow untrained network (architecture demo only)", value=False,
                                      help="Runs randomly initialised weights; every output is labelled UNTRAINED.")
score_min = st.sidebar.slider("Detection confidence threshold", 0.05, 0.9, 0.30, 0.05)
speed_label = st.sidebar.select_slider("Playback speed", list(SPEEDS), value="1x")
with st.sidebar.expander("Display options", expanded=not faculty):
    opts = {
        "show_map": st.checkbox("Road map prior", True),
        "show_lidar": st.checkbox("LiDAR points", True),
        "show_radar": st.checkbox("Radar returns + velocity", True),
        "show_gt": st.checkbox("Ground-truth boxes (comparison)", True),
        "show_tracks": st.checkbox("Track IDs / motion trails", True),
        "show_object_futures": st.checkbox("Logged object futures", False),
        "show_gt_traj": st.checkbox("Logged ego future", True),
    }
    camera_choice = st.selectbox("Main camera", ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
                                                 "CAM_BACK_LEFT", "CAM_BACK_RIGHT"])
    plan_on_gt = st.checkbox("Run planner on ground-truth objects (mode 2)", True)

# ------------------------------------------------------------- mode handling
data_pipe = get_pipeline(version, device, None, None, False)
if requested_mode == "model" and checkpoint is not None:
    pipe = get_pipeline(version, device, str(checkpoint), None, False)
    eff_mode, banner = "model", banner_html("model", "MODEL PREDICTION")
elif requested_mode == "model" and allow_untrained:
    pipe = get_pipeline(version, device, None, experiment, True)
    eff_mode, banner = "model", banner_html("untrained", "UNTRAINED NETWORK — OUTPUTS ARE NOT MEANINGFUL")
else:
    pipe = data_pipe
    eff_mode = "ground_truth" if requested_mode in ("model", "ground_truth") else "dataset"
    banner = banner_html("gt", "GROUND TRUTH / DATASET VISUALIZATION")
fallback_note = (requested_mode == "model" and checkpoint is None and not allow_untrained)

scenes = data_pipe.scenes()
scene_names = [s["name"] for s in scenes]
_real_rules = load_weather_config()["real_weather"]
REAL_ICON = {"rain": "🌧 rain", "fog": "🌫 fog", "night": "🌙 night", "wet_road": "💧 wet road (after rain)"}


def _scene_label(i: int) -> str:
    sc = scenes[i]
    tag = scene_weather_tags(sc["description"], _real_rules)["condition"]
    real = f" · {REAL_ICON[tag]}" if tag in REAL_ICON else ""
    return f"{sc['name']} · {sc['split']}{real} · {sc['description'][:38]}"


scene_idx = st.sidebar.selectbox("Scene", range(len(scenes)), format_func=_scene_label,
                                 help="Real nuScenes scenes. Tags (night / wet road / rain) come from the original "
                                      "scene description; nuScenes-mini contains no real fog or rain recording.")
scene = scenes[scene_idx]
WEATHER_ICON = {"clear": "☀️", "light_fog": "🌫", "dense_fog": "🌫🌫", "light_rain": "🌦", "moderate_rain": "🌧",
                "heavy_rain": "⛈"}
weather_conditions = data_pipe.weather_conditions()
weather = st.sidebar.selectbox(
    "Weather", list(weather_conditions),
    format_func=lambda w: f"{WEATHER_ICON.get(w, '')} {weather_conditions[w]['label']}"
                          + ("  · real data" if w == "clear" else "  · SYNTHETIC"),
    help="Fog / rain are SYNTHETIC: physically-motivated corruptions applied to the camera, LiDAR and radar of the "
         "real clear-weather frame (configs/weather.yaml, docs/ADVERSE_WEATHER.md). The trained models are unchanged.")
if ss.scene != scene["name"]:
    ss.scene, ss.frame_slider, ss.playing = scene["name"], 0, False


# ------------------------------------------------------------------- callbacks
def cb_prev() -> None:
    ss.playing = False
    ss.frame_slider = max(0, ss.frame_slider - 1)


def cb_next() -> None:
    ss.playing = False
    ss.frame_slider = min(scene["count"] - 1, ss.frame_slider + 1)


def cb_play() -> None:
    if ss.frame_slider >= scene["count"] - 1:
        ss.frame_slider = 0
    ss.playing = True


def cb_pause() -> None:
    ss.playing = False


def cb_reset() -> None:
    ss.playing = False
    ss.frame_slider = 0
    ss.scene_eval, ss.scene_eval_key, ss.scene_plan, ss.seen = None, None, [], set()
    pipe.tracker.reset()


def cb_load() -> None:
    cb_reset()


if ss.pop("_advance", False):
    if ss.frame_slider < scene["count"] - 1:
        ss.frame_slider += 1
    else:
        ss.playing = False

# ---------------------------------------------------------------------- header
exp_label = labels_by_exp.get(experiment, experiment)
sub = f"nuScenes {version} · {scene['name']} ({scene['split']} split, {scene['location']}) · {scene['description']}"
st.markdown(f'<div class="msf-title"><div><h1>MULTI-SENSOR AUTONOMOUS DRIVING SYSTEM</h1>'
            f'<div class="sub">{sub}</div></div><div>{banner}</div></div>', unsafe_allow_html=True)
if fallback_note:
    st.warning(f"No trained checkpoint found for **{method_label}** (experiment `{experiment}`). "
               "Showing ground-truth / dataset visualization instead of pretending to predict. "
               f"Train it with: `python main.py train --stage joint --experiment {experiment}`")
if weather != "clear":
    st.error(f"**SYNTHETIC WEATHER — {weather_conditions[weather]['label']}.** Simulated fog/rain applied to the camera, "
             "LiDAR and radar of this real clear-weather nuScenes frame (not a real adverse-weather recording). "
             "The trained network is unchanged; see docs/ADVERSE_WEATHER.md for the physics and the full test results.")
if eff_mode == "model" and scene["split"] == "train" and pipe.model_trained:
    st.info("This scene was part of the **training** split. Use a **val** scene (scene-0103, scene-0916 in nuScenes-mini) "
            "to show performance on data the network has never seen.")

c = st.columns([1.1, 1, 1, 1, 1, 1, 4.4])
c[0].button("📂 Load Scene", on_click=cb_load, width="stretch")
c[1].button("⏮ Previous", on_click=cb_prev, width="stretch")
c[2].button("▶ Play", on_click=cb_play, width="stretch", type="primary" if not ss.playing else "secondary")
c[3].button("⏸ Pause", on_click=cb_pause, width="stretch")
c[4].button("⏭ Next", on_click=cb_next, width="stretch")
c[5].button("🔄 Reset", on_click=cb_reset, width="stretch")
c[6].slider("Keyframe", 0, scene["count"] - 1, key="frame_slider", label_visibility="collapsed",
            help="nuScenes key-frames are 2 Hz (0.5 s apart)")
index = scene["start"] + int(ss.frame_slider)

# ---------------------------------------------------------------- processing
flow_slot = st.empty() if faculty else None
done: dict[str, float] = {}
skipped = []
if eff_mode != "model":
    skipped = ["camera", "sensors", "selection", "fusion"]
else:
    model = pipe.model
    if not model.use_camera:
        skipped.append("camera")
    if not model.use_feature_selection:
        skipped.append("selection")


def on_stage(name: str, ms: float) -> None:
    key = stage_key_for(name)
    if key is None:
        return
    done[key] = done.get(key, 0.0) + ms
    if flow_slot is not None:
        nxt = next((k for k, _, _ in FLOW_STAGES if k not in done and k not in skipped), None)
        flow_slot.markdown(flow_html(done, nxt, skipped), unsafe_allow_html=True)


cache_key = (str(checkpoint) if eff_mode == "model" and checkpoint else f"untrained:{experiment}" if eff_mode == "model" else "data",
             index, eff_mode, planner_name, plan_on_gt, weather)
res: FrameResult
if cache_key in ss.results:
    res = ss.results[cache_key]
    for k, v in res.timings_ms.items():
        key = stage_key_for(k)
        if key:
            done[key] = done.get(key, 0.0) + v
else:
    if flow_slot is not None:
        flow_slot.markdown(flow_html({}, next(k for k, _, _ in FLOW_STAGES if k not in skipped), skipped), unsafe_allow_html=True)
    res = pipe.run(index, eff_mode, planner_name, plan_on_gt, stage_callback=on_stage, weather=weather)
    ss.results[cache_key] = res
    while len(ss.results) > 24:
        ss.results.popitem(last=False)

t_render = time.perf_counter()
images = pipe.load_display_images(res)

# Scene-level running metrics (frames visited so far with the current method).
eval_key = (cache_key[0], scene["name"], eff_mode, planner_name, weather)
if ss.scene_eval_key != eval_key:
    ss.scene_eval_key, ss.scene_eval, ss.scene_plan, ss.seen = eval_key, DetectionEvaluator(pipe.classes), [], set()
if index not in ss.seen:
    ss.seen.add(index)
    pipe.add_to_evaluator(ss.scene_eval, res)
    if "planning" in res.frame_metrics:
        ss.scene_plan.append(res.frame_metrics["planning"])
scene_det = ss.scene_eval.evaluate() if eff_mode == "model" and ss.scene_eval.samples else None
scene_plan = aggregate(ss.scene_plan) if ss.scene_plan else None

# ------------------------------------------------------------------ KPI strip
dets = res.detections
n_obj = int((dets["scores"] >= score_min).sum()) if dets is not None and eff_mode == "model" else (
    len(dets["boxes"]) if dets is not None else 0)
net_ms = res.timings_ms.get("network")
fps = 1000.0 / max(res.timings_ms.get("total", 1.0) - res.timings_ms.get("data_loading", 0.0), 1e-3)
conf = float(np.mean(dets["scores"][dets["scores"] >= score_min])) if dets is not None and eff_mode == "model" and n_obj else None
fm = res.frame_metrics
plan_m = fm.get("planning", {})
kpis = [("Detection", f"{n_obj} objects", "model output" if eff_mode == "model" else "ground truth" if dets is not None else "—"),
        ("FPS", f"{fps:.1f}", f"network {net_ms:.0f} ms" if net_ms else f"pipeline {res.timings_ms.get('total', 0):.0f} ms")]
if eff_mode == "model":
    kpis.append(("NDS", f"{fm['detection']['NDS']:.3f}" if 'detection' in fm else "—",
                 f"scene so far {scene_det['NDS']:.3f} ({scene_det['num_samples']} frames)" if scene_det else "frame"))
    kpis.append(("mAP", f"{fm['detection']['mAP']:.3f}" if 'detection' in fm else "—",
                 f"present classes {fm['detection']['mAP_present_classes']:.3f}" if 'detection' in fm else ""))
kpis.append(("ADE", f"{plan_m['ade']:.2f} m" if "ade" in plan_m else "—",
             f"scene {scene_plan['ade']:.2f} m" if scene_plan and "ade" in scene_plan else "vs logged trajectory"))
kpis.append(("Collision", ("YES" if plan_m.get("collision") else "no") if plan_m else "—",
             f"scene rate {scene_plan.get('collision_rate', 0):.0%}" if scene_plan else "vs GT agents"))
kpis.append(("Fusion", experiment_labels(short=True).get(experiment, experiment) if eff_mode == "model" else "—",
             f"mean confidence {conf:.2f}" if conf else ""))
st.markdown(kpis_html(kpis), unsafe_allow_html=True)
if flow_slot is not None:
    done["output"] = 0.0
    flow_slot.markdown(flow_html(done, None, skipped), unsafe_allow_html=True)

# ---------------------------------------------------------- row A: cam + fusion
colA, colB = st.columns([1.05, 1])
with colA:
    st.markdown(panel_title(f"Panel 1 · Camera view ({camera_choice})"), unsafe_allow_html=True)
    cam_img, mosaic = panels.camera_panel(res, images, camera_choice, show_pred=True, show_gt=opts["show_gt"], score_min=score_min)
    if cam_img is not None:
        st.image(cam_img, width="stretch",
                 caption=("Predicted 3D boxes (class colour, confidence)" + (" + ground truth (green)" if opts["show_gt"] else ""))
                 if eff_mode == "model" else "Ground-truth 3D boxes projected with the camera calibration")
    with st.expander("All six cameras", expanded=False):
        st.image(mosaic, width="stretch")
with colB:
    st.markdown(panel_title("Panel 3 · Sensor fusion"), unsafe_allow_html=True)
    fs_on = eff_mode == "model" and pipe.model.use_feature_selection
    tiles = panels.fusion_tiles(res, images, fs_on)
    for row in (tiles[:3], tiles[3:]):
        cols = st.columns(3)
        for col, tile in zip(cols, row):
            col.image(tile, width="stretch")
    if faculty:
        st.caption("Top: camera → BEV, LiDAR BEV, radar BEV. Bottom: learned importance of LiDAR and radar features, "
                   "and the fused BEV used for detection and planning.")

# --------------------------------------------------------- row B: BEV + plan
colC, colD = st.columns([1.35, 1])
with colC:
    st.markdown(panel_title("Panel 2 · Autonomous driving BEV"), unsafe_allow_html=True)
    st.image(panels.driving_bev(res, pipe, {**opts, "score_min": score_min}), width="stretch")
with colD:
    st.markdown(panel_title("Panel 5 · Trajectory planning"), unsafe_allow_html=True)
    ego = panels.ego_status(res)
    st.markdown(kpis_html([("Ego X / Y", f"{ego['x']:.1f} / {ego['y']:.1f}", "global frame (m)"),
                           ("Yaw", f"{ego['yaw_deg']:.1f}°", f"yaw rate {ego['yaw_rate']:.2f} rad/s"),
                           ("Velocity", f"{ego['speed']:.2f} m/s", f"accel {ego['accel']:.2f} m/s²"),
                           ("Command", res.command_name, "navigation input")]), unsafe_allow_html=True)
    if res.plan is not None or res.info["ego_future"]["mask"].any():
        st.image(panels.trajectory_closeup(res, pipe), width="stretch")
    if res.plan is not None:
        status = chip("COLLISION PREDICTED", "bad") if res.plan.collision else chip("collision-free (predicted occupancy)", "ok")
        st.markdown(f"{chip(res.plan_source)} {status}", unsafe_allow_html=True)
        if plan_m:
            st.markdown(kpis_html([("ADE", f"{plan_m['ade']:.2f} m", "mean displacement"),
                                   ("FDE", f"{plan_m['fde']:.2f} m", "final displacement"),
                                   ("Lateral / long.", f"{plan_m['lateral_error']:.2f} / {plan_m['longitudinal_error']:.2f} m", "error"),
                                   ("GT collision", "YES" if plan_m.get("collision") else "no", "vs real agent futures")]),
                        unsafe_allow_html=True)
    elif eff_mode == "dataset":
        st.caption("Dataset viewer: the white dashed line is the logged human trajectory. Switch to mode 2 or 3 to plan.")

# ------------------------------------------------------------ row C: detection
st.markdown(panel_title("Panel 4 · Object detection"), unsafe_allow_html=True)
colE, colF = st.columns([1.6, 1])
with colE:
    table = panels.detection_table(res, score_min if eff_mode == "model" else 0.0)
    if len(table):
        st.dataframe(table, width="stretch", height=260, hide_index=True,
                     column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                                    "X (m)": st.column_config.NumberColumn(format="%.1f"), "Y (m)": st.column_config.NumberColumn(format="%.1f"),
                                    "Range (m)": st.column_config.NumberColumn(format="%.1f"),
                                    "Velocity (m/s)": st.column_config.NumberColumn(format="%.1f")})
    else:
        st.caption("No objects for this frame / mode.")
with colF:
    cmp = panels.gt_comparison(res, score_min)
    if cmp:
        st.markdown("**Ground truth vs model prediction** (2 m centre distance, same class)")
        st.markdown(kpis_html([("GT objects", str(cmp["ground_truth"]), ""), ("Predicted", str(cmp["predicted"]), ""),
                               ("True positives", str(cmp["true_positive"]), ""), ("False positives", str(cmp["false_positive"]), ""),
                               ("Missed", str(cmp["missed"]), "")]), unsafe_allow_html=True)
        if scene_det:
            st.caption(f"Scene so far ({scene_det['num_samples']} frames): NDS {scene_det['NDS']:.3f} · mAP {scene_det['mAP']:.3f} · "
                       f"mAP(present classes) {scene_det['mAP_present_classes']:.3f} · mAVE {scene_det['mAVE']:.2f} m/s")
    else:
        st.caption("Ground-truth comparison is available in mode 3 with a trained model.")

# ------------------------------------------------------ explain feature selection
st.divider()
if st.button("🔍 Explain Feature Selection", type="primary" if not ss.explain else "secondary"):
    ss.explain = not ss.explain
if ss.explain:
    st.markdown(f'<div class="msf-explain">{FS_EXPLANATION}</div>', unsafe_allow_html=True)
    if eff_mode != "model" or not pipe.model.use_feature_selection:
        st.info("Feature-selection maps come from the network's actual gates: choose a trained method with feature "
                "selection (methods 5 or 6) in mode 3.")
    else:
        ex = panels.explain_selection(res, pipe, score_min)
        if not res.model_trained:
            st.error("UNTRAINED weights: these gates are random and carry no meaning.")
        for mod, row in ex["rows"].items():
            st.markdown(f"**{'LiDAR' if mod == 'lidar' else 'Radar'}:** raw features → feature importance → selected features")
            cols = st.columns([1, 0.12, 1, 0.12, 1])
            for i, img in enumerate(row):
                cols[i * 2].image(img, width="stretch")
            cols[1].markdown("<div style='font-size:2rem;text-align:center;margin-top:120px'>→</div>", unsafe_allow_html=True)
            cols[3].markdown("<div style='font-size:2rem;text-align:center;margin-top:120px'>→</div>", unsafe_allow_html=True)
        st.caption("Importance = the network's spatial gate (0 suppressed … 1 kept). A gate only matters where the sensor "
                   "has features, so the 'selected' tile (raw × channel gate × spatial gate) shows what actually reaches fusion.")
        cols = st.columns(3)
        if ex["dominance"] is not None:
            cols[0].image(ex["dominance"], width="stretch")
        if ex["attention"] is not None:
            cols[1].image(ex["attention"], width="stretch")
        if "heatmap" in ex:
            cols[2].image(ex["heatmap"], width="stretch")
        st.markdown("**Which LiDAR and Radar features did the network consider important?**")
        gate_rows = []
        for mod, info in ex["channels"].items():
            for name, g in info["named"]:
                gate_rows.append({"sensor": "LiDAR" if mod == "lidar" else "Radar",
                                  "feature": name.replace("lidar_", "").replace("radar_", "").replace("_", " "), "gate": g})
        if gate_rows:
            df = pd.DataFrame(gate_rows)
            chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=14).encode(
                x=alt.X("gate:Q", scale=alt.Scale(domain=[0, 1]), title="channel gate (0 = suppressed, 1 = kept)"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("sensor:N", scale=alt.Scale(domain=["LiDAR", "Radar"], range=["#3987e5", "#d95926"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["sensor", "feature", alt.Tooltip("gate:Q", format=".3f")],
            ).properties(height=34 * len(df) // 2 + 40)
            gcol1, gcol2 = st.columns([1.2, 1])
            gcol1.altair_chart(chart, width="stretch")
            with gcol2:
                for mod, info in ex["channels"].items():
                    st.markdown(f"{chip('LiDAR' if mod == 'lidar' else 'Radar')} {info['selected_ratio']:.0%} of channels "
                                f"selected · learned channels mean gate {info['learned_mean']:.2f}", unsafe_allow_html=True)
                stats = res.selection_stats
                for mod in ("lidar", "radar"):
                    if f"{mod}_spatial_selected_ratio" in stats:
                        st.markdown(f"{chip('LiDAR' if mod == 'lidar' else 'Radar')} spatial cells selected "
                                    f"{stats[f'{mod}_spatial_selected_ratio']:.0%} · effective selected features "
                                    f"{stats[f'{mod}_effective_selected_ratio']:.0%}", unsafe_allow_html=True)
        if ex["class_trust"]:
            st.markdown("**Sensor trust per detected object class** (mean importance inside the detected boxes)")
            st.dataframe(pd.DataFrame(ex["class_trust"]), width="stretch", hide_index=True)

# ---------------------------------------------------------- compare methods
with st.expander("⚖️ Compare fusion methods on this scene", expanded=False):
    trained = {m: p for m, p in method_status.items() if p is not None}
    st.markdown(" ".join(chip(f"{m.split('. ', 1)[-1]}", "ok" if p else "") for m, p in method_status.items()),
                unsafe_allow_html=True)
    if not trained:
        st.info("No trained checkpoints yet. Run `python main.py ablation --train` to train all methods.")
    else:
        cc = st.columns(2)
        run_frame = cc[0].button("Run all trained methods on this frame")
        run_scene = cc[1].button("Evaluate all trained methods on the whole scene (slower)")
        if run_frame or run_scene:
            rows = []
            prog = st.progress(0.0)
            frames = range(scene["start"], scene["start"] + scene["count"]) if run_scene else [index]
            for mi, (m, path) in enumerate(trained.items()):
                p = get_pipeline(version, device, str(path), None, False)
                ev, plans, lat = DetectionEvaluator(p.classes), [], []
                for fi in frames:
                    r = p.run(fi, "model", planner_name, weather=weather)
                    p.add_to_evaluator(ev, r)
                    if "planning" in r.frame_metrics:
                        plans.append(r.frame_metrics["planning"])
                    lat.append(r.timings_ms.get("network", 0.0))
                det = ev.evaluate()
                pl = aggregate(plans) if plans else {}
                rows.append({"method": m, "experiment": methods[m], "weather": weather + ("" if weather == "clear" else " (synthetic)"),
                             "NDS": det["NDS"], "mAP": det["mAP"],
                             "mAP (present classes)": det["mAP_present_classes"], "mAVE": det["mAVE"],
                             "ADE (m)": pl.get("ade"), "FDE (m)": pl.get("fde"), "collision rate": pl.get("collision_rate"),
                             "network ms": float(np.median(lat)), "frames": len(frames)})
                prog.progress((mi + 1) / len(trained))
            ss.compare = pd.DataFrame(rows)
        if ss.compare is not None:
            df = ss.compare
            st.dataframe(df, width="stretch", hide_index=True)
            metric = st.selectbox("Chart metric", ["NDS", "mAP", "mAP (present classes)", "mAVE", "ADE (m)", "FDE (m)",
                                                   "collision rate", "network ms"])
            dfc = df.assign(proposed=df["experiment"].eq(PROPOSED).map({True: "proposed", False: "baseline"}))
            st.altair_chart(alt.Chart(dfc).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X(f"{metric}:Q"), y=alt.Y("method:N", sort=None, title=None),
                color=alt.Color("proposed:N", scale=alt.Scale(domain=["proposed", "baseline"], range=["#3987e5", "#6b6a64"]),
                                legend=None),
                tooltip=["method", alt.Tooltip(f"{metric}:Q", format=".3f"), "frames"]), width="stretch")

# ------------------------------------------------------- adverse-weather study
with st.expander("🌧 Adverse-weather study (fog / rain, SYNTHETIC, offline)", expanded=False):
    wres = resolve_path(probe_cfg.runtime.output_dir) / "adverse_weather" / "synthetic" / "results.csv"
    if not wres.exists():
        st.info("No weather study yet. Run `python main.py weather --weather all` (about 15 min on a GTX 1650).")
    else:
        wdf = pd.read_csv(wres)
        st.caption("All six trained models evaluated unchanged on the 81 validation frames with simulated fog and rain on "
                   "camera, LiDAR and radar. Precision / recall: score ≥ 0.3, same class within 2 m.")
        pivot = wdf.pivot_table(index="experiment_label", columns="condition", values="NDS", sort=False)
        st.markdown("**NDS by weather**")
        st.dataframe(pivot.style.format("{:.4f}"), width="stretch")
        rel = wdf[wdf.condition != "clear"].pivot_table(index="experiment_label", columns="condition", values="rel_NDS",
                                                        sort=False) * 100
        st.markdown("**NDS change vs clear (%)**")
        st.dataframe(rel.style.format("{:+.1f}"), width="stretch")
        f = wdf[wdf.experiment == PROPOSED][["condition_label", "precision", "recall", "NDS", "mAP", "FN", "FP", "FPS"]]
        if len(f):
            st.markdown("**Proposed model under each condition** (FN = missed objects, FP = false detections)")
            st.dataframe(f, width="stretch", hide_index=True)
        for img in ("relative_nds_heatmap.png", "precision_recall_by_weather.png", "fusion_trust_by_weather.png"):
            pth = wres.parent / "plots" / img
            if pth.exists():
                st.image(str(pth), width="stretch")

# ------------------------------------------------------------ ablation / curves
with st.expander("📊 Ablation study (offline evaluation on the validation split)", expanded=False):
    results = load_ablation_results(probe_cfg)
    if not results:
        st.info("Run `python main.py ablation --train` to produce outputs/ablation/results.json.")
    else:
        df = pd.DataFrame(results)
        show = ["label", "NDS", "mAP", "mAP_present", "mAVE", "ADE", "FDE", "collision_rate", "latency_ms",
                "lidar_selected_ratio", "radar_selected_ratio"]
        st.dataframe(df[[c for c in show if c in df]], width="stretch", hide_index=True)
        metric = st.selectbox("Metric", [c for c in show[1:] if c in df], key="abl_metric")
        dfa = df.assign(proposed=df["experiment"].eq(PROPOSED).map({True: "proposed", False: "baseline"}))
        st.altair_chart(alt.Chart(dfa).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X(f"{metric}:Q"), y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("proposed:N", scale=alt.Scale(domain=["proposed", "baseline"], range=["#3987e5", "#6b6a64"]), legend=None),
            tooltip=["label", alt.Tooltip(f"{metric}:Q", format=".4f")]), width="stretch")
        png = Path(probe_cfg.runtime.output_dir)
        png = (ROOT / png if not png.is_absolute() else png) / "ablation" / "ablation_metrics.png"
        if png.exists():
            st.image(str(png), width="stretch")

with st.expander("📈 Training curves", expanded=False):
    hist = load_history(probe_cfg, experiment)
    if not hist:
        st.info(f"No training history for `{experiment}` yet.")
    else:
        rows = [{"epoch": h["epoch"], "train loss": h["train"].get("total"),
                 "val NDS": h.get("val", {}).get("detection", {}).get("NDS"),
                 "val ADE": h.get("val", {}).get("planning_learned", {}).get("ade")} for h in hist]
        dfh = pd.DataFrame(rows)
        cols = st.columns(3)
        for col, m in zip(cols, ["train loss", "val NDS", "val ADE"]):
            col.altair_chart(alt.Chart(dfh.dropna(subset=[m])).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2,
                                                                         color="#3987e5").encode(
                x=alt.X("epoch:Q"), y=alt.Y(f"{m}:Q"), tooltip=["epoch", alt.Tooltip(f"{m}:Q", format=".4f")]
            ).properties(title=m, height=220), width="stretch")

with st.expander("ℹ️ About this system", expanded=faculty and not ss.playing):
    st.markdown(f'<div class="msf-explain">{SYSTEM_STORY}</div>', unsafe_allow_html=True)
    if pipe.checkpoint_meta:
        meta = pipe.checkpoint_meta
        st.caption(f"Checkpoint: `{pipe.checkpoint}` · experiment {meta.get('experiment')} · stage {meta.get('stage')} · "
                   f"epoch {meta.get('epoch')}")

# ------------------------------------------------------------------ playback
if ss.playing:
    interval = 0.5 / SPEEDS[speed_label]
    elapsed = time.perf_counter() - t_render
    time.sleep(max(0.0, interval - elapsed))
    ss["_advance"] = True
    st.rerun()
