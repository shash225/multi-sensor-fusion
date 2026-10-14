"""Block-by-block walkthroughs: planning, training, evaluation, inference, visualisation, application, CLI."""

WALKTHROUGHS = [
    {
        "file": "planning/occupancy_grid.py",
        "intro": "Occupancy and cost layers used by the classical planner (ego frame, layout `[iy, ix]`).",
        "blocks": [
            ('"""Occupancy / cost grids', "Docstring: dynamic occupancy (constant-velocity forecast), static "
             "configuration space, soft costs."),
            ("def planning_grid", "`planning_grid(cfg.planning)` → the planner's own `BEVGrid` (x ∈ [−10, 60), "
             "y ∈ [−30, 30), 0.5 m → 140 × 120 cells): forward-biased because the plan only covers 3 s ahead. "
             "`propagate_boxes(boxes, t)` shifts box centres by `velocity · t` (constant-velocity motion forecast)."),
            ("class OccupancyBuilder", "Constructor stores the grid, horizon (6), dt (0.5 s) and the ego box "
             "`[rear_axle_to_center, 0, 0, length, width, …]` in the ego frame (the ego frame origin is the rear axle, "
             "so the body centre is 1.4 m ahead)."),
            ("    def dynamic_occupancy", "`dynamic_occupancy(boxes, inflate, times)` → `(T, ny, nx)` uint8: for every "
             "future time `t = dt, 2dt, …, T·dt` the boxes are propagated and rasterised (with optional inflation). "
             "`static_occupancy(boxes, inflate)` → `(ny, nx)` at t = 0."),
            ("    def lidar_obstacles", "`lidar_obstacles(points, z_band)`: cells containing non-ground LiDAR returns "
             "(0.4 m < z < 2.5 m), excluding points on the ego vehicle — catches obstacles the detector missed, "
             "curbs, poles and vegetation."),
            ("    def clearance_cost", "`clearance_cost(occ)`: `exp(−d / 1.5 m)` where `d` is the OpenCV distance "
             "transform (distance to the nearest occupied cell) — a soft cost that keeps paths away from obstacles."),
            ("    def footprint_points", "`footprint_points(xy (N, 2), yaw (N,))` → `(N, 15, 2)`: a 5 × 3 grid of points "
             "covering the ego body (optionally enlarged by a margin) for each rear-axle pose. "
             "`occupied_at(occ, pts)` looks the points up in an occupancy grid (outside the grid = free)."),
        ],
    },
    {
        "file": "planning/astar.py",
        "intro": "Grid A* used by the baseline planner to find a geometric path on the static cost map.",
        "blocks": [
            ('"""Grid A*', "Imports and `_MOVES`: the 8-neighbourhood with step lengths 1 and √2."),
            ("def astar", "`astar(cost, start, goal, max_expansions, heuristic_weight)`:\n\n"
             "- `cost` contains per-cell traversal costs; `inf` marks blocked cells. Moving into a cell costs "
             "`step · (1 + cost)` — with zero cost this is the Euclidean path length.\n"
             "- Classic A* with a binary heap (`heapq`) of `(f = g + h, g, node)`, `g_score` and `parent` arrays and a "
             "`closed` set; the heuristic is the Euclidean distance (admissible).\n"
             "- The best node seen so far (closest to the goal) is tracked; if the goal is unreachable or the "
             "expansion budget runs out, the path to that node is returned with `reached = False` (best effort "
             "instead of failure).\n"
             "- The path is reconstructed backwards through `parent` and reversed."),
            ("def nearest_free", "`nearest_free(cost, cell, max_radius)`: searches square rings of increasing radius "
             "for the closest finite-cost cell — used when the goal falls inside an obstacle."),
        ],
    },
    {
        "file": "planning/mpc.py",
        "intro": "Model-predictive control by iterative LQR on a kinematic bicycle model. Turns the geometric path "
                 "and speed profile into a dynamically feasible, smooth trajectory.",
        "blocks": [
            ('"""Model Predictive Control', "Docstring: state `[x, y, yaw, v, δ]`, control `[a, ω]` (acceleration, "
             "steering rate), the discrete dynamics and the stage cost."),
            ("@dataclass", "`MPCResult`: optimised states `(N+1, 5)`, controls `(N, 2)`, final cost and iterations used."),
            ("class BicycleILQR", "Constructor: `dt` (0.1 s), iterations (12), wheelbase L (2.588 m), steering / "
             "acceleration / speed limits from `planning.ego`, and the diagonal cost matrices `Q` "
             "(position 4, yaw 1, speed 0.5, steering 2) and `R` (acceleration 0.1, steering rate 4)."),
            ("    def step", "`step(s, u)`: one Euler step of the bicycle model — position moves along the heading, "
             "`yaw += v/L · tan(δ) · dt`, `v += a·dt`, `δ += ω·dt`; speed is clipped to [0, v_max] and steering to "
             "±δ_max (no reversing)."),
            ("    def jacobians", "`jacobians(s)`: analytic linearisation `A = ∂f/∂s` (5 × 5) and `B = ∂f/∂u` (5 × 2) "
             "around the current state, needed by the Riccati recursion."),
            ("    def rollout", "`rollout`, `_clip` (control limits), `_err` (state error with the yaw difference wrapped "
             "to [−π, π)) and `cost` (Σ eᵀQe + uᵀRu over the horizon)."),
            ("    def solve", "`solve(s0, ref, u_init)` — iLQR:\n\n"
             "1. Initial rollout with the given controls (the planner passes the chosen constant acceleration).\n"
             "2. **Backward pass** (Gauss–Newton): from the last step to the first, the quadratic expansion of the "
             "cost-to-go gives feedforward `k_t = −Quu⁻¹ Qu` and feedback `K_t = −Quu⁻¹ Qux`; the value function "
             "derivatives `Vx, Vxx` are propagated (symmetrised for stability; `1e-6·I` regularises `Quu`).\n"
             "3. **Forward pass** with a backtracking line search over α ∈ {1, 0.5, 0.25, 0.1}: "
             "`u_t = u_t + α·k_t + K_t·(x_t − x̄_t)`, clipped, re-simulated; accepted if the cost decreases.\n"
             "4. Stops when no step improves or the relative improvement is < 1e-4."),
        ],
    },
    {
        "file": "planning/tracking.py",
        "intro": "Online multi-object tracking used by the replay (track IDs, smoothed velocities, motion trails).",
        "blocks": [
            ('"""Online multi-object', "Docstring and `MATCH_DISTANCE`: CenterPoint's class-specific maximum association "
             "distances (e.g. car 4 m, pedestrian 1 m, motorcycle 13 m)."),
            ("@dataclass", "`Track`: id, label, global position and velocity, score, age, consecutive misses and the "
             "position history."),
            ("class MultiObjectTracker", "Constructor and `reset()` (clears tracks, restarts ids)."),
            ("    def update", "`update(boxes_ref, labels, scores, ref_ego2global, timestamp_us)` → track ids:\n\n"
             "1. Time step `dt` from the timestamps; a jump backwards or > 2 s resets the tracker (scene change / seek).\n"
             "2. Detections are moved to the **global** frame (tracks must survive ego motion).\n"
             "3. Existing tracks are predicted with `position + velocity · dt`.\n"
             "4. Greedy association in descending score order: nearest unused track of the same class within the "
             "class distance.\n"
             "5. Matched: velocity = ½ (detected velocity) + ½ (finite-difference velocity); unmatched detection → "
             "new track; unmatched tracks accumulate misses and are removed after `max_misses` (2).\n"
             "6. Histories are truncated to the last 8 positions."),
            ("    def histories_ref", "`histories_ref(ids, ref_ego2global)`: track histories transformed back into the "
             "current reference frame for drawing motion trails."),
        ],
    },
    {
        "file": "planning/trajectory_planner.py",
        "intro": "The planners. `BaselinePlanner` needs no training; `LearnedPlanner` and `HybridPlanner` use the "
                 "network's planning head. All share the `plan(...) -> PlanResult` interface.",
        "blocks": [
            ('"""Trajectory planners', "Docstring with the path/velocity-decomposition pipeline."),
            ("from __future__", "Imports and `COMMAND_NAMES`."),
            ("@dataclass", "`PlanResult`: waypoints `(T, 2)` at t = 0.5 … 3.0 s (rear-axle positions in the reference "
             "frame), yaws, speeds, the collision flag against *predicted* occupancy, planner name, command, the A* "
             "path, the dense 10 Hz MPC states, goal, chosen acceleration, run time and a debug dict "
             "(occupancy layers)."),
            ("def _smooth_path", "Helpers: `_smooth_path` (moving average, first point fixed), `_resample` (equal arc-length "
             "steps of 0.25 m; returns points and cumulative arc length), `_pose_at(path, arclen, s)` (position and "
             "heading at arc lengths `s`, extrapolated straight beyond the end of the path)."),
            ("class BaselinePlanner", "Constructor: occupancy builder, planning grid, iLQR solver, horizon, dt, classes."),
            ("    def _goal", "`_goal(speed, command)`: lookahead distance `max(speed, 2) · 3 s · 1.2`, clipped to [6, 45] m; "
             "direction 0° for STRAIGHT, ±35° for LEFT/RIGHT. Coordinate helpers `_to_cell` / `_to_metric`."),
            ("    def collision_check", "`collision_check(waypoints, yaws, dyn_occ)` → per-step booleans: ego footprint "
             "points at step t tested against the dynamic occupancy at step t."),
            ("    def build_scene", "`build_scene(detections, lidar_points, map_drivable)`:\n\n"
             "- keeps detections with score ≥ `occupancy.min_score` (0.3);\n"
             "- *static* obstacles = objects slower than 1 m/s, rasterised inflated by half the ego width + margin "
             "(configuration space for a point robot);\n"
             "- *dynamic* occupancy over the horizon for all objects (constant-velocity forecast);\n"
             "- cost = ½·clearance cost + LiDAR obstacle cost (dilated) + optional map cost; static cells → `inf`.\n\n"
             "Moving objects are deliberately *not* blocked geometrically (the ego should follow a moving car, not "
             "plan around it); they are handled by the speed profile."),
            ("    def plan(self", "`plan()`, part 1 — geometry: clears a small region around the ego start (it may start "
             "inside an inflated obstacle), computes the goal, moves it to the nearest free cell, runs A*, converts "
             "cells to metres, smooths and resamples the path (a straight 1 m path is the fallback)."),
            ("        # Speed profile", "Part 2 — speed: reference distance `s_ref` from the current speed and (clipped) "
             "acceleration; for each candidate constant acceleration the speed profile (clipped to [0, v_max]), the "
             "travelled distance, the poses on the path and the collisions against the dynamic occupancy are "
             "computed; cost = progress error + comfort (a²) + 1000 per colliding step; the cheapest candidate wins."),
            ("        # MPC tracking at 10 Hz", "Part 3 — MPC: the chosen profile is interpolated to 10 Hz, the "
             "reference states `[x, y, yaw, v, 0]` are built, the initial steering is estimated from the yaw rate, "
             "and iLQR tracks the reference. States at t = 0.5 … 3.0 s become the waypoints; the final collision "
             "flag is recomputed on them."),
            ("class LearnedPlanner", "`LearnedPlanner.plan()` takes the planning head's waypoints, derives headings and "
             "speeds from consecutive differences and reports collisions against the same predicted occupancy "
             "(raises `ValueError` if no learned waypoints were given)."),
            ("class HybridPlanner", "`HybridPlanner`: uses the learned trajectory unless it collides with the predicted "
             "occupancy, in which case the baseline planner's result is returned (planner label "
             "`hybrid(baseline fallback)`)."),
            ("def build_planner", "`build_planner(cfg, name)`: factory by name (default `planning.planner`)."),
        ],
    },
    {
        "file": "training/trainer.py",
        "intro": "One `Trainer` instance trains one stage of one experiment.",
        "blocks": [
            ('"""Stage-aware trainer', "Docstring: what a stage is and the memory/speed features."),
            ("from __future__", "Imports."),
            ("def stage_checkpoint_dir", "`stage_checkpoint_dir(cfg, experiment, stage)`: shared single-sensor stages "
             "(`shared: true`) are stored under `checkpoints/_shared/<stage>/` so all experiments can reuse them; "
             "other stages under `checkpoints/<experiment>/<stage>/`."),
            ("def build_stage_config", "`build_stage_config(experiment, stage, overrides)`: loads the experiment "
             "config, deep-merges the stage's `overrides` (e.g. camera-only for stage 1) and re-applies the CLI "
             "overrides so they still win; records `runtime.stage`."),
            ("class Trainer", "Constructor, part 1: configuration, checkpoint and log directories, logging to "
             "`outputs/logs/<experiment>/<stage>/run.log`, seeding, device and AMP decision (fp16 self-test), "
             "epoch count (CLI > stage > `train.epochs`), active losses."),
            ("        self.train_set =", "Part 2 — data: train and validation datasets, DataLoaders with `collate_batch`, "
             "`num_workers`, pinned memory on CUDA, persistent workers and per-worker seeding; training drops the last "
             "incomplete batch; validation uses batch size 1."),
            ("        self.model: MultiSensorFusionNet", "Part 3 — model and optimisation: build the network, enable "
             "gradient checkpointing, load initial weights from earlier stages, freeze modules, count parameters, "
             "AdamW (lr × stage `lr_scale`, weight decay), gradient accumulation, a LambdaLR schedule with linear "
             "warm-up followed by cosine decay over all optimiser steps, the AMP `GradScaler`, and optional resume."),
            ("    def _initialise_from_stages", "`_initialise_from_stages()`: for every `init_from` entry of the stage, "
             "loads `best.pt` (or `last.pt`) of the source stage and copies only the parameters whose names start "
             "with the given prefixes and whose shapes match (`load_model_weights`); a missing source checkpoint "
             "is logged and those weights are trained from scratch."),
            ("    def _freeze", "`_freeze()`: `trainable_only` freezes everything except the listed prefixes (stage 6 "
             "trains only `plan_head`); `frozen` freezes the listed prefixes (e.g. the three encoders in stage 4)."),
            ("    def _resume", "`_resume(path)`: restores model, optimiser, scheduler, scaler, epoch, step and the best score."),
            ("    def _score", "`_score(res)`: the model-selection criterion for `best.pt` — NDS for stages with a "
             "detection loss, −ADE for the trajectory stage, −validation loss otherwise."),
            ("    def _set_train_mode", "`_set_train_mode()`: `model.train()`, then every BatchNorm whose parameters are "
             "all frozen is put back into eval mode — otherwise the running statistics of \"frozen\" modules would "
             "keep drifting during later stages."),
            ("    def train_epoch", "`train_epoch(epoch)`: for each batch — move to device; forward and "
             "`compute_losses(active)` under autocast; divide by `grad_accum`; skip the batch (and log) if the loss "
             "is not finite; scaled backward; every `grad_accum` iterations unscale → clip the gradient norm "
             "(`train.grad_clip`) → optimiser step → scaler update → scheduler step → zero gradients. Running "
             "averages of every loss are logged every `log_interval` iterations with the learning rate, "
             "iterations/s and peak GPU memory."),
            ("    def fit", "`fit()`: epoch loop — train, validate every `val_interval` epochs with `evaluate_model`, log a "
             "summary line, save `best.pt` when the score improves and `last.pt` every epoch, and append a JSON record "
             "to `history.jsonl` (used for the dashboard's training curves)."),
            ("    def _save", "`_save(name, epoch, res)`: metrics without the per-class table + `save_checkpoint` with "
             "optimiser/scheduler/scaler state, the full configuration and `extra` (best score, experiment, stage)."),
        ],
    },
    {
        "file": "evaluation/detection_metrics.py",
        "intro": "In-memory port of the official nuScenes detection evaluation; cross-checked against the devkit "
                 "(max difference 3·10⁻⁴ on NDS/mAP/TP errors).",
        "blocks": [
            ('"""nuScenes detection metrics', "Docstring: algorithm summary and the official rule that classes without GT "
             "score AP = 0 (relevant on mini-val)."),
            ("CLASS_RANGE =", "Official constants: class-specific evaluation ranges (30–50 m), the four matching "
             "distances (0.5, 1, 2, 4 m), the 2 m threshold for TP metrics, minimum recall/precision 0.1, 500 boxes per "
             "sample, the five TP metrics and the 101-point recall grid."),
            ("def default_attribute", "`default_attribute(name, velocity)`: attributes are not predicted by the network; "
             "this heuristic assigns moving/parked (vehicles), moving/standing (pedestrians), with/without rider "
             "(cycles) from the predicted speed (> 0.2 m/s)."),
            ("def _angle_diff", "Helpers: `_angle_diff` (smallest absolute angle difference with a period — π for "
             "barriers, 2π otherwise), `_scale_iou` (IoU of two aligned boxes from their sizes), `_cummean` (NaN-aware "
             "cumulative mean; all-NaN → ones)."),
            ("class DetectionEvaluator", "`DetectionEvaluator.__init__`: per-class lists of GT and predictions."),
            ("    def add_sample", "`add_sample(...)`: keeps the 500 highest-scoring predictions, drops boxes beyond the "
             "class range, drops bicycles/motorcycles inside bicycle racks (both GT and predictions), drops GT with "
             "zero LiDAR+radar points and sets the GT velocity to NaN where it is undefined."),
            ("    def _accumulate", "`_accumulate(name, dist_th)` — the core:\n\n"
             "1. Predictions of the class sorted by score.\n"
             "2. Greedy matching: each prediction takes the nearest unmatched GT of the same sample (2-D centre "
             "distance); a match below the threshold is a TP (errors recorded: translation, velocity, scale, "
             "orientation, attribute), otherwise an FP.\n"
             "3. Cumulative TP/FP → precision and recall; precision and confidence are interpolated on the 101-point "
             "recall grid.\n"
             "4. Each TP-error list is turned into a cumulative mean and re-sampled at the interpolated confidences "
             "(official alignment).\n"
             "No GT or no TP → precision 0, errors 1."),
            ("    @staticmethod", "`_calc_ap`: mean of `max(precision − 0.1, 0)` above 10 % recall, normalised by 0.9. "
             "`_calc_tp`: mean TP error between 10 % recall and the maximum achieved recall (1.0 if that range is empty)."),
            ("    def evaluate", "`evaluate()`: per class AP at the four thresholds (mean = class AP) and TP errors at 2 m "
             "(NaN for metrics that do not apply: cone orientation/velocity/attribute, barrier velocity/attribute); "
             "mAP = mean AP over the 10 classes; mean TP errors (NaN-mean); "
             "`NDS = (5·mAP + Σ_metrics (1 − min(1, error))) / 10`; plus `mAP_present_classes` for context."),
        ],
    },
    {
        "file": "evaluation/trajectory_metrics.py",
        "intro": "Open-loop planning metrics against the logged human trajectory.",
        "blocks": [
            ('"""Planning metrics', "Docstring listing the metrics."),
            ("def _ego_box", "`_ego_box(xy, yaw, ego_cfg)`: ego bounding box from a rear-axle pose. "
             "`_polygons_intersect(a, b)`: separating-axis test for two convex quadrilaterals."),
            ("def trajectory_headings", "`trajectory_headings(xy)`: heading of each step from the previous point "
             "(0 if the ego barely moves)."),
            ("def collision_flags", "`collision_flags(pred, agent_future, collision_labels, ego_cfg)` → `(T,)`: at each "
             "step, the ego polygon is tested (SAT) against the **real** future boxes of the collision classes within "
             "12 m. Using the logged agent positions (not a forecast) makes this an objective collision measure; "
             "sanity check: the logged human trajectory scores 0 % collisions."),
            ("def sample_planning_metrics", "`sample_planning_metrics(pred, gt, mask, dt, …)`: ADE, FDE, L2 at 1/2/3 s, "
             "lateral and longitudinal error (in the GT trajectory's heading frame), mean jerk and curvature change of "
             "the prediction, collision flags overall and up to 1/2/3 s."),
            ("def aggregate", "`aggregate(metrics)`: mean of every metric over the samples that define it; "
             "`collision*` keys are renamed `collision_rate*`."),
        ],
    },
    {
        "file": "evaluation/evaluator.py",
        "intro": "Runs a model over a DataLoader and computes all metric families (used by the trainer, the ablation "
                 "runner and `scripts/evaluate.py`).",
        "blocks": [
            ('"""Run a trained model', "Docstring."),
            ("@torch.no_grad()", "`evaluate_model(model, loader, cfg, device, use_amp, compute_losses, "
             "run_baseline_planner, max_batches)`: eval mode; for each batch forward (timed per stage), optional "
             "validation losses, decoding at `eval.score_threshold`, feature-selection statistics; per sample the "
             "detections are added to the `DetectionEvaluator` with the *unfiltered* GT from the infos (the evaluator "
             "applies the official filters); for samples with a complete 3 s future the learned planner (and "
             "optionally the baseline planner fed with the detections) are scored. Returns detection metrics, "
             "planning metrics, FS stats, mean losses and median latency per stage."),
            ("def summary_line", "`summary_line(res)`: the one-line log summary (NDS, mAP, mAP present, mAVE, ADE, FDE, "
             "collision rate, latency)."),
        ],
    },
    {
        "file": "inference/pipeline.py",
        "intro": "Frame-by-frame processing shared by the dashboard and `scripts/inference.py` / `render_bev.py`.",
        "blocks": [
            ('"""Frame-by-frame inference', "Docstring: the three honest modes (dataset, ground_truth, model)."),
            ("from __future__", "Imports. `MACHINE_KEYS` lists configuration keys that must come from *this* machine "
             "rather than from the checkpoint (dataset location, device, output folders)."),
            ("@dataclass", "`FrameResult`: everything a view needs for one frame — info, mode, trained flag, raw LiDAR "
             "and radar points, detections (+ track ids and histories), plan, feature maps (energies, gates, "
             "attention), channel gates with names, selection statistics, frame metrics, timings and raw outputs "
             "(learned waypoints). `command_name` converts the command index to text."),
            ("def split_of_scene", "`split_of_scene(name)`: train / val / test membership (mini lists or the devkit "
             "lists) — used to warn when a training scene is shown."),
            ("class FusionPipeline", "Constructor: with a checkpoint, its embedded config is used and the machine keys "
             "are replaced by the local ones; loads the infos of the split (`'all'` in the dashboard), creates a "
             "dataset with `load_all_sensors=True, with_targets=False`, builds the model and loads weights (or a "
             "random model only when explicitly allowed), creates the tracker, planner cache and map prior."),
            ("    def scenes", "`scenes()` groups the infos into scenes (name, description, first index, frame count, "
             "location, split). `planner(name)` caches planner instances; `map_patch()` returns the map prior "
             "crop around the ego in display layout (warnings instead of failures)."),
            ("    def run(self", "`run(index, mode, planner, plan_on_gt, score_threshold, stage_callback)`:\n\n"
             "1. loads the dataset item (timed);\n"
             "2. resets the tracker unless this frame directly follows the previous one in the same scene;\n"
             "3. `model` mode → `_run_model`; `ground_truth` mode → GT boxes with ≥ 1 point become the "
             "\"detections\" (scores 1.0);"),
            ("        if res.detections is not None:", "4. tracking of confident detections (score ≥ "
             "`occupancy.min_score`; others get id 0) in the global frame;\n"
             "5. planning (in model mode, or on GT objects in mode 2 when enabled); learned/hybrid fall back to the "
             "baseline when no learned waypoints exist;\n"
             "6. frame metrics and total time."),
            ("    @torch.no_grad()", "`_run_model()`: collates the single item (without the raw point keys), runs the "
             "network under the stage timer, decodes detections, and extracts NumPy maps for the views: per-sensor raw "
             "and selected feature energies, the fused energy, the detection heatmap (max over classes), fusion "
             "attention weights, spatial gates, named channel gates, selection statistics, learned waypoints and the "
             "planning attention map."),
            ("    def _frame_metrics", "`_frame_metrics()`: single-frame NDS/mAP (model mode only) and planning metrics "
             "against the logged trajectory. `add_to_evaluator()` accumulates a frame into a running scene evaluator "
             "(dashboard \"scene so far\")."),
        ],
    },
    {
        "file": "app/streamlit_app.py",
        "intro": "The Streamlit dashboard. Streamlit re-executes this script from top to bottom on every interaction; "
                 "persistent state lives in `st.session_state` and heavy objects in `st.cache_resource`.",
        "blocks": [
            ('"""Multi-Sensor Autonomous', "Docstring: launch commands, the three modes and the no-fake-output rule."),
            ("from __future__", "Standard imports; `ROOT` (project root) is put on `sys.path` so `app/...` imports work "
             "when Streamlit runs the file directly."),
            ("def _inside_streamlit", "`_inside_streamlit()` checks whether the Streamlit runtime is active. When the "
             "file is started with plain `python app/streamlit_app.py`, the block below re-launches it through "
             "`streamlit.web.cli` with the dark theme — so both launch commands work."),
            ("import altair as alt", "Third-party and project imports (after the re-launch guard so that plain "
             "`python` does not import Streamlit components twice)."),
            ("st.set_page_config", "Page configuration, CSS injection, and constants: playback speeds, the mode labels, "
             "the proposed experiment id, and the two explanation texts shown in the faculty demo."),
            ("@st.cache_resource", "`get_pipeline(version, device, checkpoint, experiment, allow_untrained)`: cached "
             "`FusionPipeline` per argument combination (≤ 12 entries). A data-only pipeline disables the network's "
             "camera preprocessing (images for display are loaded separately)."),
            ("def ss_default", "`ss_default` initialises session-state keys once: slider position, playing flag, current "
             "scene, the frame-result cache, the explain toggle, scene-level evaluators and the comparison table."),
            ("st.sidebar.markdown(", "Sidebar: faculty toggle, "
             "nuScenes version (only versions present on disk), device (default from `-- --device`), demo mode, fusion "
             "method (✅ when a checkpoint exists — `find_checkpoint`), planner, untrained-network opt-in, confidence "
             "threshold, playback speed, display options and the main camera."),
            ("data_pipe = get_pipeline(", "Mode handling — the "
             "honesty rule: mode 3 uses the model only if a checkpoint exists (banner MODEL PREDICTION) or the user "
             "explicitly allowed an untrained network (red UNTRAINED banner); otherwise it falls back to ground truth "
             "with the banner GROUND TRUTH / DATASET VISUALIZATION and a warning. The scene selector resets the "
             "frame when the scene changes."),
            ("def cb_prev", "Button callbacks (run *before* the rerun, so they may change the slider's session key): "
             "previous/next frame, play (restarts at the end), pause, reset (also clears scene metrics and the "
             "tracker), load scene. The `_advance` flag set by the playback loop advances the slider at the top "
             "of the next run."),
            ("exp_label = labels_by_exp", "Header with title, "
             "scene description and banner; warnings (no checkpoint, training scene); control buttons and the "
             "key-frame slider; `index` = global info index of the selected frame."),
            ("flow_slot = st.empty()", "Processing: in faculty "
             "mode an empty placeholder holds the pipeline strip; stages that do not apply are marked skipped. "
             "`on_stage` (timer callback) re-renders the strip after every stage, so the audience sees the stages "
             "tick live."),
            ("cache_key =", "Frame results are cached per (method, index, mode, planner, plan_on_gt) in an LRU of 24 "
             "entries; otherwise `pipe.run(...)` is called. Then the scene-level running evaluator accumulates each "
             "frame once."),
            ("dets = res.detections", "KPI strip: number of "
             "objects, FPS (excluding data loading), frame and scene NDS / mAP, ADE, collision, fusion method and mean "
             "confidence."),
            ("colA, colB = st.columns", "Rows A–C: camera panel and the six "
             "sensor-fusion tiles; the autonomous-driving BEV and the trajectory panel (ego pose, command, close-up, "
             "planner source, collision chip, ADE/FDE/lateral/longitudinal/GT collision); the detection table and the "
             "ground-truth comparison (TP/FP/missed at 2 m)."),
            ("st.divider()", "\"Explain Feature "
             "Selection\": toggled by a button; available only for trained models with feature selection. Shows raw → "
             "importance → selected for LiDAR and radar, the sensor-contribution map, fusion attention, the detection "
             "heatmap, an Altair bar chart of the named channel gates, selected ratios and the per-class sensor "
             "trust table."),
            ("with st.expander(\"⚖️", "\"Compare fusion methods\": "
             "runs every trained method on the current frame or the whole scene and charts the chosen metric (proposed "
             "method highlighted)."),
            ("with st.expander(\"📊", "Ablation results "
             "(`outputs/ablation/results.json` + the PNG), training curves from `history.jsonl`, and the \"About\" "
             "text with the checkpoint metadata."),
            ("if ss.playing:", "Playback: while playing, "
             "sleep for the remainder of `0.5 s / speed` (nuScenes key-frames are 2 Hz), set `_advance` and "
             "`st.rerun()`."),
        ],
    },
    {
        "file": "main.py",
        "intro": "The single command-line entry point.",
        "blocks": [
            ('"""Single entry point', "Docstring = the usage text printed by `python main.py --help`."),
            ("from __future__", "`COMMANDS` maps sub-commands to scripts. `main()`: `app` and `test` start subprocesses "
             "(Streamlit, pytest); every other command puts the project root on `sys.path`, rewrites `sys.argv` and "
             "executes the script with `runpy.run_path(..., run_name='__main__')`, so the scripts behave exactly as "
             "if they were run directly."),
        ],
    },
]
