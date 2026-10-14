"""Narrative chapters, part A (orientation + sensor pipelines). `{{AUTO:name}}` lines are replaced by generated sections."""

CHAPTERS_A = [
# =====================================================================================================
("About This Document", r"""
This document is the technical handover of the project located in `D:\researcpaper\multi_sensor_fusion`
(nuScenes data in `D:\researcpaper`). It is written for the next developer or researcher, so that the project can be
run, understood, debugged, modified and extended without asking the original author.

**How it was produced.** The reference parts — directory tree, file inventory, every class and function signature,
parameter types, return types, docstrings, raised exceptions, file side effects, import graph and call graph,
configuration values, command-line arguments, test list and checkpoint metadata — are **generated from the source code
and the files on disk** by `docs/handover/build_handover.py` (static analysis with Python's `ast` module). The code
listings in the walkthrough chapter are read from the actual files at build time and each explanation is attached to an
exact line of code; the build fails if code and explanation do not match. Narrative chapters were written after
inspecting the complete code base. Measured numbers (validation, timings, metrics) come from runs performed on the
development machine (GTX 1650, 4 GB) and are labelled as such.

**Regenerating the document** after code changes:
```bash
C:\Users\shash\.venvs\msf_fusion\Scripts\python.exe docs/handover/build_handover.py
```

**Conventions used in this document**

- `monospace` = code identifiers, file paths, configuration keys (`dataset.root` means key `root` in section `dataset`).
- Tensor shapes are written `(B, C, H, W)`; the symbols are defined in the *Tensor Dimension Reference* chapter.
- Default numbers refer to the configuration actually used on the development machine
  (`configs/*.yaml` + `configs/hardware/gtx1650.yaml` + `configs/local.yaml`).
- "Not identifiable from the current codebase." marks information that cannot be determined from the code or files.

> **Scope note.** The request that defined this document was received truncated after its section 14
> ("Deep-Learning Feature Selection"). Sections 1–14 of the request are covered in full. The remaining chapters
> (fusion, detection, planning, losses, training, evaluation, application, configuration, testing, debugging,
> extension guide, artifacts, limitations) follow standard technical-handover practice; adjust them if the missing part
> of the request asked for something different.
"""),
# =====================================================================================================
("Executive Summary", r"""
## What the system does

The project is a research prototype for **multi-sensor perception and planning for autonomous driving** on the
**nuScenes** dataset. For every nuScenes key-frame (2 Hz) it:

1. loads the six surround cameras, the LIDAR_TOP point cloud (multiple sweeps) and the five radars (multiple sweeps),
   time-synchronised and calibrated into one vehicle frame;
2. converts the cameras into a **Bird's-Eye-View (BEV)** feature map with a Lift-Splat-Shoot network;
3. encodes LiDAR and radar into BEV feature maps on the same 200 × 200 grid (0.5 m cells, ±50 m);
4. applies a **learned, camera-guided feature-selection** network that gates LiDAR and radar features per channel and per
   BEV cell (the research contribution);
5. fuses the (selected) BEV maps, detects 3-D objects of 10 classes with velocity (CenterPoint-style head);
6. tracks the objects and plans a 3-second ego trajectory (classical A* + speed profile + iLQR MPC, or a learned planner);
7. visualises everything in a Streamlit dashboard built for a faculty demonstration (scene replay, explainability).

## Status at hand-over

| Area | Status |
|---|---|
| Data pipeline, calibration, synchronisation | Implemented and validated on nuScenes-mini (LiDAR-in-box agreement r = 0.9997) |
| Camera BEV, LiDAR, radar encoders, feature selection, fusion, detection, learned planner | Implemented, trained on nuScenes-mini (6 experiments × 20 epochs) |
| Classical planner, tracker | Implemented; planner beats constant velocity with GT perception (ADE 1.65 vs 1.77 m, collisions 1.4 % vs 14.5 %) |
| Metrics | nuScenes NDS/mAP port matches the official devkit to 3·10⁻⁴; planning metrics with a calibrated collision check |
| 7-stage training curriculum | Implemented and mechanically verified (all stages run, weights hand over); not trained to convergence |
| Dashboard | Three modes, replay, explainability, method comparison — tested headlessly and in the browser |
| Full nuScenes (trainval) | Supported by the code; **never run** (needs a larger GPU and the full dataset) |
| Tests | 29 automated tests, all passing |

## Key results (nuScenes-mini, 20 epochs, final weights, 81 validation frames, one seed)

| Experiment | NDS | mAP | velocity error (m/s) | planning ADE (m) |
|---|---|---|---|---|
| A LiDAR only | 0.060 | 0.032 | 1.10 | 1.37 |
| B Radar only | 0.053 | 0.007 | 1.48 | 1.26 |
| C LiDAR + radar, concatenation | **0.116** | 0.035 | **0.77** | 1.36 |
| D LiDAR + radar, attention | 0.070 | 0.021 | 1.42 | 1.28 |
| E LiDAR + radar, feature selection | 0.055 | 0.027 | 1.03 | 1.51 |
| F Camera + LiDAR + radar, feature selection (proposed) | 0.094 | **0.044** | 0.91 | 1.53 |

Radar improves LiDAR (C vs A: +0.056 NDS, −0.33 m/s velocity error) and the camera BEV improves LiDAR+radar (F vs E:
+0.039 NDS). **Learned feature selection did not beat plain concatenation on nuScenes-mini** (E vs C: −0.061 NDS); the
hypothesis remains untested at a meaningful data scale. Absolute numbers are low because nuScenes-mini contains only
323 training frames.

## Most important things to know

- **Environment**: Python 3.11 virtual environment at `C:\Users\shash\.venvs\msf_fusion` (outside the project folder).
- **Data**: `configs/local.yaml` sets `dataset.root: D:/researcpaper` and `runtime.hardware_profile: gtx1650`.
- **fp16 is disabled on the GTX 1650** because cuDNN 9.10 returns NaN for fp16 3×3 convolutions with 256 input
  channels on that GPU (`utils/device.py` tests this at start-up for every GPU).
- **One command per task**: `python main.py <command>` (see *Command-Line Reference*).
- **Every checkpoint embeds its full configuration**; evaluation and the dashboard rebuild the exact model from it.
"""),
# =====================================================================================================
("Getting Started in Ten Minutes", r"""
## 1. Environment

The tested environment is Windows 11, Python 3.11, NVIDIA driver 566 (CUDA 12.7), GTX 1650. The virtual environment
already exists on the development machine:

```bash
C:\Users\shash\.venvs\msf_fusion\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On a new machine:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install nuscenes-devkit --no-deps
```

`nuscenes-devkit` is optional (official evaluation and the full train/val scene lists) and is installed without
dependencies because it pins a matplotlib version without a Python 3.11 wheel; its real runtime dependencies are listed
in `requirements.txt`.

## 2. Dataset

The code expects the standard nuScenes layout under `dataset.root`:

```text
<root>/samples/   <root>/sweeps/   <root>/maps/   <root>/v1.0-mini/   [<root>/v1.0-trainval/]   [<root>/can_bus/]
```

Set the root without touching code — `configs/local.yaml` (already set to `D:/researcpaper`), the environment variable
`NUSCENES_ROOT`, or `--set dataset.root=...` on any command. Then build the caches:

```bash
python main.py preprocess --maps
```

## 3. First commands

| Goal | Command | Output |
|---|---|---|
| Check calibration / sync | `python main.py validate` | `outputs/validation/report.json` + images |
| Check every model config | `python main.py smoke --experiments all --quiet` | shapes, losses, memory, latency |
| Start the dashboard | `python app/streamlit_app.py` | browser at http://localhost:8501 |
| Train the proposed model | `python main.py train --stage joint --experiment F_cam_lidar_radar_fs --epochs 20` | `checkpoints/F_cam_lidar_radar_fs/joint/` |
| Evaluate a checkpoint | `python main.py evaluate --checkpoint checkpoints/F_cam_lidar_radar_fs/joint/last.pt` | `outputs/evaluation/.../report.json` |
| Run the ablation | `python main.py ablation --train --epochs 20` | `outputs/ablation/` |
| Render all BEVs of a frame | `python main.py bev --scene scene-0916 --frame 20` | `outputs/bev/*.jpg` |
| Export a replay video | `python main.py infer --scene scene-0103 --video` | `outputs/inference/...` |
| Run the tests | `python main.py test` | 29 tests |

All commands are run from `D:\researcpaper\multi_sensor_fusion` with the virtual environment's Python
(`C:\Users\shash\.venvs\msf_fusion\Scripts\python.exe`).

## 4. Where things are written

| Folder | Content |
|---|---|
| `outputs/cache/` | per-sample info caches (`infos_<version>_<split>_<hash>.pkl`), reduced map rasters |
| `outputs/logs/<experiment>/<stage>/` | `run.log`, `history.jsonl` |
| `checkpoints/<experiment>/<stage>/` | `best.pt`, `last.pt` (shared single-sensor stages under `_shared/`) |
| `outputs/ablation/` | results table, CSV/JSON, plot |
| `outputs/validation/`, `outputs/bev/`, `outputs/inference/` | images and videos |
"""),
# =====================================================================================================
("Project File Structure & Responsibilities", r"""
The tree below is generated from the files on disk (generated data folders such as `outputs/` and `checkpoints/` are
summarised; their content is listed in the *Artifacts* chapter). The following table lists **every file** with its
purpose, inputs, outputs and dependencies (dependencies of Python files are taken from their import statements: project
modules first, then third-party packages).

{{AUTO:tree}}

## File inventory

{{AUTO:inventory}}

## Package dependency graph

Which project packages import which (derived from the import statements):

{{AUTO:package_graph}}
"""),
# =====================================================================================================
("System Architecture", r"""
![Figure 1 – System architecture as implemented (class / file names and default tensor shapes).](docs/handover/build/architecture.png)

The architecture has five layers.

1. **Data layer** (`datasets/`). `NuScenesDB` reads the nuScenes tables; `info_builder` turns them into cached,
   self-contained per-sample records ("infos"); three loaders read the sensor files; `NuScenesFusionDataset` assembles
   training/inference items and `collate_batch` builds batches.
2. **Sensor encoders** (`models/camera_bev`, `models/lidar_encoder`, `models/radar_encoder`). Each produces a
   `(B, 64, 200, 200)` BEV map on the *same* grid in the *same* reference frame (ego vehicle at the LiDAR key-frame
   time). LiDAR and radar outputs end with explicit, named physical channels.
3. **Feature selection and fusion** (`models/feature_selection`, `models/fusion`). The selection network gates LiDAR and
   radar features, using the camera BEV as a guide; the fusion module combines camera, LiDAR and radar maps
   (`concat` or `attention_gated`, chosen by configuration).
4. **Heads** (`models/fusion/bev_backbone.py`, `models/detection`, `models/trajectory`). A SECOND-FPN backbone feeds a
   CenterPoint detection head and a learned planning head. `models/fusion_model.py` wires everything and computes the
   total loss.
5. **Downstream and application** (`planning/`, `inference/`, `visualization/`, `app/`, `scripts/`). Tracking and the
   classical/learned/hybrid planners, the frame pipeline, rendering, the Streamlit dashboard and the CLI tools.
   Training and evaluation live in `training/` and `evaluation/`.

## Module map

| Component | File | Main class / function | Output |
|---|---|---|---|
| Table access | `datasets/nuscenes_db.py` | `NuScenesDB` | indexed tables |
| Sample infos | `datasets/info_builder.py` | `load_infos`, `build_sample_info` | list of info dicts (cached) |
| Camera input | `datasets/camera_loader.py` | `CameraLoader`, `lidar_depth_maps` | images + calibration, depth targets |
| LiDAR input | `datasets/lidar_loader.py` | `LidarLoader` | `(P, 5)` points |
| Radar input | `datasets/radar_loader.py` | `RadarLoader`, `read_radar_pcd` | `(Q, 10)` points |
| Dataset | `datasets/nuscenes_dataset.py` | `NuScenesFusionDataset`, `collate_batch` | item / batch dicts |
| Camera BEV | `models/camera_bev/lss.py` | `CameraBEVEncoder` | `(B, 64, 200, 200)` |
| LiDAR BEV | `models/lidar_encoder/pillar_encoder.py` | `LidarEncoder` | `(B, 64, 200, 200)` |
| Radar BEV | `models/radar_encoder/radar_encoder.py` | `RadarEncoder` | `(B, 64, 200, 200)` |
| Feature selection | `models/feature_selection/selection.py` | `CrossModalFeatureSelection` | selected maps + gates |
| Fusion | `models/fusion/fusion_methods.py` | `AttentionGatedFusion`, `ConcatFusion` | `(B, 128, 200, 200)` |
| BEV backbone | `models/fusion/bev_backbone.py` | `SECONDFPN` | `(B, 256, 100, 100)` |
| Detection | `models/detection/*.py` | `CenterHead`, `decode_detections` | boxes `(K, 9)` |
| Learned planner | `models/trajectory/planning_head.py` | `PlanningHead` | waypoints `(B, 6, 2)` |
| Whole network | `models/fusion_model.py` | `MultiSensorFusionNet` | output dict, losses |
| Tracking | `planning/tracking.py` | `MultiObjectTracker` | track ids |
| Planning | `planning/trajectory_planner.py` | `BaselinePlanner`, `LearnedPlanner`, `HybridPlanner` | `PlanResult` |
| Frame pipeline | `inference/pipeline.py` | `FusionPipeline` | `FrameResult` |
| Application | `app/streamlit_app.py` | (script) | web UI |
"""),
# =====================================================================================================
("End-to-End Data Flow", r"""
This chapter follows **one nuScenes sample** through the system as implemented in `FusionPipeline.run()` (dashboard and
scripts). Training differs only in that augmentation is random, targets are built and losses are computed (described at
the end).

![Figure 2 – Data flow of one sample with the functions that perform each step and the resulting data.](docs/handover/build/dataflow.png)

## Step by step

1. **Scene → sample → info.** The dashboard lists scenes from the cached infos (`FusionPipeline.scenes()`); the selected
   key-frame is an index into the info list. The info contains relative file paths of all sensor files, every
   calibration matrix, GT boxes already in the reference frame, ego state, ego future and future agents.
2. **Dataset item.** `NuScenesFusionDataset.__getitem__(index)`:
   - `_sample_bda()` → identity at inference (random flip/rotate/scale in training);
   - `CameraLoader.load()` → six images resized/cropped to 128 × 352 (GTX 1650 profile), normalised, plus `K`,
     `post_rot`, `post_trans`, `cam2ref`, `cam_valid`;
   - `LidarLoader.load()` → up to 5 sweeps (key-frame + 4 previous) transformed to the reference frame with a time-lag
     column, range-filtered, ≤ 150 000 points → `(P, 5)`;
   - `RadarLoader.load()` → 5 radars × 5 sweeps, filtered, positions and velocities in the reference frame → `(Q, 10)`;
   - planning inputs: ego state `[speed, yaw rate, accel]`, command, logged future.
3. **Batch.** `collate_batch([item])` stacks arrays (`B = 1`) and concatenates point clouds with a batch-index column.
4. **Network.** `MultiSensorFusionNet.forward(batch)` runs camera BEV → LiDAR encoder → radar encoder → feature selection
   → fusion → SECOND-FPN → detection head → planning head, timing each stage.
5. **Decoding.** `decode()` → boxes `[x, y, z, l, w, h, yaw, vx, vy]`, scores, labels (score ≥ 0.1, circle NMS).
6. **Tracking.** Detections with score ≥ 0.3 are associated with tracks in the global frame → track ids and trails.
7. **Planning.** `BaselinePlanner.plan()` (default) builds occupancy from the detections and the LiDAR obstacles, finds
   an A* path, selects a speed profile and smooths with iLQR → six waypoints at 0.5 s steps and a collision flag.
8. **Metrics.** Single-frame NDS/mAP against the GT of the frame and ADE/FDE/collision against the logged future.
9. **Visualisation.** `app/ui/panels.py` renders camera images with boxes, the BEV tiles, the driving BEV, the
   trajectory close-up and tables; the dashboard displays them.

## Training variant

During training (`Trainer.train_epoch`) the same dataset produces, in addition: depth targets `depth_gt (6, 8, 22)`,
CenterPoint targets on the 100 × 100 head grid (`heatmap`, `ind`, `mask`, `vel_mask`, `reg_target`), the class-agnostic
foreground heatmap `fg_heatmap (1, 200, 200)` and the GT future occupancy `future_occ (6, 200, 200)`.
`compute_losses()` combines detection, trajectory, feature-selection, fusion (deep supervision) and depth losses with the
configured weights; the optimiser updates the trainable parameters.
"""),
# =====================================================================================================
("Program Execution Flow", r"""
## Dashboard (`python app/streamlit_app.py`)

![Figure 3 – Execution flow of the Streamlit dashboard. Streamlit re-executes the script on every interaction.](docs/handover/build/exec_dashboard.png)

Important properties of this flow:

- **Re-execution model.** Every click re-runs the whole script. Expensive objects are created once through
  `st.cache_resource` (`get_pipeline`: data, model and checkpoint) and frame results are cached in
  `st.session_state.results` (LRU, 24 entries). Button callbacks modify the slider state *before* the rerun.
- **Honest mode selection.** Mode 3 uses the network only when a checkpoint exists (or untrained weights are explicitly
  allowed, with a red banner); otherwise ground truth is shown with the banner "GROUND TRUTH / DATASET VISUALIZATION".
- **Playback.** While playing, the script ends with `sleep(0.5 s / speed − render time)` and `st.rerun()`; the next run
  advances the slider.

## Training (`python main.py train ...`)

![Figure 4 – Execution flow of one training stage.](docs/handover/build/exec_training.png)

## Evaluation, ablation and scripts

| Command | Flow |
|---|---|
| `main.py evaluate` | load checkpoint → rebuild model from embedded config (local data paths) → `evaluate_model` on the validation split → optional official devkit → JSON report |
| `main.py ablation` | for each experiment: optional `Trainer(..., 'joint').fit()` → `evaluate_checkpoint(last.pt)` → result row → tables, hypothesis deltas, plot |
| `main.py infer` | `FusionPipeline(checkpoint)` → for each frame of the scene: `run` → compose 1920×1080 frame → JPEG (+ MP4) → summary |
| `main.py validate` | load infos → sync report → LiDAR/radar agreement → augmentation round trip → verification images |
| `main.py smoke` | per experiment: dataset → one training step → timed inference → table |
| `main.py preprocess` | load tables → build/cache infos for train, val, all → optional map caches |
"""),
# =====================================================================================================
("Tensor Dimension Reference", r"""
## Symbols

| Symbol | Meaning | Default value |
|---|---|---|
| B | batch size | 1 on the GTX 1650 (`train.batch_size`); 1 at inference |
| N | number of cameras | 6 |
| H, W | network image height/width | 128, 352 (GTX 1650); 256, 704 (RTX profiles) |
| h, w | image feature height/width = H/16, W/16 | 8, 22 |
| D | depth bins | 59 (1–60 m, 1 m steps) |
| C | channels (context / encoder output) | 64 |
| ny, nx | BEV grid rows (y) / columns (x) | 200, 200 (0.5 m cells, ±50 m) |
| P | number of LiDAR points in a batch | ≈ 100 000–150 000 per sample (dynamic) |
| Q | number of radar points in a batch | ≈ 400–900 per sample (dynamic) |
| U | number of occupied pillars | dynamic (`torch.unique`) |
| K | number of detections | ≤ 300 (`max_detections`), dynamic after thresholding |
| M | number of fused modalities | 1–3 (configuration) |
| T | planning horizon | 6 steps of 0.5 s |
| L | planning-head tokens | 13 × 13 = 169 |

## Module-by-module shapes (proposed model F, GTX 1650 profile)

| Module | Input | Shape | Output | Shape |
|---|---|---|---|---|
| CameraLoader | 6 JPEG images 1600×900 | – | `images` | (B, 6, 3, 128, 352) |
| ImageBackbone | images flattened | (6B, 3, 128, 352) | neck features | (6B, 256, 8, 22) |
| DepthNet | features + camera params | (6B, 256, 8, 22), (6B, 16) | depth logits / context | (6B, 59, 8, 22) / (6B, 64, 8, 22) |
| Lift | softmax(depth) ⊗ context | – | lifted features | (B, 6, 59, 8, 22, 64) |
| get_geometry | frustum + matrices | (59, 8, 22, 3) | point positions | (B, 6, 59, 8, 22, 3) |
| voxel_pool + refine | lifted + positions | – | camera BEV `F_cam` | (B, 64, 200, 200) |
| LidarLoader → collate | sweeps | – | `lidar_points` | (P, 6) = batch, x, y, z, intensity, dt |
| DynamicPillarVFE (LiDAR) | decorated points | (P, 10) | pillar features | (U, 64) → canvas (B, 64, 200, 200) |
| LidarEncoder | canvas + raster | (B, 64+4, 200, 200) | `F_lidar` | (B, 64, 200, 200) |
| RadarLoader → collate | sweeps | – | `radar_points` | (Q, 11) |
| DynamicPillarVFE (radar) | decorated points | (Q, 15) | pillar features | (U, 64) → canvas |
| RadarEncoder | canvas + raster | (B, 64+5, 200, 200) | `F_radar` | (B, 64, 200, 200) |
| Feature selection: channel branch | pooled statistics | (B, 384) | channel gates | (B, 64) per sensor |
| Feature selection: spatial branch | descriptors | (B, 30, 200, 200) | spatial gates | (B, 1, 200, 200) per sensor |
| Feature selection: output | F_lidar, F_radar | (B, 64, 200, 200) | selected | (B, 64, 200, 200) |
| AttentionGatedFusion | 3 maps | (B, 64, 200, 200) × 3 | fused, weights | (B, 128, 200, 200), (B, 3, 200, 200) |
| SECONDFPN | fused | (B, 128, 200, 200) | BEV features | (B, 256, 100, 100) |
| CenterHead | BEV features | (B, 256, 100, 100) | heatmap, reg, height, dim, rot, vel | (B, C_k, 100, 100) with C_k = 10, 2, 1, 3, 2, 2 |
| decode_detections | head maps | – | boxes, scores, labels | (K, 9), (K,), (K,) |
| PlanningHead | BEV + ego state + command | (B, 256, 100, 100), (B, 3), (B,) | waypoints, attention | (B, 6, 2), (B, 13, 13) |
| BaselinePlanner | detections | (K, 9) | waypoints | (6, 2) |

## Training targets

| Key | Shape | Meaning |
|---|---|---|
| `depth_gt` | (B, 6, 8, 22) | nearest LiDAR depth per image feature cell (0 = none) |
| `heatmap` | (B, 10, 100, 100) | Gaussian class heatmaps (peak = 1) |
| `ind`, `mask`, `vel_mask` | (B, 500) | flat centre index, object mask, velocity-defined mask |
| `reg_target` | (B, 500, 10) | dx, dy, z, log l, log w, log h, sin, cos, vx, vy |
| `fg_heatmap` | (B, 1, 200, 200) | class-agnostic centre heatmap at BEV resolution |
| `ego_future`, `ego_future_mask` | (B, 6, 2), (B, 6) | logged trajectory in the reference frame |
| `future_occ` | (B, 6, 200, 200) | GT occupancy of other agents at each future step |
| `ego_state`, `command` | (B, 3), (B,) | [speed, yaw rate, accel]; 0 right / 1 left / 2 straight |

## Dynamic shapes

- **P, Q** depend on the scene and the number of sweeps; points outside the grid are dropped and LiDAR is randomly
  subsampled above `dataset.lidar.max_points`.
- **U** is computed with `torch.unique` on the flat pillar index each forward pass; the canvas size is fixed.
- **K** depends on `score_threshold` and NMS; the evaluator caps at 500 per sample.
- **M** and the feature-selection sizes depend on the enabled sensors: `stat_dim = 2·64·(#inputs)`,
  spatial input `10·(#inputs)` channels, `#selectable` output maps.
- Image sizes and the BEV grid follow the configuration; the frustum and grids are rebuilt from it.
"""),
# =====================================================================================================
("Coordinate Systems & Transformations", r"""
![Figure 5 – Frames and transformation chain as implemented.](docs/handover/build/frames.png)

## Frames

| Frame | Definition | Where it appears |
|---|---|---|
| Camera | x right, y down, z forward (optical axis); pinhole intrinsics `K` | `camera_intrinsic` of calibrated_sensor |
| LiDAR (LIDAR_TOP) | sensor frame of the spinning LiDAR (roof, z ≈ 1.84 m above ground) | `.pcd.bin` points |
| Radar (RADAR_*) | sensor frame of each radar; Doppler velocities `vx_comp, vy_comp` also in this frame | `.pcd` points |
| Ego vehicle | x forward, y left, z up; origin at the rear-axle centre on the ground | `ego_pose` at a timestamp |
| Global | fixed metric map frame per location | annotations, ego poses, map rasters |
| **Reference** | ego frame at the LIDAR_TOP key-frame timestamp | all model inputs, GT boxes, predictions, plans |
| Augmented BEV | reference frame after `bda` (training only) | network inputs/targets in training |
| BEV grid | cells `(iy, ix)`, rows = y, columns = x | all `(…, ny, nx)` tensors |

## Homogeneous transformations

A rigid transform from frame *a* to frame *b* is a 4 × 4 matrix with rotation `R` (from a unit quaternion
`[w, x, y, z]`, `utils/geometry.py: quat_to_rotmat`) and translation `t`:

$$T_{a\rightarrow b} = \left[\,R\;\;t\,\right],\qquad \tilde{p}_b = T_{a\rightarrow b}\,\tilde{p}_a,\qquad \tilde{p} = [x, y, z, 1]^{T}$$

The inverse is computed analytically (`invert_transform`): rotation `Rᵀ`, translation `−Rᵀt`.

## Sensor → reference frame (time synchronisation)

Each sensor sample has its own timestamp and ego pose. The chain, computed once per sweep in
`datasets/info_builder.py: _sensor_record`, is:

$$T_{s\rightarrow ref} = \left(T^{(t_{ref})}_{ego\rightarrow global}\right)^{-1}\;T^{(t_{s})}_{ego\rightarrow global}\;T_{s\rightarrow ego}$$

- `T_s→ego` = calibrated_sensor (extrinsics);
- `T_ego→global(t_s)` = ego_pose at the sensor's own timestamp;
- `T_ego→global(t_ref)` = ego_pose at the LIDAR_TOP key-frame.

Going through the global frame with each sensor's own pose compensates the ego motion between capture times (measured
offsets on mini: cameras ≤ 45 ms, radars ≤ 37 ms from the LiDAR key-frame). Sweeps additionally carry
`time_lag = t_ref − t_s` (seconds) as a feature.

## Ground truth

Annotations are global boxes; `transform_boxes(boxes, inv(T_ego→global(t_ref)))` moves them to the reference frame:
centres as points, heading and velocity as vectors. Yaw is re-extracted from the rotated heading vector,

$$\psi = \mathrm{atan2}(R_{10},\,R_{00})$$

which equals the devkit's `quaternion_yaw`. nuScenes sizes `[w, l, h]` are converted to `[l, w, h]`.

## Radar velocity

Radar velocities are vectors, so only the rotation is applied (`radar_loader.py: load_sweep`):

$$v_{ref} = R^{(2\times 2)}_{s\rightarrow ref}\;v_{s}$$

## Camera projection and lifting

Projection of a reference-frame point (validation, drawing, depth targets):

$$p_c = T_{ref\rightarrow c}\,p_{ref},\qquad [u, v]^{T} = \frac{1}{z_c}\,(K\,p_c)_{1:2}$$

Image augmentation is an affine map in pixel space (`camera_loader.py: image_aug_matrices`):

$$[u_a, v_a, d]^{T} = R_{post}\,[u, v, d]^{T} + t_{post}$$

The LSS encoder lifts every frustum point `(u_a, v_a, d)` back (`lss.py: get_geometry`):

$$[u', v', d]^{T} = R_{post}^{-1}\left([u_a, v_a, d]^{T} - t_{post}\right),\qquad p_{ref} = R_{c\rightarrow ref}\,K^{-1}\,[u'd,\;v'd,\;d]^{T} + t_{c\rightarrow ref}$$

followed by the BEV augmentation `p_bev = bda · p_ref`. Verified: projection followed by lifting returns the original
point with a maximum error of 6·10⁻¹⁴ m (`scripts/validate_sensors.py`, `tests/test_dataset.py`).

## BEV grid

$$i_x = \left\lfloor \frac{x - x_{min}}{r} \right\rfloor,\qquad i_y = \left\lfloor \frac{y - y_{min}}{r} \right\rfloor,\qquad r = 0.5\,\mathrm{m}$$

Tensors are `(…, ny, nx)`. For display, `visualization/common.py: bev_to_display` rotates arrays so that forward points
up and left points left. The detection head uses `grid.downsample(2)` (1 m cells, 100 × 100); the planner uses its own
grid (x ∈ [−10, 60), y ∈ [−30, 30), 0.5 m).

## Where each transformation happens

| Transformation | Code |
|---|---|
| quaternion → matrix, 4×4 build/invert | `utils/geometry.py` |
| sensor → reference (all sensors) | `datasets/info_builder.py: _sensor_record` |
| LiDAR points → reference | `datasets/lidar_loader.py: LidarLoader.load_sweep` |
| radar points + velocities → reference | `datasets/radar_loader.py: RadarLoader.load_sweep` |
| GT boxes global → reference | `datasets/info_builder.py: _boxes_for_sample` (`transform_boxes`) |
| ego future / past → reference | `datasets/info_builder.py: build_sample_info` |
| BEV augmentation | `datasets/nuscenes_dataset.py` (`bev_rotation_matrix`, `augment_boxes_bev`, `_apply_bda_points`), `lss.py` |
| image augmentation matrices | `datasets/camera_loader.py: image_aug_matrices` |
| projection into images | `visualization/camera_view.py: project_points`, `camera_loader.lidar_depth_maps` |
| frustum lifting | `models/camera_bev/lss.py: get_geometry` |
| metres ↔ grid cells | `utils/bev_grid.py`, `models/common/pillars.py`, `lss.voxel_pool`, `utils/raster.py` |
| reference → global (submission) | `evaluation/official_eval.py: box_to_global` |
| reference ↔ global (tracking) | `planning/tracking.py` |
| map raster → display | `datasets/map_prior.py: crop_to_ego` |
"""),
# =====================================================================================================
("Camera → BEV Pipeline", r"""
![Figure 6 – Camera BEV pipeline (Lift-Splat-Shoot with a camera-aware depth network).](docs/handover/build/lss.png)

| Stage | File · class / function | Shape (GTX 1650 profile) | Operation |
|---|---|---|---|
| Image loading | `camera_loader.py · CameraLoader.load` | 6 × (3, 128, 352) | resize to width 352 (factor 0.22), bottom-biased crop of 128 rows, ImageNet normalisation; training: resize jitter 0.94–1.11, crop jitter, flip, ±5.4° rotation |
| Calibration | `camera_loader.py · image_aug_matrices`; info `sensor2ref` | `K` (6, 3, 3), `post_rot` (6, 3, 3), `post_trans` (6, 3), `cam2ref` (6, 4, 4) | affine image-augmentation matrices; extrinsics + ego motion |
| Backbone | `image_backbone.py · ImageBackbone.forward` | (6B, 256, 8, 22) | ResNet18 (ImageNet), stem + layer1 frozen, BN in eval mode; layer3 ⊕ upsampled layer4 → neck |
| Camera descriptor | `lss.py · CameraBEVEncoder.camera_params` | (6B, 16) | normalised fx, fy, cx, cy of `post_rot·K` + 12 extrinsic entries |
| Depth / context | `lss.py · DepthNet.forward` | (6B, 59, 8, 22), (6B, 64, 8, 22) | camera-aware SE gates; depth logits and context features |
| Lift | `lss.py · CameraBEVEncoder.forward` | (B, 6, 59, 8, 22, 64) | `softmax(depth) ⊗ context` (outer product per cell) |
| 3-D projection | `lss.py · get_geometry` | (B, 6, 59, 8, 22, 3) | frustum → reference frame (equations in the previous chapter) |
| BEV pooling | `lss.py · voxel_pool` | (B, 64, 200, 200) | `index_add_` sum-pooling into cells, z collapsed |
| Refinement | `lss.py · bev_refine` | (B, 64, 200, 200) | ConvBNReLU + 2 ResBlocks |
| Depth supervision | `camera_loader.py · lidar_depth_maps`; `lss.py · depth_loss` | (B, 6, 8, 22) | nearest LiDAR point per cell; BCE to the one-hot depth bin |

**Mathematical core.** For feature cell `(i, j)` of camera `n` with depth distribution `α_d` and context `c`, the lifted
feature at depth `d` is `α_d · c`; its 3-D location is the lifted frustum point; all lifted features that fall into BEV
cell `(iy, ix)` are summed:

$$F_{cam}[:, i_y, i_x] = \sum_{n, d, i, j \,:\, cell(p_{n,d,i,j}) = (i_y, i_x)} \alpha_{n,d,i,j}\; c_{n,i,j}$$

**Design notes.**
- The frustum uses bin *centres* and feature-cell *centre* pixels; the depth targets use the same binning, so geometry
  and supervision are consistent.
- A missing or unreadable camera image (`cam_valid = False`) multiplies its depth distribution by zero, so it adds
  nothing to the BEV.
- The camera BEV is bright mostly near the vehicle: many more frustum rays fall into near cells (sum pooling). This is a
  property of LSS, not an error.
- The dashboard additionally shows a *geometric* camera BEV (`visualization/camera_view.py: camera_ipm_bev`): inverse
  perspective mapping of the ground plane using only the calibration (no learning, flat-ground assumption, ±25 m). It
  works before training and illustrates what the learned depth distribution improves on.
"""),
# =====================================================================================================
("LiDAR Pipeline", r"""
| Stage | Code | Details |
|---|---|---|
| Raw LiDAR | `lidar_loader.py · read_lidar_bin` | `.pcd.bin`, float32 × 5: x, y, z, intensity (0–255), ring index |
| Sweep selection | `info_builder.py · build_sample_info` | key-frame + previous sweeps via the `prev` chain (up to 10 cached; `dataset.lidar.sweeps` = 5 on the GTX 1650, 10 on RTX profiles) |
| Ego-return removal | `LidarLoader.load_sweep` | drop points within 1.5 m (xy) of the sensor |
| Coordinate transformation | `LidarLoader.load_sweep` | `sensor2ref` per sweep (motion compensation through the sweep's ego pose); `time_lag` column |
| Point filtering | `LidarLoader.load` | keep x, y ∈ [−50, 50), z ∈ [−5, 5); random subsampling above 150 000 points (seeded at evaluation) |
| Batching | `collate_batch` | `(P, 6)` = batch index, x, y, z, intensity, time_lag |
| Pillarisation | `pillars.py · PillarIndexer` | cell indices, unique pillars, inverse index (dynamic, no point cap) |
| Point decoration | `DynamicPillarVFE.forward` | `[x/50, y/50, z/5, intensity/255, dt/0.5, Δ to pillar mean (3), Δ to pillar centre (2)]` → 10 features |
| PointNet | `DynamicPillarVFE` | Linear(10→16)-BN-ReLU → max per pillar → concat → Linear(32→64)-BN-ReLU → max per pillar |
| Scatter to BEV | `DynamicPillarVFE.forward` | `(U, 64)` → canvas `(B, 64, 200, 200)` |
| Raster channels | `LidarEncoder.raster` | density `log(1+n)/3`, max height / 3 m, mean height / 3 m, mean intensity / 100 |
| Encoder | `LidarEncoder.forward` | ConvBNReLU(68 → 60) + 2 ResBlocks, then concatenate the 4 raster channels → `F_lidar (B, 64, 200, 200)` |

**Why dynamic pillars.** The original PointPillars caps the points per pillar and the number of pillars. Multi-sweep
nuScenes clouds (≈ 110 000 points for 5 sweeps) are dense near the vehicle, and caps would discard information. Scatter
operations (`index_add_`, `scatter_reduce_`) handle any number of points in pure PyTorch, so no custom CUDA extension
(spconv/mmcv) has to be compiled on Windows.

**Why explicit raster channels.** The last four output channels are interpretable physical quantities. The
feature-selection channel gates on these channels can therefore be read as "how much does the network use LiDAR height /
density / intensity". On the trained model F the gate of *mean height* is 0.001 (switched off) while *max height* is
0.98.

**Robustness.** A missing or corrupt sweep file is skipped with a single warning; with no LiDAR at all the encoder
returns an all-zero map, and the rest of the network still runs.
"""),
# =====================================================================================================
("Radar Pipeline", r"""
## nuScenes radar fields

The radar files are binary PCD files with 18 fields per point (43 bytes, packed). `read_radar_pcd` parses all of them;
the following are **used**:

| Field | Meaning | Use |
|---|---|---|
| `x, y, z` | position in the radar frame (z is unreliable) | transformed to the reference frame |
| `rcs` | radar cross-section (dBsm) | feature (÷30) and raster channel |
| `vx_comp, vy_comp` | Doppler velocity compensated for ego motion | rotated to the reference frame; feature (÷10) and raster channels |
| `vx, vy` | raw (uncompensated) velocity | rotated; feature (÷10) |
| `dyn_prop` | dynamic property (0 moving … 7 stopped) | filter and feature (÷7) |
| `invalid_state` | validity flag | filter (keep 0) |
| `ambig_state` | Doppler ambiguity state | filter (keep 3 = unambiguous) |

Not used: `id`, `is_quality_valid`, `x_rms`, `y_rms`, `vx_rms`, `vy_rms`, `pdh0` (parsed but ignored).

## Processing chain

| Stage | Code | Details |
|---|---|---|
| Filtering | `RadarLoader._filter` | devkit defaults: `invalid_state ∈ {0}`, `dyn_prop ∈ {0..6}`, `ambig_state ∈ {3}` (`dataset.radar.filters`) |
| Sweeps | `info_builder` / `RadarLoader.load` | 5 radars × key-frame + previous sweeps (5 on the GTX 1650, 6 cached) |
| Coordinate transformation | `RadarLoader.load_sweep` | positions with `sensor2ref`; velocity vectors with its rotation only |
| Velocity processing | `RadarLoader.load_sweep`, `RadarEncoder.raster` | compensated velocity in the reference frame = object motion over ground |
| Point features | `RADAR_POINT_FEATURES` | x, y, z, rcs, vx_comp, vy_comp, vx, vy, dyn_prop, time_lag → (Q, 10) |
| BEV representation | `DynamicPillarVFE` + `RadarEncoder.raster` | pillar features (64) + raster (density, vx, vy, RCS, time lag) |
| Radar features | `RadarEncoder.forward` | ConvBNReLU(69 → 59) + ResBlocks with dilation 1, 2, 4 → concatenate the 5 raster channels → `F_radar (B, 64, 200, 200)` |

## How velocity is represented

1. **Per point:** `vx_comp, vy_comp` (and raw `vx, vy`) are 2-D vectors in the reference frame, normalised by 10 m/s,
   inside the PointNet input.
2. **Per BEV cell, explicitly:** the mean compensated velocity of the cell (`radar_vx`, `radar_vy`, ÷10) is appended
   verbatim as two of the last five output channels. Velocity therefore reaches the feature selection, the fusion, the
   detection head's velocity branch and the planner without having to survive the learned layers.
3. **Downstream:** the detection head regresses object velocity `(vx, vy)`; the tracker and the planner use it for the
   motion forecast. Radar measurably helps velocity: adding radar to LiDAR lowered the velocity error (mAVE) from
   1.10 to 0.77 m/s on mini-val.

**Why the velocity rotation matters.** The devkit's `RadarPointCloud.from_file_multisweep` transforms radar positions
but leaves the velocity columns in each radar's own frame; with five radars facing different directions the vectors
would be inconsistent. The project rotates them explicitly.

**Sparsity.** After filtering and accumulating 5 sweeps of the 5 radars, a key-frame has ≈ 400–800 radar returns
(observed on mini) versus ≈ 110 000 LiDAR points. The convolution stack (one 3×3 conv plus residual blocks with dilation
1, 2 and 4) has a receptive field of ≈ 31 cells (≈ 15 m), so each isolated return influences its neighbourhood.
"""),
]
