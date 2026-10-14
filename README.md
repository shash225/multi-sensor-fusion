# Multi-Sensor Fusion with Learned Feature Selection — nuScenes

Camera → BEV · LiDAR + Radar encoders · **learned feature selection** · deep sensor fusion ·
3D object detection · tracking · trajectory planning · interactive scene replay for demos.

A research prototype that tests one hypothesis:

> *Learned feature selection before LiDAR–Radar fusion improves multi-sensor perception and
> trajectory planning by selecting informative spatial, semantic and motion features while
> suppressing redundant or noisy sensor information.*

Everything runs on a **4 GB GTX 1650** with **nuScenes-mini** and scales to the full dataset and
bigger GPUs through config profiles. The pipeline has been executed end-to-end on the real
nuScenes-mini data; the numbers in this README are measured, not estimated.

---

## 1. Architecture at a glance

```
 Camera (6) ──► LSS camera BEV (ResNet18 + camera-aware depth) ─────────────┐ F_cam (B,64,200,200)
 LiDAR      ──► dynamic PointPillars + height/intensity raster ──► F_lidar  │
 Radar (5)  ──► velocity-preserving radar pillars + Doppler raster ► F_radar │ guide
                                │                                            ▼
                                └──► Camera-guided cross-modal FEATURE SELECTION
                                     channel gates c_m + spatial gates s_m:  F̃_m = F_m ⊙ c_m ⊙ s_m
                                                       │
                       F_cam, F̃_lidar, F̃_radar ──► attention-gated FUSION ──► fused BEV (B,128,200,200)
                                                       │
                                            SECOND-FPN (B,256,100,100)
                                ┌──────────────────────┴───────────────────────┐
                        CenterPoint detection head                     learned planning head
                     boxes + velocities ─► tracker ─► classical planner (A* + speed profile + iLQR MPC)
```

Full tensor-flow table, per-module shapes/losses/compute and the reasoning behind every model
choice: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Component | Choice | Params | GTX 1650 fp32 |
|---|---|---|---|
| Camera → BEV | Lift-Splat-Shoot + BEVDepth camera-aware DepthNet, LiDAR depth supervision | 16.4 M | 26 ms |
| LiDAR encoder | dynamic PointPillars (scatter-max) + raster channels | 0.17 M | 19 ms |
| Radar encoder | dynamic pillars + dilated convs + explicit velocity channels | 0.23 M | 16 ms |
| Feature selection | camera-guided cross-modal SE (channels) + CBAM (space), soft or hard-concrete masks | 0.11 M | 5 ms |
| Fusion | attention-gated (per-cell sensor softmax + CBAM); `concat` baseline via registry | 0.51 M | 19 ms |
| BEV backbone + detection | SECOND-FPN + CenterPoint head (class, box, yaw, velocity) | 4.4 M | 15 ms |
| Planning | classical A* + iLQR MPC baseline; learned ego-query/GRU head | 0.78 M | 4 ms (+30 ms classical) |
| **Total (proposed model F)** | | **22.6 M** | **≈100 ms ≈ 10 FPS** |

## 2. What has been validated (real nuScenes-mini)

| Check | Result | Script |
|---|---|---|
| LiDAR points inside GT boxes vs `num_lidar_pts` (calibration + ego pose + box conventions) | Pearson r = **0.9997**, 86 % exact, 97 % within 10 % (3 322 boxes) | `scripts/validate_sensors.py` |
| Camera projection ↔ LSS lift round trip (with augmentation) | max error **6·10⁻¹⁴ m** | same |
| Camera / radar key-frame time offset vs LiDAR | ≤ 45 ms / ≤ 37 ms (compensated through per-sensor ego poses) | same |
| In-memory nuScenes metrics vs official devkit | NDS, mAP, mATE…mAAE agree within **3·10⁻⁴** | `scripts/check_eval_vs_devkit.py` |
| All 9 fusion configurations: forward, losses, backward | finite, ≤ 3.2 GB training peak | `scripts/smoke_test.py --experiments all` |
| Collision checker on the logged human trajectory | 0.0 % collisions (sanity) | `scripts/eval_planner.py --planner logged` |
| Unit / integration tests (geometry, data, models, metrics, planner, dashboard) | `python main.py test` | `tests/` |

Classical planner with ground-truth perception on mini-val (69 frames with a full 3 s future):

| Planner | ADE | FDE (= L2@3 s) | L2@1 s | Collision rate | Time |
|---|---|---|---|---|---|
| Constant velocity | 1.77 m | 4.03 m | 0.54 m | 14.5 % | – |
| **A\* + speed profile + iLQR (ours)** | **1.65 m** | **3.48 m** | 0.59 m | **1.4 %** | 30 ms |

## 3. Project structure

```
multi_sensor_fusion/
├── configs/            dataset / model / fusion / planning / training / ablation YAML, hardware/ profiles, local.yaml
├── datasets/           nuscenes_db (tables), info_builder (calibration, GT, ego futures), camera/lidar/radar loaders,
│                       nuscenes_dataset (torch Dataset + collate), map_prior, splits
├── models/
│   ├── camera_bev/     image_backbone, lss (LSS + DepthNet)
│   ├── lidar_encoder/  pillar_encoder          ├── radar_encoder/ radar_encoder
│   ├── feature_selection/ selection, losses    ├── fusion/ fusion_methods, registry, bev_backbone
│   ├── detection/      center_head, targets, decoder
│   ├── trajectory/     planning_head, losses   ├── common/ blocks, pillars
│   └── fusion_model.py MultiSensorFusionNet (wiring + total loss)
├── planning/           occupancy_grid, astar, mpc (iLQR), tracking, trajectory_planner
├── training/           trainer (stage-aware), cli, train_bev / lidar / radar / fusion / detection / trajectory / end_to_end / joint
├── evaluation/         detection_metrics (nuScenes NDS port), trajectory_metrics, evaluator, official_eval, ablation
├── visualization/      common, camera_view, bev_view, lidar_view, radar_view, feature_selection, trajectory_view
├── inference/          pipeline (frame-by-frame, used by app + scripts)
├── app/                streamlit_app.py + ui/ (styles, panels, resources)
├── scripts/            download_dataset, preprocess, validate_sensors, smoke_test, evaluate, inference, benchmark,
│                       eval_planner, check_eval_vs_devkit
├── tests/              pytest suite          ├── docs/  ARCHITECTURE, EXPERIMENTS, FACULTY_DEMO
├── checkpoints/        <experiment>/<stage>/{best,last}.pt (config embedded)
├── requirements.txt    environment.yml       main.py (single CLI)
```

## 4. Installation

Python 3.10/3.11 and an NVIDIA driver supporting CUDA 12.x.

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
```
```bash
pip install -r requirements.txt
```
```bash
pip install nuscenes-devkit --no-deps
```

The devkit is optional (official evaluation + full train/val scene lists). It is installed
without dependencies because it pins a matplotlib version that has no wheel for Python 3.11.
Conda users: `conda env create -f environment.yml`.

## 5. Dataset

Expected layout (`dataset.root`):
```
<root>/samples/  sweeps/  maps/  v1.0-mini/  [v1.0-trainval/]  [can_bus/ optional]
```
Point the project at it **without editing code** — any of:
* `configs/local.yaml` → `dataset: {root: "D:/datasets/nuscenes"}` (copy `configs/local.example.yaml`)
* environment variable `NUSCENES_ROOT`
* `--set dataset.root=/data/nuscenes` on any command

```bash
python main.py download          # verifies the folder; --download-mini fetches v1.0-mini after you accept the terms
```
```bash
python main.py preprocess --maps # builds the per-sample info caches (+ 0.5 m/px map priors)
```

Full dataset: `--set dataset.version=v1.0-trainval dataset.train_split=train dataset.val_split=val`.

## 6. Quick start (≈5 minutes, no training needed)

```bash
python main.py validate          # sync / calibration report + images in outputs/validation/
```
```bash
python main.py smoke             # tensor shapes, losses, GPU memory, per-stage latency
```
```bash
python app/streamlit_app.py      # dashboard: modes 1 and 2 work before any training
```

## 7. Training

Seven-stage curriculum (`configs/training.yaml → stages`), each stage with its own trainable
modules, losses, initialisation and epochs:

| Stage | Command | Trains | Losses |
|---|---|---|---|
| 1 camera BEV | `python training/train_bev.py` | camera-only model | detection + depth |
| 2 LiDAR | `python training/train_lidar.py` | LiDAR-only model | detection |
| 3 radar | `python training/train_radar.py` | radar-only model | detection |
| 4 fusion | `python training/train_fusion.py` | feature selection + fusion + heads (encoders frozen, loaded from 1–3) | detection + FS + fusion |
| 5 detection | `python training/train_detection.py` | + LiDAR/radar encoders | detection + FS + fusion |
| 6 trajectory | `python training/train_trajectory.py` | planning head only | trajectory |
| 7 end-to-end | `python training/train_end_to_end.py` | everything except the image backbone | all |

```bash
python main.py train --stage curriculum --experiment F_cam_lidar_radar_fs     # stages 1-7 in order
```
```bash
python main.py train --stage joint --experiment F_cam_lidar_radar_fs --epochs 20   # single-stage (used for the ablation)
```

Useful overrides: `--set train.batch_size=2 train.grad_accum=2 train.amp=true train.grad_checkpoint=true`.
Hardware profiles (`configs/hardware/`): `gtx1650` (default in `local.yaml`), `rtx3060`, `rtx4060`,
`rtx4070`, `jetson_orin_nano` (inference only) — select with `--set runtime.hardware_profile=rtx3060`.

Checkpoints: `checkpoints/<experiment>/<stage>/{best,last}.pt` (atomic writes, full config embedded,
resumable with `--resume`). Logs: `outputs/logs/<experiment>/<stage>/{run.log,history.jsonl}`.

## 8. Evaluation

```bash
python main.py evaluate --checkpoint checkpoints/F_cam_lidar_radar_fs/joint/best.pt --official
```
* Detection: mAP, mATE, mASE, mAOE, mAVE, mAAE, NDS (in-memory port, cross-checked with the devkit;
  `--official` also runs `DetectionEval`).
* Planning: ADE, FDE, L2@1/2/3 s, collision rate (ego footprint vs real future agent boxes), lateral /
  longitudinal error, jerk, curvature change — for the learned head and for the classical planner fed
  with the model's detections.
* Feature selection: selected-feature ratio, mean LiDAR / radar importance.
* Latency per stage (CUDA-synchronised).

## 9. Ablation study (the research experiment)

| Exp | Sensors | Fusion | Feature selection |
|---|---|---|---|
| A | LiDAR | – | – |
| B | Radar | – | – |
| C | LiDAR + Radar | concatenation | – |
| D | LiDAR + Radar | attention-gated | – |
| E | LiDAR + Radar | attention-gated | ✓ |
| F | Camera BEV + LiDAR + Radar | attention-gated | ✓ (camera-guided) |
| G / H / I | Cam+LiDAR, Cam+Radar, Cam+LiDAR+Radar w/o FS | attention-gated | ✓ / ✓ / – |

```bash
python main.py ablation --train --epochs 20     # trains missing checkpoints, evaluates, plots
```
Outputs in `outputs/ablation/`: `results.md` (table + hypothesis deltas: E−C, E−D, F−I, F−E, C−A),
`results.csv/json`, `ablation_metrics.png`. Evaluation uses the final-epoch weights by default
(`--which best` = best validation epoch). Protocol and full analysis: **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

**Results on nuScenes-mini** (20 epochs, final weights, 1 seed, 81 val frames — small and noisy):

| | A LiDAR | B Radar | C L+R concat | D L+R attention | E L+R + FS | F Cam+L+R + FS |
|---|---|---|---|---|---|---|
| NDS | 0.060 | 0.053 | **0.116** | 0.070 | 0.055 | 0.094 |
| mAP | 0.032 | 0.007 | 0.035 | 0.021 | 0.027 | **0.044** |
| velocity error (m/s) | 1.10 | 1.48 | **0.77** | 1.42 | 1.03 | 0.91 |
| planning ADE (m) | 1.37 | **1.26** | 1.36 | 1.28 | 1.51 | 1.53 |

* Radar improves LiDAR (C vs A: +0.056 NDS, −0.33 m/s velocity error) and the camera BEV improves
  LiDAR+radar (F vs E: +0.039 NDS, +0.017 mAP).
* **Learned feature selection did not beat simple concatenation on mini** (E vs C: −0.061 NDS). The
  hypothesis is not supported at this data scale; docs/EXPERIMENTS.md lists likely causes and the
  follow-up runs (no-regulariser variant, full trainval, 3 seeds) needed to test it properly.
* All learned planners beat constant velocity (ADE 1.26–1.53 m vs 1.77 m).

## 10. Inference and replay video

```bash
python main.py infer --scene scene-0103 --video
```
Writes 1920×1080 composite frames (camera with boxes · sensor-fusion tiles · driving BEV · trajectory),
an MP4 (2 fps = real time), a summary JSON and detections in the nuScenes global frame to
`outputs/inference/<scene>_<mode>/`. Without a checkpoint it runs in ground-truth mode and says so.

## 11. Dashboard and faculty demo

```bash
python app/streamlit_app.py
```
* **Modes**: 1 Dataset Viewer · 2 Ground Truth Simulation · 3 AI Autonomous Driving.
* **Scene replay**: Load Scene, ⏮ Previous, ▶ Play, ⏸ Pause, ⏭ Next, 🔄 Reset, speed 0.25×–5×.
* **Panels**: camera with predicted (+GT) 3D boxes; camera/LiDAR/radar BEV, LiDAR and radar importance,
  fused BEV; autonomous-driving BEV (map, points, radar velocities, boxes, tracks, planned vs logged
  trajectory); detection table; trajectory panel (ego X/Y/yaw/velocity, ADE, FDE, collision).
* **🎓 Faculty Demo**: live pipeline strip (each stage ticks with its latency), simplified captions.
* **🔍 Explain Feature Selection**: raw → importance → selected for LiDAR and radar, sensor-dominance map,
  fusion attention, named channel gates (e.g. radar velocity), per-class sensor trust — all from the
  network's actual gates.
* **⚖️ Compare fusion methods** on the same frame or whole scene; **📊 Ablation**; **📈 Training curves**.
* **No fake outputs**: without a checkpoint mode 3 shows *GROUND TRUTH / DATASET VISUALIZATION*;
  untrained weights are only shown behind an explicit checkbox with a red *UNTRAINED* banner.

Step-by-step presentation script: **[docs/FACULTY_DEMO.md](docs/FACULTY_DEMO.md)**.

## 12. Hardware notes

| GPU | Profile | Notes |
|---|---|---|
| GTX 1650 4 GB | `gtx1650` | 128×352 images, 5 sweeps, batch 1 × accum 4, grad checkpointing, **fp32**. Training peak 3.2 GB; inference 0.77 GB / ~100 ms. ≈2 min per mini epoch. |
| RTX 3060 12 GB / 4070 12 GB | `rtx3060` / `rtx4070` | 256×704 images, 10 LiDAR sweeps, batch 4, AMP. Full trainval becomes feasible (days, not weeks). |
| RTX 4060 8 GB | `rtx4060` | 256×704, batch 2 × accum 2, AMP. |
| Jetson Orin Nano | `jetson_orin_nano` | dashboard / inference only. |

**fp16 on the GTX 1650**: with cuDNN 9.10 this GPU returns NaN for fp16 3×3 convolutions with 256
input channels (measured; `utils/device.py` documents the probe). AMP is therefore disabled in the
`gtx1650` profile and is automatically turned off on any GPU that fails the start-up self-test. fp32 is
also 3.4× faster than fp16 on this card (no tensor cores).

Full nuScenes training is not realistic on a GTX 1650 (≈28 k samples → ~3 h per epoch at this size,
dozens of epochs). nuScenes-mini is the validation target here; use an RTX-class GPU for trainval.

## 13. Limitations (read before presenting numbers)

* nuScenes-mini has 323 training and 81 validation samples (2 val scenes). Absolute mAP/NDS are far
  below full-dataset papers and noisy; three classes (construction vehicle, trailer, barrier) have no GT
  in mini-val, which caps official mAP — `mAP_present_classes` is reported as well.
* The navigation command (left/right/straight) is derived from the logged future, as in ST-P3/UniAD
  open-loop evaluation; there is no route planner.
* Planning is evaluated open-loop against the logged trajectory (no closed-loop simulator).
* The map prior is the nuScenes semantic raster (0.1 m → cached at 0.5 m); the vector map expansion is not used.
* Attributes (moving/parked…) come from a velocity heuristic, not a learned head.

## 14. Command cheat-sheet

| Task | Command |
|---|---|
| Validate sensors | `python main.py validate` |
| Smoke test all configs | `python main.py smoke --experiments all --quiet` |
| Train proposed model | `python main.py train --stage joint --experiment F_cam_lidar_radar_fs --epochs 20` |
| Full curriculum | `python main.py train --stage curriculum` |
| Evaluate | `python main.py evaluate --checkpoint checkpoints/F_cam_lidar_radar_fs/joint/best.pt` |
| Ablation | `python main.py ablation --train --epochs 20` |
| Planner with GT perception | `python main.py planner --planner baseline` |
| Benchmark | `python main.py benchmark` |
| Replay video | `python main.py infer --scene scene-0916 --video` |
| Dashboard | `python app/streamlit_app.py` |
| Tests | `python main.py test` |

## Adverse-weather testing (fog / rain)

```bash
python main.py weather --weather all      # SYNTHETIC fog + rain on all sensors, models A-F, weights read-only
python main.py weather --source real      # real nuScenes rain/night scenes (needs full nuScenes; leakage-guarded)
```
Results, method, datasets and limitations: [docs/ADVERSE_WEATHER.md](docs/ADVERSE_WEATHER.md); outputs in `outputs/adverse_weather/`.

## Opening the dashboard from other devices

```powershell
.\scripts\start_public_demo.ps1
```
Same network: `http://<this-PC-IP>:8501` (allow Python in the Windows firewall prompt).
Anywhere: the script prints a temporary `https://<random>.trycloudflare.com` link (Cloudflare quick tunnel; the PC must keep running).
