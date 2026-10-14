# Adverse-weather testing (fog and rain)

Evaluation-only extension: the trained models A–F are tested, **unchanged**, under fog and rain.
Nothing here trains or fine-tunes a model, and normal training / evaluation / the dashboard are unaffected
(the weather code is only active when `cfg.weather.synthetic` is set by the weather runner).

## 1. Existing architecture (what is being tested)

| Part | Implementation | File |
|---|---|---|
| Camera | 6 surround cameras → Lift-Splat-Shoot BEV, BEVDepth-style depth net (ResNet-18) | `models/camera_bev/lss.py` |
| LiDAR | 32-beam LIDAR_TOP, multi-sweep, dynamic PointPillars + raster channels | `models/lidar_encoder/` |
| Radar | 5 radars, velocity-preserving pillars + dilated context | `models/radar_encoder/` |
| Sync / calibration | every sensor → ego(t_sensor) → global → ego(t_ref) | `datasets/info_builder.py` |
| Fusion type | **intermediate (feature-level) BEV fusion**; concat (A–C) or attention-gated (D–F) | `models/fusion/fusion_methods.py` |
| Reliability mechanisms | per-cell softmax **sensor trust** (attention fusion) and **feature-selection gates** (E, F). No Kalman filter, no explicit confidence estimation | `models/feature_selection/selection.py` |
| Detection | CenterPoint heatmap head, 10 classes, 3-D boxes with velocity | `models/detection/` |
| Tracking | greedy centre matching with velocity prediction (no Kalman filter) | `planning/tracking.py` |
| Evaluation | nuScenes NDS/mAP port (devkit-verified), planning metrics, latency | `evaluation/` |

Trained models: A LiDAR, B radar, C LiDAR+radar concat, D attention, E attention + feature selection,
F camera+LiDAR+radar + feature selection (proposed). All trained on nuScenes-mini `mini_train`
(8 scenes), evaluated on `mini_val` (scene-0103, scene-0916). There is **no camera-only checkpoint**,
so camera-only robustness cannot be measured without training a new model (not done: it would be a new experiment).

## 2. Datasets considered

| Dataset | Weather | Sensors | Annotations | Licence / access | Usable here? |
|---|---|---|---|---|---|
| **nuScenes (full v1.0-trainval)** | rain (≈41 val scenes, from descriptions), night; **no fog** | 6 cams, LiDAR, 5 radars — same rig as training | 3-D boxes, same format | CC BY-NC-SA 4.0, nuScenes account required | **Yes — genuine multimodal real-rain test**, implemented (`--source real`); needs the download |
| nuScenes-mini (local) | none (only "Night, after rain" = wet road, and it is a *training* scene) | same | same | same | Real-weather test **not possible** without leakage |
| Seeing Through Fog / DENSE | clear, light fog, dense fog, snow/rain (frame tags) | stereo camera, HDL-64 LiDAR, single front radar, gated, FIR | 2-D + 3-D boxes | custom academic licence, registration | Not compatible without retraining: different rig (1 front camera vs 6, different LiDAR, different radar fields) |
| ACDC | fog, rain, night, snow | single front camera | panoptic segmentation | registration, non-commercial | No: image-only 2-D labels; the project has no 2-D image detector |
| DAWN | fog, rain, snow, sand | single images from the web | 2-D vehicle boxes | CC BY 4.0 | No: no calibration, no LiDAR/radar, 2-D only |

Using DENSE/ACDC/DAWN would mean adding a new detector or retraining on a new sensor rig — that
would not test *this* fusion system, so it was not done. No LiDAR, radar or calibration was fabricated.

## 3. What was implemented

**Real weather (`--source real`)** — scenes of the evaluation split are tagged from their *original*
nuScenes description (`rain`, `fog`, `night`, `wet_road`, `clear`; keyword rules in
`configs/weather.yaml`). The description text is stored with every tag; severity is recorded as
"not provided" (nuScenes has none). Scenes in the checkpoint's training split are excluded
(leakage guard) and listed in the manifest. Evaluation runs the unchanged pipeline on the remaining
scenes, grouped by condition, with the clear scenes of the same split as baseline.

**Synthetic weather (`--source synthetic`)** — clear `mini_val` frames corrupted consistently on all sensors.
Every table, chart and image is labelled SYNTHETIC.

| Sensor | Fog (visibility V) | Rain (rate R) |
|---|---|---|
| Camera | Koschmieder `I = J·t + A(1−t)`, `t = e^(−βd)`, `β = 3.912/V`; per-pixel distance `d` from densified key-frame LiDAR, else ground plane, else sky (1 km) | same model with optical rain extinction `α = 1.45·R^0.64` dB/km (LISA) + rain streaks, wet-optics blur, contrast loss (visual approximation) |
| LiDAR | mini-LISA rule `ρ·e^(−2βr)/r² ≥ 0.9/R_max²` (points dropped, intensities attenuated) + near-range backscatter clutter (count not calibrated) | same rule with `α = 1.45·R^0.64` dB/km |
| Radar | unchanged (fog attenuation at 77 GHz ≪ 1 dB over 100 m) | ITU-R P.838-3 at 77 GHz: `γ = 1.0733·R^0.7104` dB/km, two-way loss subtracted from RCS; points below the frame's weakest RCS removed |

Conditions (configs/weather.yaml): clear; light fog V = 200 m; dense fog V = 50 m;
light / moderate / heavy rain 2.5 / 7.5 / 25 mm/h (AMS rain-rate classes).

**Integrity**: weights are loaded read-only and their SHA-256 is checked before/after every
experiment; evaluation uses the same final-epoch checkpoints (`last.pt`) as the ablation; the
synthetic test frames are the validation scenes (never trained on); corruption is deterministic
(seed 1234 + sample index); each run is saved separately and errors are reported, not skipped.

## 4. Metrics

* **Precision / recall** — detections with score ≥ 0.3, matched greedily (highest score first) to
  ground truth of the same class within **2 m centre distance**. FN = missed objects, FP = incorrect detections.
* **nuScenes mAP / NDS** — official centre-distance protocol (0.5/1/2/4 m). *mAP50 (2-D IoU ≥ 0.5) is not
  defined for this 3-D BEV detector and is therefore not reported.*
* **Centre error** — mean distance of matched detections (localisation accuracy).
* **FPS** — 1000 / median network latency on the GPU (data loading and weather simulation excluded).
* **Relative change** — `(condition − clear) / clear` for the same model.
* **Sensor statistics** — camera transmission and RMS-contrast ratio, LiDAR points kept and backscatter
  points, radar points kept and mean rain loss.
* **Reliability mechanisms** — mean attention weight per sensor (D–F) and feature-selection keep ratios (E, F).
* Tracking consistency is **not** measured: tracking runs only in the dashboard replay and has no
  ground-truth track-ID evaluation in this project.

## 5. Commands

```bash
python main.py weather --weather all                 # synthetic: clear + 2 fog + 3 rain, models A-F
python main.py weather --weather fog                 # clear + light/dense fog
python main.py weather --weather rain                # clear + light/moderate/heavy rain
python main.py weather --weather dense_fog --experiments F_cam_lidar_radar_fs
python main.py weather --source real                 # real nuScenes weather (needs full nuScenes for rain)
python main.py weather --report-only                 # rebuild reports from saved runs
python scripts/test_adverse_weather.py --weather fog # same entry point
```

Full nuScenes for the real-rain test (account at nuscenes.org; accept the licence there):
download `v1.0-trainval_meta` and the sensor blobs that contain the val scenes into the dataset root, then

```bash
python main.py weather --source real --set dataset.version=v1.0-trainval --split val
```

Outputs: `outputs/adverse_weather/<synthetic|real>/` — `results.md/.csv/.json`, `plots/`, `samples/`,
`runs/` (one JSON per model × condition), `manifest.json` (parameters, data, checkpoint hashes, environment).

## 6. Files

Created: `configs/weather.yaml`, `datasets/weather.py`, `evaluation/adverse_weather.py`,
`visualization/weather_view.py`, `scripts/test_adverse_weather.py`, `tests/test_weather.py`, this file.
Modified (additive only): `datasets/nuscenes_dataset.py` (optional weather hook),
`evaluation/detection_metrics.py` (`precision_recall`, `PR@2m` key), `evaluation/evaluator.py`
(records attention trust and weather statistics), `main.py` (`weather` command).

## 7. Results (SYNTHETIC weather, nuScenes-mini val: 81 frames, 2 scenes; final-epoch weights)

Run on 2026-10-09, GTX 1650, `python main.py weather --weather all` — 36/36 evaluations succeeded, all six
checkpoints byte-identical before and after (SHA-256). Full tables: `outputs/adverse_weather/synthetic/results.md`.

**Proposed model F (camera + LiDAR + radar + feature selection)**

| Weather condition | Data / scenes | Precision | Recall | nuScenes mAP | NDS | FPS | Relative NDS change |
|---|---|---|---|---|---|---|---|
| Clear | real clear frames, scene-0103 + scene-0916 | 0.642 | 0.287 | 0.0440 | 0.0943 | 9.3 | baseline |
| Light fog (V 200 m) | synthetic, same frames | 0.552 | 0.237 | 0.0342 | 0.0872 | 9.4 | −7.6 % |
| Dense fog (V 50 m) | synthetic, same frames | 0.272 | 0.127 | 0.0065 | 0.0417 | 9.3 | −55.8 % |
| Light rain (2.5 mm/h) | synthetic, same frames | 0.670 | 0.236 | 0.0382 | 0.0901 | 9.1 | −4.4 % |
| Moderate rain (7.5 mm/h) | synthetic, same frames | 0.723 | 0.186 | 0.0324 | 0.0821 | 9.6 | −13.0 % |
| Heavy rain (25 mm/h) | synthetic, same frames | 0.595 | 0.122 | 0.0233 | 0.0844 | 9.6 | −10.5 % |

(mAP50 is not defined for this 3-D detector; nuScenes centre-distance mAP is reported.)

**NDS change vs clear, all models**

| Model | light fog | dense fog | light rain | moderate rain | heavy rain |
|---|---|---|---|---|---|
| A LiDAR only | −3.0 % | −28.9 % | −0.0 % | −0.1 % | −0.3 % |
| B Radar only | −0.0 % | −0.0 % | +0.7 % | +0.6 % | +0.5 % |
| C LiDAR + Radar (concat) | −28.6 % | −56.7 % | −4.4 % | −4.6 % | −4.6 % |
| D LiDAR + Radar (attention) | −22.8 % | −64.8 % | −22.2 % | −22.1 % | −21.2 % |
| E LiDAR + Radar + feature selection | −5.1 % | −32.7 % | −3.8 % | −3.7 % | −3.5 % |
| F Camera + LiDAR + Radar + FS | −7.6 % | −55.8 % | −4.4 % | −13.0 % | −10.5 % |

### What the data says

1. **Dense fog is by far the worst condition** for every model that uses LiDAR or the camera (−29 % to −65 % NDS);
   the simulated LiDAR loses 21 % of its points and its range shrinks to ≈ 25 m, camera transmission drops to 0.33.
2. **Radar is unaffected** (−0 % in fog, +0.5 % in rain), as expected physically (77 GHz attenuation ≪ 1 dB at these ranges).
   But radar alone is the weakest detector (NDS 0.053), so robustness does not mean good performance.
3. **Fusion is not automatically more robust.** In absolute terms the fused models stay best or equal in fog
   (F recall 0.127 vs LiDAR-only 0.083 in dense fog), but they lose a *larger share* of their clear-weather performance
   (C −57 %, D −65 % vs A −29 % in dense fog).
4. **The camera is F's weak point in rain**: F's recall falls 57 % in heavy rain (0.287 → 0.122), below LiDAR-only (0.192),
   while LiDAR-only loses 7 %. The rain mainly affects the image (streaks, blur, contrast 0.74).
5. **Reliability mechanisms**: the attention fusion of F gives the camera *more* weight in dense fog (0.369 → 0.398)
   and heavy rain (→ 0.397), i.e. it does **not** down-weight the degraded sensor. The feature-selection gates react
   to LiDAR point loss (LiDAR features kept 0.335 → 0.212 in dense fog) and slightly raise radar in rain (0.106 → 0.123).
   Feature-selection model E degrades least among the fused models (−33 % dense fog, −4 % rain).
6. **Attention-only fusion D is fragile**: −22 % already in light rain, although LiDAR and radar change by only ~2 % / 0.6 %.
   The cause was not isolated (candidates: loss of weak far LiDAR returns, RCS shift).
7. Latency is unchanged by weather (the network does the same work); the weather simulation runs in the data loader.

### Limitations

* Synthetic, not real weather: simplified physics (no rain splash/spray, wet-road reflections, droplets on the
  lens, sensor-specific noise); LiDAR backscatter count and camera rain appearance are not calibrated.
* nuScenes-mini: 81 frames, one seed per model; absolute scores are low (see docs/EXPERIMENTS.md), so compare relatively.
* No real fog/rain test was possible locally (section 2). The real-rain path is implemented and needs full nuScenes.
* No camera-only model; tracking robustness not measured.

### Recommended next steps (optional, separate experiments — baseline untouched)

1. Real-rain evaluation on full nuScenes val (≈41 rain scenes): `python main.py weather --source real --set dataset.version=v1.0-trainval --split val`.
2. Weather-aware reliability: train with `dataset.augmentation.modality_dropout` and synthetic weather as augmentation
   (new experiment id, new checkpoint folder) so the attention learns to down-weight a degraded camera.
3. Isolate D's rain sensitivity (LiDAR-only vs radar-only corruption at a time).

## 8. Weather in the dashboard

Sidebar → **Weather**: Clear (real data) or one of the five SYNTHETIC conditions. The choice is applied per
request to the network input (camera, LiDAR, radar) *and* to the displayed camera images, in all three modes and in
"Compare fusion methods". Synthetic frames show a red banner and a "SYNTHETIC <condition>" stamp burned into the
camera images. The scene list also shows real conditions taken from the nuScenes descriptions (🌙 night,
💧 wet road). The expander "Adverse-weather study" shows the offline results (tables + charts).
