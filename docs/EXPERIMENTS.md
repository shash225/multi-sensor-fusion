# Research experiment: does learned feature selection help LiDAR–Radar fusion?

## Hypothesis

> Learned feature selection before LiDAR–Radar fusion can improve multi-sensor perception and
> trajectory planning by selecting informative spatial, semantic and motion features while
> suppressing redundant or noisy sensor information.

## Design

| Exp | Sensors | Fusion | Feature selection | Question answered |
|---|---|---|---|---|
| A | LiDAR | – | – | single-sensor reference |
| B | Radar | – | – | single-sensor reference (velocity-rich, geometry-poor) |
| C | LiDAR + Radar | concatenation | – | naive fusion baseline |
| D | LiDAR + Radar | attention-gated | – | is *attention alone* enough? |
| E | LiDAR + Radar | attention-gated | ✓ | **effect of feature selection** (E vs C, E vs D) |
| F | Camera BEV + LiDAR + Radar | attention-gated | ✓ camera-guided | proposed full system (F vs E: camera; F vs I: FS with camera) |
| G, H, I | Cam+LiDAR, Cam+Radar, Cam+LiDAR+Radar (no FS) | attention-gated | ✓, ✓, – | remaining sensor combinations |

**Controlled variables** — identical for all experiments: training stage `joint` (all losses, from
scratch, no curriculum advantage), 20 epochs, seed 42, AdamW lr 2e-4 cosine, batch 1 × grad-accum 4,
same data / augmentation (flip-y, scale), same BEV grid, BEV backbone and heads. Only the sensor set
and the fusion / feature-selection module change.

**Metrics** (mini-val, 81 key-frames, 2 scenes): NDS, mAP (official 10-class and present-classes),
mATE/mASE/mAOE/**mAVE** (velocity), mAAE; planning ADE / FDE / L2@3 s / collision rate (learned head)
and the classical planner fed with each model's detections; feature-selection ratios; latency.

**Primary comparisons**: E − C (selection vs concatenation), E − D (selection vs attention only),
F − I (selection with the camera present), F − E (value of the camera BEV), C − A (value of radar).

## Running

```bash
python main.py ablation --train --epochs 20
```
Writes `outputs/ablation/results.{md,csv,json}` and `outputs/ablation/ablation_metrics.png`.

## Statistical caution

Mini-val has 81 frames from 2 scenes and each configuration is trained once (one seed); the
seed-to-seed variance has not been measured here, so small differences between configurations may be
noise. Treat mini results as a functional check of the pipeline and a *trend*; confirm the hypothesis on full nuScenes
(`--set dataset.version=v1.0-trainval dataset.train_split=train dataset.val_split=val`) with ≥ 3 seeds
per configuration (`--set runtime.seed=...`).

## Results (GTX 1650, nuScenes-mini, 20 epochs, seed 42)

Protocol: **final-epoch weights** of every experiment (identical training budget, no model selection
on the evaluation split). Raw table: `outputs/ablation/results.md`; chart: `outputs/ablation/ablation_metrics.png`.

| Exp | Configuration | NDS ↑ | mAP ↑ | mAP present ↑ | mAVE m/s ↓ | ADE m ↓ | FDE m ↓ | Collision ↓ | Latency ms |
|---|---|---|---|---|---|---|---|---|---|
| A | LiDAR only | 0.060 | 0.032 | 0.046 | 1.10 | 1.37 | 2.86 | 8.7 % | 45 |
| B | Radar only | 0.053 | 0.007 | 0.010 | 1.48 | 1.26 | 2.85 | 11.6 % | 43 |
| C | LiDAR + Radar, concat | **0.116** | 0.035 | 0.050 | **0.77** | 1.36 | 3.15 | 7.2 % | 63 |
| D | LiDAR + Radar, attention | 0.070 | 0.021 | 0.030 | 1.42 | 1.28 | 2.74 | 7.2 % | 67 |
| E | LiDAR + Radar, feature selection | 0.055 | 0.027 | 0.039 | 1.03 | 1.51 | 3.51 | 7.2 % | 71 |
| F | Camera + LiDAR + Radar, feature selection | 0.094 | **0.044** | **0.063** | 0.91 | 1.53 | 3.47 | 13.0 % | 103 |
| – | constant-velocity planner (reference) | – | – | – | – | 1.77 | 4.03 | 14.5 % | – |

Planning columns are the learned planning head. Classical planner (A* + iLQR) fed with each model's
own detections: ADE 1.85–2.07 m, collision rate 4.3 % (A, F) to 15.9 % (D).

### What the data says

1. **The hypothesis is not supported at this scale.** Learned feature selection did *not* beat simple
   concatenation: E − C = −0.061 NDS, −0.008 mAP, +0.26 m/s velocity error. Against attention-only
   fusion the result is mixed (E − D: −0.015 NDS, but +0.007 mAP and −0.39 m/s velocity error).
2. **Radar clearly helps LiDAR.** C − A = +0.056 NDS and −0.33 m/s velocity error: the Doppler
   channels do what they should. This is the most consistent effect in the study.
3. **The camera BEV helps.** F − E = +0.039 NDS, +0.017 mAP; F has the best mAP of all configurations.
4. **Every learned planner beats constant velocity** (ADE 1.26–1.53 m vs 1.77 m, collisions 7–13 % vs
   14.5 %). Differences between fusion variants in planning are small and not consistent.
5. **What the selection network learned** (final epoch, measured on 21 val frames):
   * about 34 % of LiDAR features and 11 % of radar features pass the gates in F (E: 39 % / 10 %);
   * LiDAR spatial gate ≈ 0.99 on cells with LiDAR returns vs 0.42 (F) / 0.55 (E) elsewhere;
   * radar velocity (vx, vy), RCS, density and time-lag channels are all kept (gates 0.96–0.99);
   * with the camera present (F) the LiDAR *mean-height* channel is switched off (gate 0.001) while
     max-height is kept (0.98); without the camera (E) it stays on (0.89);
   * in both E and F the radar spatial gate is *lower on the cells containing radar returns*
     (0.14–0.18) than around them (0.35–0.38): the network relies on the dilated radar context rather than
     on the exact return positions — consistent with radar's poor angular accuracy, but this
     interpretation is a hypothesis, not something the study tested.

The best-validation-epoch protocol (`--which best`, archived as `results_best_epoch.*`) gives the same
picture: C has the best NDS (0.120), F the best mAP (0.043), E − C = −0.036 NDS, C − A = +0.045 NDS.

### Possible reasons (not yet tested)

* **Data scale**: 323 training samples; the gates plus the sparsity loss (target: keep 35 % of cells)
  may discard information a model this data-starved still needs. E's validation NDS peaked at epoch 6
  and then declined, which points to unstable / over-regularised training.
* **Noise**: 81 validation frames from 2 scenes, one seed.

### Next experiments that would actually test the hypothesis

```bash
python main.py train --stage joint --experiment E_lidar_radar_fs --set loss.lambda_feature_selection=0.0
```
(feature selection without the sparsity/coverage losses — separates the gating from the regulariser), then
the full study on nuScenes trainval with three seeds per configuration on an RTX-class GPU:
```bash
python main.py ablation --train --epochs 20 --set dataset.version=v1.0-trainval dataset.train_split=train dataset.val_split=val runtime.hardware_profile=rtx3060
```
