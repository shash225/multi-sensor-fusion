"""Narrative chapters, part B (research component, heads, planning, training, evaluation, app, reference, operations)."""

CHAPTERS_B = [
# =====================================================================================================
("Deep-Learning Feature Selection", r"""
This chapter documents the project's research component: **Camera-Guided Cross-Modal Feature Selection (CG-CMFS)**,
class `CrossModalFeatureSelection` in `models/feature_selection/selection.py`, trained with
`models/feature_selection/losses.py`.

## Problem it addresses

A naive fusion concatenates every LiDAR and every radar channel at every BEV cell. Part of this information is
*redundant* (LiDAR and radar both encode "something is here") and part is *noisy* (radar multipath ghosts, LiDAR returns
from vegetation, far-range sparsity). The hypothesis of the project is that a network which **learns to select**
informative spatial, semantic and motion features of each sensor before fusion yields better perception and planning.

## Where it sits

`F_cam`, `F_lidar`, `F_radar` (each `(B, 64, 200, 200)`) come out of the encoders. The selection module receives
`F_lidar` and `F_radar` (the *selectable* modalities) and, if the camera is enabled and `fusion.selection.camera_guided`
is true, `F_cam` as a **guide**. It outputs selected maps `F̃_lidar`, `F̃_radar` of the same shapes plus the gates. The
fusion module then receives `{camera: F_cam, lidar: F̃_lidar, radar: F̃_radar}` (`MultiSensorFusionNet.forward`).

![Figure 7 – Structure of the feature-selection module with the default sizes of model F.](docs/handover/build/feature_selection.png)

## Mathematical definition

For each selectable sensor `m ∈ {lidar, radar}` the module predicts a **channel gate** `c_m ∈ (0,1)^C` and a **spatial
gate** `s_m ∈ (0,1)^{H×W}`; the selected features are

$$\tilde{F}_m = F_m \odot c_m \odot s_m$$

(`c_m` broadcast over space, `s_m` over channels).

**Channel gates (cross-modal squeeze-and-excitation).** With `μ(·)` and `max(·)` denoting global average and max pooling
over H × W of every input (LiDAR, radar and the camera guide):

$$[\ell^{c}_{lidar},\,\ell^{c}_{radar}] = \mathrm{MLP}\left([\mu(F_l), \max(F_l), \mu(F_r), \max(F_r), \mu(F_c), \max(F_c)]\right),\qquad c_m = g(\ell^{c}_m)$$

The MLP is `Linear(384 → 96) → ReLU → Linear(96 → 128)` for model F. Because its input contains statistics of all
sensors, the gates of one sensor depend on the content of the others (e.g. radar velocity channels can be kept when
LiDAR is sparse).

**Spatial gates (CBAM-style, camera-guided).** Each input is compressed by a 1×1 ConvBNReLU to 8 channels and
complemented by its channel-mean and channel-max maps (10 channels per input, 30 for model F):

$$[\ell^{s}_{lidar},\,\ell^{s}_{radar}] = \mathrm{Conv}_{3\times3}\left(\mathrm{CBR}_{3\times3}\left(\mathrm{CBR}_{7\times7}\left([\phi_m(F_m),\,\bar{F}_m,\,\hat{F}_m]_{m}\right)\right)\right),\qquad s_m = g(\ell^{s}_m)$$

**Gate function `g` (`_gate`, `fusion.selection.gate_mode`).**
- `soft` (default, used for all reported experiments): `g(ℓ) = σ(ℓ)`.
- `hard_concrete`: in training a relaxed Bernoulli sample with straight-through gradient, at evaluation `ℓ > 0`:

$$g = \sigma\left(\frac{\ell + \log u - \log(1-u)}{\tau}\right),\quad u \sim U(0, 1),\qquad g_{hard} = [g > 0.5],\qquad \mathrm{output} = g_{hard} + g - \mathrm{sg}(g)$$

**Initialisation.** The final layers of both branches have weights `N(0, 0.01)` and bias `init_open = 2.0`, so all gates
start at σ(2) ≈ 0.88: training starts from "keep almost everything" and learns what to suppress.

## Training objective

`feature_selection_loss` (weights in `loss.feature_selection`, targets in `fusion.selection.target_keep_ratio`):

$$L_{cov} = \frac{\sum_{cells} y\,\left(-\log \max_m s_m\right)}{\sum_{cells} y}$$

$$L_{sp} = \sum_m \left|\,\overline{s_m} - \rho_s\right| + \left|\,\overline{c_m} - \rho_c\right|,\qquad \rho_s = 0.35,\ \rho_c = 0.75$$

$$L_{bin} = \overline{4\,g\,(1 - g)}\quad\mathrm{(over\ all\ gates)}$$

$$L_{FS} = 1.0\,L_{cov} + 0.5\,L_{sp} + 0.1\,L_{bin},\qquad \mathrm{weighted\ by}\ \lambda_3 = 0.5\ \mathrm{in\ the\ total\ loss}$$

`y` is the class-agnostic object-centre heatmap (`fg_heatmap`). The coverage term only penalises *missing* selection at
objects (at least one sensor must keep its features there); the sparsity term limits how much is kept overall; the
binarisation term pushes gates towards crisp 0/1 decisions. The detection, trajectory and fusion losses also flow back
through the gates.

## Statistics and explainability

`selection_stats()` reports per sensor the fraction of spatial cells / channels with gate > 0.5, mean importances and the
*effective selected ratio* (fraction of cells × fraction of channels). The dashboard's **Explain Feature Selection**
section and `scripts/render_bev.py` visualise, from the actual network outputs:

| View | Function | Meaning |
|---|---|---|
| Raw → importance → selected | `visualization/feature_selection.py: energy_map, importance_map` | feature energy before selection, spatial gate, energy after selection |
| Sensor contribution | `contribution`, `dominance_map` | where LiDAR (blue) or radar (orange) information actually passes the selection |
| Fusion attention | `modality_weight_map` | per-cell softmax weights of the fusion module (camera green, LiDAR blue, radar orange) |
| Named channel gates | `channel_gate_table` | gates of the explicit physical channels (height, intensity, velocity, RCS, …) |
| Sensor trust per class | `object_importance`, `class_importance_summary` | mean gate of each sensor inside detected boxes, per class |

![Figure 8 – All BEV representations of a held-out frame (scene-0916) from model F, including the LiDAR and radar importance maps.](outputs/bev/scene-0916_frame020_bev.jpg)

## What the trained network learned (measured)

Measured on 21 validation frames with the final (epoch 20) weights of model F (camera-guided) and model E (no camera):

| Quantity | F (camera + LiDAR + radar) | E (LiDAR + radar) |
|---|---|---|
| Effective selected ratio LiDAR / radar | 0.34 / 0.11 | 0.39 / 0.10 |
| LiDAR spatial gate: cells with LiDAR returns / elsewhere | 0.99 / 0.42 | 1.00 / 0.55 |
| Radar spatial gate: cells with radar returns / elsewhere | 0.18 / 0.35 | 0.14 / 0.38 |
| LiDAR gates density / max height / **mean height** / intensity | 0.99 / 0.98 / **0.001** / 0.99 | 0.99 / 0.93 / 0.89 / 0.98 |
| Radar gates density / vx / vy / RCS / time lag | 0.99 / 0.98 / 0.99 / 0.98 / 0.98 | 0.96 / 0.96 / 0.97 / 0.97 / 0.98 |
| Learned channels kept (gate > 0.5) LiDAR / radar | 77 % / 73 % | 77 % / 75 % |

Interpretation (descriptive, not tested causally): LiDAR is kept wherever it has data; radar *velocity* and *RCS*
channels are kept; with the camera present the network drops the redundant LiDAR *mean height* channel while keeping
*max height*; the radar gate is lower exactly on the cells that contain radar returns than in the surrounding context,
i.e. the network relies on the spread radar context rather than on the exact (angularly inaccurate) return positions.

## Effect on performance

| Comparison (final weights, mini-val) | ΔNDS | ΔmAP | ΔmAVE (m/s, lower = better) | ΔADE (m) |
|---|---|---|---|---|
| E (feature selection) − C (concatenation) | −0.061 | −0.008 | +0.255 | +0.151 |
| E (feature selection) − D (attention only) | −0.015 | +0.007 | −0.391 | +0.237 |
| F (with camera) − E | +0.039 | +0.017 | −0.113 | +0.015 |

**On nuScenes-mini the hypothesis is not supported**: the selection did not beat plain concatenation. With 323 training
frames, a single seed and 81 validation frames the comparison is underpowered; E's validation NDS peaked at epoch 6 and
then declined, which suggests the extra gating and its regularisers make training on tiny data less stable. Testing the
hypothesis properly requires the full dataset and several seeds (see *Known Limitations & Next Steps*).

## Configuration knobs

| Key | Default | Effect |
|---|---|---|
| `fusion.feature_selection` | true (experiment dependent) | enable the module |
| `fusion.selection.camera_guided` | true | use the camera BEV as guide when the camera is enabled |
| `fusion.selection.gate_mode` | soft | `soft` or `hard_concrete` |
| `fusion.selection.temperature` | 0.67 | hard-concrete temperature τ |
| `fusion.selection.init_open` | 2.0 | initial gate logit (σ(2) ≈ 0.88) |
| `fusion.selection.channel_reduction` | 4 | MLP bottleneck factor |
| `fusion.selection.spatial_hidden`, `spatial_kernel` | 32, 7 | spatial branch width and first kernel |
| `fusion.selection.threshold` | 0.5 | gate value counted as "selected" in statistics |
| `fusion.selection.target_keep_ratio.spatial / channel` | 0.35 / 0.75 | sparsity targets ρ_s, ρ_c |
| `loss.lambda_feature_selection` | 0.5 | λ3 of L_FS in the total loss |
| `loss.feature_selection.coverage / sparsity / binarization` | 1.0 / 0.5 / 0.1 | weights inside L_FS |
"""),
# =====================================================================================================
("Sensor Fusion & BEV Backbone", r"""
## Fusion methods

`models/fusion/registry.py` maps `fusion.method` to a class; both classes take `{modality: (B, C_m, H, W)}` and return
`(fused (B, 128, H, W), info)`.

**`concat`** — `ConcatFusion`: concatenation of all maps → 3×3 ConvBNReLU → ResBlock. Used by experiments A, B, C.

**`attention_gated`** — `AttentionGatedFusion` (experiments D–I). With per-sensor projections `P_m = φ_m(F_m)`
(1×1 ConvBNReLU to 128 channels):

$$a = \mathrm{softmax}_m\left(g([F_1, \ldots, F_M])\right),\qquad F_w = \sum_m a_m\,P_m,\qquad F_{fused} = \mathrm{CBAM}\left(\mathrm{Res}\left(\mathrm{CBR}_{1\times1}([F_w, P_1, \ldots, P_M])\right)\right)$$

`g` is a 3×3 ConvBNReLU (→ 64) followed by a 1×1 conv (→ M). `a` is a per-cell probability over sensors ("which sensor
to trust where"); keeping the individual projections in the mixer input preserves complementary information. The
weights are returned (`info['modality_weights']`) and shown in the dashboard.

## Deep supervision (L_fusion)

`MultiSensorFusionNet` adds a 1×1 conv ("aux head", bias −2.19) on every (selected) sensor map and on the fused map.
During training each predicts the class-agnostic centre heatmap `fg_heatmap`; the Gaussian focal losses are averaged:

$$L_{fusion} = \frac{1}{M+1}\left(\sum_m L_{focal}(\mathrm{aux}_m(\tilde{F}_m), y) + L_{focal}(\mathrm{aux}_{fused}(F_{fused}), y)\right)$$

This keeps every branch discriminative on its own. Weight `loss.lambda_fusion = 0.25`; disable with
`fusion.deep_supervision: false`.

## BEV backbone

`SECONDFPN` (`model.bev_backbone`): block 1 = stride-2 conv + 3 convs (128 ch, 100 × 100), block 2 = stride-2 conv +
5 convs (256 ch, 50 × 50); deblock 1 = 1×1 conv, deblock 2 = transposed conv ×2; concatenation → `(B, 256, 100, 100)`.
Gradient checkpointing per block when `train.grad_checkpoint` is true.

## Adding a fusion method

```python
from models.fusion.registry import register_fusion

@register_fusion("my_fusion")
class MyFusion(nn.Module):
    def __init__(self, in_channels: dict[str, int], out_channels: int): ...
    def forward(self, feats: dict[str, torch.Tensor]):
        return fused, {}            # fused: (B, out_channels, H, W)
```
The module must be imported before the model is built (e.g. add the import in `models/fusion/registry.py`); then set
`fusion.method: my_fusion` and add an experiment in `configs/ablation.yaml`.
"""),
# =====================================================================================================
("3D Object Detection", r"""
## Classes

The ten nuScenes detection classes in index order (`classes` in `configs/dataset.yaml`): car, truck,
construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone. nuScenes categories are mapped
with the official table `CATEGORY_TO_DETECTION` (`datasets/info_builder.py`); other categories (animals, debris,
emergency vehicles, bicycle racks, …) are not detection targets.

## Targets (`models/detection/targets.py`)

On the head grid (100 × 100, 1 m): a Gaussian per object in its class heatmap (radius from the CornerNet formula with
`gaussian_overlap = 0.1`, at least `min_radius = 2` cells) and, at the centre cell, the regression target
`[dx, dy, z, log l, log w, log h, sin ψ, cos ψ, vx, vy]`. GT boxes with no LiDAR/radar return or outside the grid are
dropped (`dataset.gt_filter.min_points = 1`). Velocities that nuScenes leaves undefined are masked.

## Head and loss (`models/detection/center_head.py`)

Shared 3×3 conv (256 → 64) and six branches. Classification uses the CornerNet focal loss on probabilities `p` with
targets `y` (α = 2, γ = 4), normalised by the number of centre cells:

$$L_{cls} = -\frac{1}{N_{pos}}\left[\sum_{y=1}(1-p)^{\alpha}\log p + \sum_{y<1}(1-y)^{\gamma}\,p^{\alpha}\log(1-p)\right]$$

Regression is an L1 loss at the centre cells weighted by `code_weights` (1 for offset, z, size, sin, cos; 0.2 for
velocity) and split into centre, box, orientation and velocity terms:

$$L_{det} = 1.0\,L_{cls} + 0.25\,L_{center} + 0.25\,L_{bbox} + 0.25\,L_{orient} + 0.25\,L_{vel}$$

## Decoding (`models/detection/decoder.py`)

Sigmoid → 3×3 max-pool peak extraction → top-300 → metric boxes (`x = (i_x + dx)·1 m + x_min`, sizes `exp(·)` with the
log-size clamped at 4, `ψ = atan2(sin, cos)`) → score threshold (`detection_head.score_threshold = 0.1`; evaluation uses
`eval.score_threshold = 0.1`, the dashboard shows ≥ 0.3 by default) → class-wise circle NMS with radii
`detection_head.nms_radius` (car 1 m, pedestrian 0.25 m, bus 2 m, …).

## Attributes

The network does not predict attributes; `evaluation/detection_metrics.py: default_attribute` derives them from the
predicted speed (> 0.2 m/s → moving). This affects only mAAE.
"""),
# =====================================================================================================
("Trajectory Planning & Tracking", r"""
## Inputs

- **Ego state** `[speed, yaw rate, acceleration]`: from the CAN-bus expansion if present, otherwise estimated *causally*
  from the 20 Hz LiDAR ego poses (`info_builder._ego_state_from_poses`). The CAN-bus folder is not present in the current
  dataset, so the pose estimate is used.
- **Navigation command** RIGHT / LEFT / STRAIGHT from the lateral offset (±2 m) of the logged future's last point — the
  standard open-loop protocol (ST-P3, UniAD); it replaces a route planner.
- **Detections** (model mode) or GT boxes (ground-truth mode), the LiDAR key-frame points, optional map prior.

![Figure 9 – Planners as implemented.](docs/handover/build/planner.png)

## Baseline planner (`planning/trajectory_planner.py: BaselinePlanner`)

1. **Scene model** (`build_scene`): static obstacles (detections slower than 1 m/s, score ≥ 0.3) inflated by half the
   ego width + 0.3 m → blocked cells; dynamic occupancy `(6, 120, 140)` from all detections propagated with their velocity;
   soft costs from obstacle clearance `exp(−d/1.5 m)` and non-ground LiDAR returns.
2. **Goal** (`_goal`): lookahead `clip(1.2 · 3 s · max(v, 2), 6, 45)` m, bearing 0° / ±35° by command.
3. **A\*** (`planning/astar.py`) on the cost grid (8-connected, cost `step·(1 + c)`), then smoothing and 0.25 m resampling.
4. **Speed profile**: nine constant accelerations (−6 … +1.5 m/s²); each is scored by progress error, comfort (a²) and
   +1000 per step where the ego footprint hits the dynamic occupancy.
5. **iLQR MPC** (`planning/mpc.py`) on the kinematic bicycle model (state `[x, y, ψ, v, δ]`, control `[a, ω]`,
   wheelbase 2.588 m, dt 0.1 s, 30 steps, 12 iterations):

$$x_{t+1} = x_t + v_t\cos\psi_t\,\Delta t,\quad y_{t+1} = y_t + v_t\sin\psi_t\,\Delta t,\quad \psi_{t+1} = \psi_t + \frac{v_t}{L}\tan\delta_t\,\Delta t$$

$$v_{t+1} = v_t + a_t\,\Delta t,\qquad \delta_{t+1} = \delta_t + \omega_t\,\Delta t,\qquad J = \sum_t e_t^{T} Q\, e_t + u_t^{T} R\, u_t$$

6. **Output**: states at 0.5, 1.0, …, 3.0 s → `PlanResult.waypoints (6, 2)` plus a collision flag against the
   *predicted* occupancy.

## Learned and hybrid planners

`LearnedPlanner` uses the `PlanningHead` waypoints (residuals on a constant-velocity prior, trained with L1 to the logged
trajectory + collision + smoothness + goal losses). `HybridPlanner` uses the learned trajectory unless it collides with
the predicted occupancy, in which case the baseline is used. Select with `planning.planner` or the dashboard dropdown.

## Tracking (`planning/tracking.py`)

Greedy association of detections (score ≥ 0.3) to tracks in the **global** frame, predicted with constant velocity,
class-specific gates (car 4 m, pedestrian 1 m, …), track removal after two missed frames, velocity smoothing. Provides
the IDs and motion trails in the BEV view.

## Measured planning performance (mini-val, 69 frames with a full 3 s future)

| Planner and perception | ADE (m) | FDE (m) | Collision rate |
|---|---|---|---|
| Constant velocity | 1.77 | 4.03 | 14.5 % |
| Baseline planner with GT objects | 1.65 | 3.48 | 1.4 % |
| Logged human trajectory (sanity check of the metric) | 0.00 | 0.00 | 0.0 % |
| Learned planning heads (ablation models, final weights) | 1.26 – 1.53 | 2.74 – 3.51 | 7.2 – 13.0 % |
| Baseline planner with each model's detections | 1.80 – 2.07 | – | 2.9 – 15.9 % |

The learned planners imitate the logged trajectory more closely (lower ADE) than the classical planner but collide more
often; the classical planner is safer when perception is good. Collisions are measured against the **real** future
positions of other agents (`evaluation/trajectory_metrics.py: collision_flags`).
"""),
# =====================================================================================================
("Loss Functions", r"""
`MultiSensorFusionNet.compute_losses()` builds the total loss from the loss groups a training stage activates:

$$L_{total} = \lambda_1 L_{det} + \lambda_2 L_{traj} + \lambda_3 L_{FS} + \lambda_4 L_{fusion} + \lambda_5 L_{depth}$$

| Term | λ (key) | Default | Components (weights in `loss.*`) | Defined in |
|---|---|---|---|---|
| L_det | `lambda_detection` | 1.0 | heatmap 1.0, center 0.25, bbox 0.25, orientation 0.25, velocity 0.25 | `models/detection/center_head.py` |
| L_traj | `lambda_trajectory` | 1.0 | regression 1.0, collision 2.0, smoothness 0.1, goal 0.5 | `models/trajectory/losses.py` |
| L_FS | `lambda_feature_selection` | 0.5 | coverage 1.0, sparsity 0.5, binarization 0.1 | `models/feature_selection/losses.py` |
| L_fusion | `lambda_fusion` | 0.25 | mean of auxiliary heatmap focal losses | `models/detection/center_head.py: aux_heatmap_loss` |
| L_depth | `lambda_depth` | 1.0 | BCE to one-hot LiDAR depth bins | `models/camera_bev/lss.py: depth_loss` |

## Trajectory loss details

$$L_{traj} = \frac{\sum_t m_t\,\|\hat{p}_t - p_t\|_1}{\sum_t m_t} + 2.0\,\frac{\sum_t m_t\,\max_k O_t(f_{t,k})}{\sum_t m_t} + 0.1\,\overline{\|\Delta^2 \hat{p}\|^2} + 0.5\,\|\hat{p}_{T} - p_{T}\|_2$$

`m_t` masks unknown future steps, `O_t` is the softened GT future occupancy and `f_{t,k}` are five ego-footprint points
(bilinear `grid_sample`, differentiable), `Δ²` are second differences including the current velocity.

## Depth loss

$$L_{depth} = \frac{1}{|V|}\sum_{cells \in V}\ \sum_{d} \mathrm{BCE}\left(\alpha_d,\ [d = \lfloor (z - d_{min})/\Delta d \rfloor]\right)$$

over cells `V` with a LiDAR depth in [1, 60) m.

## Which stage uses which loss

{{AUTO:stage_losses}}
"""),
# =====================================================================================================
("Training System", r"""
## Training curriculum

{{AUTO:stages}}

Shared single-sensor stages (`shared: true`) are trained once and reused by all experiments
(`checkpoints/_shared/<stage>/`). `init_from` copies the parameters whose names start with the given prefixes from the
source stage's `best.pt` (falling back to `last.pt`). `frozen` / `trainable_only` control `requires_grad`; frozen
BatchNorm layers are kept in eval mode.

The **ablation** uses the single stage `joint` (all losses, from scratch) so that every experiment has an identical
recipe. The curriculum was verified mechanically (all seven stages run on 2 samples, weights are handed over and frozen as
specified) but has **not** been trained to convergence.

## Trainer internals (`training/trainer.py`)

| Feature | Implementation |
|---|---|
| Optimiser | AdamW, `lr = train.lr × stage.lr_scale` (2e-4), weight decay 0.01 |
| Schedule | LambdaLR: linear warm-up (`train.warmup_iters`, ≤ 10 % of steps) × cosine decay to 0 |
| Gradient accumulation | `train.grad_accum` (4 on the GTX 1650 → effective batch 4) |
| Mixed precision | `torch.autocast` + `GradScaler` if `train.amp` **and** the fp16 self-test passes |
| Gradient clipping | global norm `train.grad_clip` (35) after unscaling |
| Robustness | non-finite losses skip the batch (logged); missing sensor files are skipped with one warning |
| Gradient checkpointing | image backbone layers 2–4 and BEV backbone blocks (`train.grad_checkpoint`) |
| Data loading | `num_workers` (2), pinned memory, persistent workers, per-worker seeds |
| Validation | every `val_interval` epochs, `evaluate_model` on the validation split |
| Checkpoints | `best.pt` (by NDS / −ADE / −loss) and `last.pt`, atomic writes, config embedded, optimiser state included |
| Logs | `outputs/logs/<experiment>/<stage>/run.log` and `history.jsonl` |
| Resume | `--resume <checkpoint>` restores model, optimiser, scheduler, scaler, epoch, step, best score |
| Reproducibility | `runtime.seed` (42) for Python/NumPy/PyTorch; `runtime.deterministic` for cuDNN |

## Hardware and measured cost (GTX 1650, 4 GB, fp32)

| Item | Value |
|---|---|
| Proposed model F, batch 1, 128×352 images | training peak 3.2 GB, inference 0.5–0.8 GB, ≈ 100 ms per frame |
| LiDAR/radar-only models | inference 0.2–0.4 GB, 43–71 ms |
| Training speed | ≈ 2.6–2.8 iterations/s (F) |
| 20 epochs on nuScenes-mini | F 39 min; A 19 min; B 18 min; C 24 min; D 25 min; E 28 min |
| Checkpoint size | 65–70 MB (LiDAR/radar models), 258 MB (F) — includes AdamW moments |

**fp16 note.** On this GPU with cuDNN 9.10, fp16 3×3 convolutions with 256 input channels return NaN, and fp32 is also
3.4× faster here (no tensor cores). The `gtx1650` profile sets `train.amp: false`; `utils/device.py: fp16_is_reliable`
turns AMP off automatically on any GPU that fails the same probe.

## Training commands

```bash
python main.py train --stage joint --experiment F_cam_lidar_radar_fs --epochs 20
python main.py train --stage curriculum --experiment F_cam_lidar_radar_fs
python training/train_fusion.py --experiment E_lidar_radar_fs --set train.lr=1e-4
python main.py train --stage joint --experiment F_cam_lidar_radar_fs --resume checkpoints/F_cam_lidar_radar_fs/joint/last.pt
python main.py train --stage joint --set dataset.version=v1.0-trainval dataset.train_split=train dataset.val_split=val runtime.hardware_profile=rtx3060
```
"""),
# =====================================================================================================
("Evaluation & Experiments", r"""
## Detection metrics and their verification

`evaluation/detection_metrics.py` re-implements the official nuScenes detection evaluation (centre-distance matching at
0.5/1/2/4 m, 101-point interpolated precision, minimum recall/precision 0.1, TP errors at 2 m with confidence-aligned
cumulative means, class ranges, zero-point and bike-rack filtering, NaN rules). NDS:

$$NDS = \frac{1}{10}\left[5\,mAP + \sum_{k \in \{ATE, ASE, AOE, AVE, AAE\}} \left(1 - \min(1,\ mTP_k)\right)\right]$$

`scripts/check_eval_vs_devkit.py` evaluates the same (perturbed-GT) predictions with this port and with the official
`nuscenes.eval.detection.evaluate.DetectionEval` on mini-val:

| Metric | Port | Official devkit | Absolute difference |
|---|---|---|---|
| NDS | 0.3566 | 0.3566 | 0.0001 |
| mAP | 0.2982 | 0.2984 | 0.0002 |
| mATE | 0.6689 | 0.6692 | 0.0003 |
| mASE / mAOE / mAVE / mAAE | 0.4345 / 0.4329 / 0.7113 / 0.6776 | 0.4345 / 0.4329 / 0.7112 / 0.6776 | ≤ 0.0001 |

{{AUTO:official}}

**mini-val caveat.** Three classes (construction vehicle, trailer, barrier) have no GT in the two mini-val scenes; the
official rule gives them AP = 0, which caps mAP at 0.7. The project therefore also reports `mAP_present_classes`.

## Planning metrics

ADE, FDE, L2 at 1/2/3 s, collision rate (ego polygon vs. real future agent boxes, SAT test), lateral/longitudinal error,
jerk and curvature change (`evaluation/trajectory_metrics.py`). Only frames with a complete 3 s future are scored
(69 of 81 mini-val frames). The collision check is calibrated: the logged human trajectory scores 0 %.

## Ablation study

Design and hypothesis: `docs/EXPERIMENTS.md`. All experiments use stage `joint`, 20 epochs, seed 42, identical data and
augmentation; evaluation uses the **final** weights (`--which last`, default) to avoid model selection on the evaluation
split. Results generated from `outputs/ablation/results.json`:

{{AUTO:ablation}}

![Figure 10 – Ablation results (final weights, mini-val). Proposed model in blue.](outputs/ablation/ablation_metrics.png)

**Findings.** Radar improves LiDAR (C vs A: +0.056 NDS, −0.33 m/s velocity error); the camera BEV improves
LiDAR+radar (F vs E: +0.039 NDS, +0.017 mAP; F has the best mAP); learned feature selection does not beat concatenation
on mini (E vs C: −0.061 NDS). The best-validation-epoch protocol (archived as `results_best_epoch.*`) gives the same
picture. The evaluation is underpowered (81 frames, 2 scenes, one seed).
"""),
# =====================================================================================================
("Visualization & Streamlit Application", r"""
## Visualisation modules

| Module | Main functions | Output |
|---|---|---|
| `visualization/common.py` | `BEVCanvas`, `bev_to_display`, `heatmap_image`, `colorize`, `put_label`, `stack_images`, `CLASS_COLORS` | drawing primitives (OpenCV, RGB uint8) |
| `visualization/camera_view.py` | `project_points`, `draw_boxes_on_image`, `draw_lidar_on_image`, `camera_mosaic`, `camera_ipm_bev` | camera images with boxes / points, geometric camera BEV |
| `visualization/bev_view.py` | `SceneLayers`, `render_driving_scene` | the autonomous-driving BEV |
| `visualization/lidar_view.py`, `radar_view.py` | `render_lidar_bev`, `render_radar_bev` | raw-sensor BEV tiles |
| `visualization/feature_selection.py` | `energy_map`, `importance_map`, `contribution`, `dominance_map`, `modality_weight_map`, `object_importance`, `channel_gate_table` | explainability views |
| `visualization/trajectory_view.py` | `render_trajectory_closeup` | planned vs logged trajectory close-up |

Display convention: **forward up, left to the left**, ego in the lower-middle of the driving BEV
(x ∈ [−25, 55), y ∈ [−40, 40), 8 px/m).

## Dashboard structure

| Area | Content | Code |
|---|---|---|
| Sidebar | faculty toggle, version, device, mode, fusion method (✅ trained), planner, untrained opt-in, confidence threshold, speed, display options, scene | `app/streamlit_app.py` |
| Header | title, scene description, mode banner, warnings | – |
| Controls | Load Scene, Previous, Play, Pause, Next, Reset, key-frame slider | callbacks `cb_*` |
| Pipeline strip (faculty mode) | CAMERA → BEV … DRIVING OUTPUT with live ticks and times | `styles.flow_html`, `on_stage` |
| KPI strip | objects, FPS, NDS, mAP, ADE, collision, fusion method | `styles.kpis_html` |
| Panel 1 | camera image with predicted (+GT) 3D boxes; all six cameras | `panels.camera_panel` |
| Panel 3 | camera BEV, LiDAR BEV, radar BEV, LiDAR/radar importance, fused BEV | `panels.fusion_tiles` |
| Panel 2 | autonomous-driving BEV | `panels.driving_bev` |
| Panel 5 | ego pose, command, trajectory close-up, ADE/FDE/collision | `panels.trajectory_closeup`, `ego_status` |
| Panel 4 | detection table, GT comparison | `panels.detection_table`, `gt_comparison` |
| Explain Feature Selection | raw → importance → selected, contribution, attention, channel gates, class trust | `panels.explain_selection` |
| Compare / Ablation / Curves / About | method comparison, results, training curves, system story | – |

![Figure 11 – A frame of the exported replay video (model F, held-out scene-0103): camera with predicted boxes, sensor-fusion tiles, driving BEV, trajectory close-up and detection list.](outputs/inference/scene-0103_model/frame_020.jpg)

## Modes and honesty rules

| Mode | Shows | Requires |
|---|---|---|
| 1 · Dataset Viewer | raw sensors, GT boxes, logged trajectory | data only |
| 2 · Ground Truth Simulation | GT objects with tracks, logged trajectory, classical planner on GT (labelled) | data only |
| 3 · AI Autonomous Driving | network predictions, gates, plans, metrics vs GT | a checkpoint (else falls back to mode 2 with a warning) |

Untrained networks are shown only after ticking "Allow untrained network", with a red UNTRAINED banner. Scenes from the
training split are flagged.

## Launching

```bash
python app/streamlit_app.py
streamlit run app/streamlit_app.py
streamlit run app/streamlit_app.py -- --device cpu
```
The first command re-launches itself through Streamlit with the dark theme. `-- --device cpu` sets the default device
(useful while the GPU is busy training). Port 8501.
"""),
# =====================================================================================================
("Configuration Reference", r"""
Every key of the merged default configuration, generated from the YAML files (comments from the files are shown as
descriptions). Merge order: `dataset.yaml`, `model.yaml`, `fusion.yaml`, `planning.yaml`, `training.yaml` →
`hardware/<profile>.yaml` → experiment overrides (`ablation.yaml`) → `local.yaml` → `--set key=value`.

{{AUTO:config}}

## Hardware profiles

{{AUTO:hardware}}

## Experiments

{{AUTO:experiments}}
"""),
# =====================================================================================================
("Command-Line Reference", r"""
`python main.py <command> [arguments]` dispatches to the scripts below; every script can also be run directly. The
arguments are extracted from the `argparse` definitions in the code.

{{AUTO:cli}}
"""),
# =====================================================================================================
("Line-by-Line Code Walkthroughs", r"""
This chapter explains the important source files **block by block**. Each code listing is read from the file at
build time and shows the original line numbers; the explanation below it describes what the lines do, why they are
needed, what data and tensor shapes they operate on and how they connect to the next block. Trivial lines (imports,
simple assignments) are explained in groups; tensor transformations, geometry, attention, selection, fusion, losses,
decoding and planning logic are explained in detail. Files not covered here (small scripts, tests, styles) are fully
described in the *API Reference*. Emoji characters inside string literals (the dashboard's button labels) are printed
as `[name]` placeholders because the PDF fonts contain no emoji glyphs; the source files contain the emoji.

{{AUTO:walkthroughs}}
"""),
# =====================================================================================================
("API Reference", r"""
Generated from the source code for **every module, class, method and function** of the project. For each function:
signature with parameter and return types, purpose (docstring or hand-written note), internal calls to other project
functions, callers, raised exceptions and file side effects detected in the code. "–" means none found.

{{AUTO:api}}
"""),
# =====================================================================================================
("Testing", r"""
Run all tests with `python main.py test` (≈ 30 s on the development machine; tests that need nuScenes are skipped when
the dataset is not configured). Last run: **29 passed**.

{{AUTO:tests}}

Additional validation tools (not part of pytest because they produce reports and images):

| Tool | Verifies |
|---|---|
| `python main.py validate` | sensor time offsets, LiDAR-in-box agreement with `num_lidar_pts`, radar agreement, augmentation round trip, projection images |
| `python scripts/check_eval_vs_devkit.py` | detection metric port vs official devkit |
| `python main.py planner --planner logged` | collision metric calibration (expects 0 %) |
| `python main.py smoke --experiments all --quiet` | every model configuration: shapes, finite losses, memory, latency |
"""),
# =====================================================================================================
("Debugging & Troubleshooting Guide", r"""
| Symptom | Likely cause | Where to look / fix |
|---|---|---|
| `FileNotFoundError: nuScenes tables not found at ...` | wrong `dataset.root` / `dataset.version` | `configs/local.yaml`, `NUSCENES_ROOT`, `--set dataset.root=...` |
| Losses are NaN from the first iteration | fp16 on a GPU with the cuDNN fp16 bug | check the log for "fp16 convolutions are numerically broken"; set `train.amp=false` |
| `CUDA out of memory` | profile too large for the GPU, or another process on the GPU (e.g. the dashboard during training) | use the `gtx1650` profile, lower `dataset.camera.image_size`, `dataset.lidar.sweeps`, `train.batch_size`; enable `train.grad_checkpoint`; close other GPU programs |
| Validation NDS stays ≈ 0 | too few epochs, or wrong labels/frames | run `python main.py validate` (calibration), `python main.py smoke`; inspect `outputs/bev` images |
| Boxes drawn in the wrong place | frame/convention error in new code | compare with `utils/geometry.py` conventions; run `tests/test_geometry.py`, `tests/test_dataset.py` |
| Config change has no effect on data | stale info cache | caches are keyed by the relevant settings; force with `python main.py preprocess --rebuild` |
| `KeyError: Unknown experiment` / `Unknown stage` | typo | names in `configs/ablation.yaml` / `configs/training.yaml → stages` |
| Missing weights warning when training a later stage | source stage not trained | train the earlier stage or ignore (those weights start from scratch) |
| `size mismatch` when loading a checkpoint | model config differs from the checkpoint | load through the checkpoint's embedded config (as `scripts/evaluate.py` and the app do) |
| DataLoader hangs / slow start on Windows | worker processes are spawned | `--set dataset.num_workers=0` to debug |
| Dashboard mode 3 shows ground truth | no checkpoint for the selected method | train it; check `find_checkpoint` paths in `checkpoints/<experiment>/<stage>/` |
| Dashboard is slow | model running on CPU or GPU shared with training | select `cuda` in the sidebar when the GPU is free |
| `ImportError: nuscenes` | optional devkit missing | `pip install nuscenes-devkit --no-deps` (needed for official eval and trainval splits) |
| Map prior missing / slow first frame | map PNG decoded on first use (≈ 5 s, ≈ 1.5 GB RAM peak) | `python main.py preprocess --maps` |
| Radar/LiDAR file errors in the log | corrupt or missing sweep | warned once; the sample still runs (sensor contributes zeros) |

## Useful inspection points

- `out = model(batch)` returns all intermediates: `out['raw']`, `out['selected']`, `out['selection']`, `out['fused']`,
  `out['bev_features']`, `out['det']`, `out['plan']` — inspect shapes with `scripts/smoke_test.py`'s `describe()`.
- `FusionPipeline.run(index, 'model')` returns a `FrameResult` with maps, gates, timings and metrics for one frame.
- `python main.py bev --scene <scene> --frame <k>` renders every BEV of a frame into one image.
- Logs: `outputs/logs/<experiment>/<stage>/run.log` (iteration losses, LR, it/s, peak memory) and `history.jsonl`.
"""),
# =====================================================================================================
("Extending the Project", r"""
| Task | Steps |
|---|---|
| Add a fusion method | implement a class with the fusion contract, decorate with `@register_fusion("name")`, ensure it is imported, set `fusion.method`, add an experiment to `configs/ablation.yaml` (see *Sensor Fusion*) |
| Add an experiment | add an entry with `overrides` under `experiments` in `configs/ablation.yaml`; optionally add it to `dashboard_methods`; train with `python main.py ablation --train --experiments <name>` |
| Use full nuScenes | download trainval into `dataset.root`; `--set dataset.version=v1.0-trainval dataset.train_split=train dataset.val_split=val`; use an RTX profile; requires the devkit for the split lists |
| Change the BEV grid | edit `bev` in `configs/dataset.yaml`; keep `(x_max − x_min)/resolution` divisible by 4 (backbone strides); planning grid is separate |
| Change image size / backbone | `dataset.camera.image_size` (divisible by 16), `model.camera_bev.backbone` (resnet18/34/50) |
| Add a radar/LiDAR point feature | add the column in the loader, update the encoder's `in_features` and `normalizer`; for an explicit raster channel extend `raster()` and the `*_RASTER_NAMES` |
| Use binary feature masks | `--set fusion.selection.gate_mode=hard_concrete` |
| Test feature selection without its regulariser | `--set loss.lambda_feature_selection=0.0` |
| Replace the planner | implement `plan(ego_state, command, detections, **kwargs) -> PlanResult`, add it to `build_planner`, set `planning.planner` |
| Add a detection class | extend `classes`, `CATEGORY_TO_DETECTION`, evaluation `CLASS_RANGE`, `nms_radius`, colours; rebuild caches |
| Add a dashboard panel | write a renderer in `app/ui/panels.py` using `FrameResult`, display it in `app/streamlit_app.py` |
| Use the CAN bus | place the CAN-bus expansion in `<root>/can_bus/`; rebuild the info cache (the ego state then comes from CAN) |
"""),
# =====================================================================================================
("Artifacts: Checkpoints & Outputs", r"""
## Checkpoints

Read from the checkpoint files on disk (epoch, stage and stored validation metrics). Every checkpoint contains
`model`, `config` (full merged configuration), `optimizer`, `scheduler`, `scaler`, `epoch`, `step`, `metrics`, `extra`.

{{AUTO:checkpoints}}

## Generated outputs

{{AUTO:outputs}}
"""),
# =====================================================================================================
("Known Limitations, Open Issues & Next Steps", r"""
## Limitations

1. **Data scale.** All training and results use nuScenes-mini (323 training, 81 validation frames, 10 scenes). Absolute
   metrics are far below published full-nuScenes results and comparisons between configurations are noisy (one seed).
2. **Hypothesis not yet supported.** Learned feature selection did not improve over concatenation on mini.
3. **Curriculum not trained to convergence.** The 7-stage curriculum runs end-to-end but the reported models use the
   single `joint` stage.
4. **Open-loop planning.** Planning is evaluated against the logged trajectory; there is no closed-loop simulator. The
   navigation command is derived from the logged future.
5. **No attribute head** — attributes come from a speed heuristic (affects mAAE only).
6. **Map prior** — only the nuScenes semantic raster is used (optional planner cost and display); the vector map
   expansion is not used.
7. **GTX 1650 constraints** — small images (128 × 352), 5 sweeps, fp32 only.

## Recommended next steps

1. Run the ablation on nuScenes trainval with ≥ 3 seeds per configuration on an RTX-class GPU.
2. Isolate the effect of the gates from their regulariser: `--set loss.lambda_feature_selection=0.0`; try
   `gate_mode=hard_concrete` and other `target_keep_ratio` values.
3. Train the 7-stage curriculum and compare it with `joint`.
4. Add the remaining ablation configurations G, H, I (camera + LiDAR, camera + radar, camera + LiDAR + radar without
   selection) — already defined in `configs/ablation.yaml`.
5. Add temporal fusion (previous BEV frames) and an attribute head.
6. Closed-loop evaluation (e.g. in a simulator) for the planners.
"""),
# =====================================================================================================
("Appendix A – Glossary", r"""
| Term | Meaning |
|---|---|
| BEV | Bird's-Eye View: top-down 2-D grid around the vehicle |
| LSS | Lift-Splat-Shoot: camera → BEV by predicting a depth distribution per pixel |
| Frustum | set of 3-D points along each image ray at the depth-bin centres |
| Pillar | vertical column of a BEV cell containing points |
| VFE | voxel/pillar feature encoder (PointNet over the points of a pillar) |
| SE / CBAM | squeeze-and-excitation (channel attention) / convolutional block attention module |
| Gate | value in [0, 1] multiplying a feature (channel or cell) |
| bda | BEV data augmentation matrix (flip, rotation, scale) |
| Key-frame / sweep | annotated 2 Hz sample / intermediate unannotated sensor capture |
| Reference frame | ego frame at the LIDAR_TOP key-frame time |
| NDS, mAP, mATE, mASE, mAOE, mAVE, mAAE | nuScenes detection score, mean AP, mean translation/scale/orientation/velocity/attribute errors |
| ADE / FDE | average / final displacement error of a trajectory |
| iLQR / MPC | iterative linear-quadratic regulator / model-predictive control |
| A* | best-first graph search with an admissible heuristic |
| Infos | cached per-sample dictionaries produced by `datasets/info_builder.py` |
"""),
# =====================================================================================================
("Appendix B – Not Identifiable from the Current Codebase", r"""
The following information was requested or is commonly part of a handover but cannot be determined from the code or the
files of the project. Each item is marked **Not identifiable from the current codebase.**

| Item | Status |
|---|---|
| Licence of the project code | Not identifiable from the current codebase. (No licence file inside `multi_sensor_fusion/`; the nuScenes data itself is CC BY-NC-SA 4.0 under the nuScenes terms.) |
| Version-control history / authorship per file | Not identifiable from the current codebase. (The project folder is not a git repository.) |
| Results on full nuScenes (trainval) | Not identifiable from the current codebase. (Supported by the code, never run.) |
| Performance on RTX 3060/4060/4070 and Jetson Orin Nano | Not identifiable from the current codebase. (Profiles exist; not measured.) |
| Results of the 7-stage curriculum | Not identifiable from the current codebase. (Implemented and mechanically tested, not trained to convergence.) |
| Results of experiments G, H, I | Not identifiable from the current codebase. (Defined, not trained.) |
| CAN-bus based ego state | Not identifiable from the current codebase. (Reader implemented; the CAN-bus expansion is not present in the dataset folder.) |
| Use of the folders `D:\researcpaper\project`, `D:\researcpaper\dataset`, `D:\researcpaper\nuScenes-panoptic-v1.0-all` | Not identifiable from the current codebase. (Not referenced by any code.) |
| Seed-to-seed variance of the results | Not identifiable from the current codebase. (Single seed.) |
"""),
# =====================================================================================================
("Appendix C – Software Environment", r"""
Versions measured in the project's virtual environment on the development machine:

{{AUTO:env}}
"""),
]
