"""Block-by-block walkthroughs of the neural-network modules (models/*).

Each block = (anchor, explanation). The code shown in the PDF starts at the first line that contains
the anchor (searched in order) and runs to the next block's anchor, so every line of the file is shown.
"""

WALKTHROUGHS = [
    {
        "file": "models/common/blocks.py",
        "intro": "Small reusable building blocks used by every BEV network in the project. Nothing here is "
                 "sensor-specific; the blocks operate on 4-D feature maps `(B, C, H, W)`.",
        "blocks": [
            ('"""Reusable', "Module docstring and imports. `torch.nn` provides the layer classes; `from __future__ import "
             "annotations` lets the type hints use modern syntax on Python 3.10/3.11."),
            ("class ConvBNReLU", "`ConvBNReLU` is a `nn.Sequential` of Conv2d → BatchNorm2d → (optional) ReLU.\n\n"
             "- `padding=dilation * (kernel // 2)` keeps the spatial size unchanged for odd kernels at stride 1 "
             "(also for dilated kernels), so `(B, in_ch, H, W) → (B, out_ch, H/stride, W/stride)`.\n"
             "- `bias=False` because the following BatchNorm has its own learnable shift; a conv bias would be redundant.\n"
             "- `relu=False` is used as the last layer of a residual branch (the ReLU is applied after the addition)."),
            ("class ResBlock", "Basic residual block: two 3×3 `ConvBNReLU` (the second without ReLU), then "
             "`ReLU(x + branch(x))`. Input and output shapes are identical `(B, C, H, W)`. The `dilation` argument "
             "enlarges the receptive field without downsampling — used by the radar encoder (d = 1, 2, 4) and the "
             "depth network (d = 2)."),
            ("class SqueezeExcitation", "Channel attention (Squeeze-and-Excitation). `weights()` pools every channel "
             "globally with both mean and max (`x.mean((2, 3))`, `x.amax((2, 3))` → `(B, 2C)`), passes the result "
             "through a bottleneck MLP `2C → max(C/reduction, 8) → C` and a sigmoid, giving one weight in (0, 1) per "
             "channel `(B, C)`. `forward()` multiplies the feature map by these weights broadcast over H and W. "
             "Used inside `CBAM`."),
            ("class SpatialAttention", "CBAM spatial attention. The channel-wise mean and max maps (`(B, 1, H, W)` "
             "each) are concatenated and convolved with a 7×7 kernel to a single map; the sigmoid gives a per-pixel "
             "weight `(B, 1, H, W)` which `forward()` multiplies onto every channel."),
            ("class CBAM", "Convolutional Block Attention Module = channel attention followed by spatial attention. "
             "Used as the final refinement of `AttentionGatedFusion` (models/fusion/fusion_methods.py). Shape preserving."),
        ],
    },
    {
        "file": "models/common/pillars.py",
        "intro": "Pure-PyTorch *dynamic pillar* encoding shared by the LiDAR and radar encoders. It turns an "
                 "unordered point list into a dense BEV canvas without custom CUDA kernels (important on Windows). "
                 "Point clouds arrive as one tensor for the whole batch with a leading batch-index column "
                 "(see `collate_batch`).",
        "blocks": [
            ('"""Dynamic pillar', "Docstring: explains *dynamic voxelisation* — every point is kept (no maximum "
             "number of points per pillar, unlike the original PointPillars) and features are aggregated with a "
             "scatter-max. Input `points (P, 1+F)`, output `(B, C, ny, nx)`."),
            ("def scatter_mean", "`scatter_mean(src (P, F), index (P,), size U) → (U, F)`: per-group mean.\n\n"
             "1. `index_add_` sums the rows of `src` into `out[index]` (one row per pillar).\n"
             "2. A second `index_add_` of ones counts the points per pillar.\n"
             "3. Division by `count.clamp(min=1)` avoids division by zero for (impossible) empty groups.\n\n"
             "Used for the per-pillar mean xyz (cluster offsets) and for the raster statistics."),
            ("def scatter_max", "`scatter_max(src (P, C), index (P,), size U) → (U, C)`: per-group maximum using "
             "`Tensor.scatter_reduce_(..., reduce='amax', include_self=True)`. The output is initialised with zeros, "
             "which is correct only because `src` is post-ReLU (non-negative) — documented in the docstring. This "
             "is the PointNet max-pooling inside each pillar."),
            ("class PillarIndexer", "`PillarIndexer` maps points to pillars on a `BEVGrid`.\n\n"
             "- Column 0 is the batch index `b`; columns 1 and 2 are x and y in metres (reference frame).\n"
             "- `ix = floor((x − x_min)/r)`, `iy = floor((y − y_min)/r)` give the cell indices (r = 0.5 m → 200 × 200).\n"
             "- `keep` drops points outside the grid or with an invalid batch index.\n"
             "- `flat = (b·ny + iy)·nx + ix` gives one integer id per (sample, cell).\n"
             "- `torch.unique(flat, return_inverse=True)` returns the U occupied pillars (`unique`) and, for every "
             "point, the index of its pillar (`inverse`, shape `(P',)`). This inverse index drives all scatter "
             "operations.\n\nReturns the filtered points and the index tensors in a dict."),
            ("class DynamicPillarVFE", "`DynamicPillarVFE` (Voxel/Pillar Feature Encoder). Constructor builds a "
             "small PointNet: for `channels = (32, 64)` the layers are `Linear(F+5 → 16)` and `Linear(32 → 64)`, "
             "each followed by BatchNorm1d and ReLU. Non-final layers output `c_out // 2` features because the "
             "pillar-max feature (another `c_out // 2`) is concatenated afterwards (PointPillars PFN design), so the "
             "next layer again sees `c_out` inputs. `in_features + 5` = raw point features + 3 offsets to the pillar "
             "mean + 2 offsets to the pillar centre."),
            ("def forward(self, points", "Forward pass, step by step:\n\n"
             "1. `canvas` `(B, C, ny, nx)` of zeros is created first, so empty inputs (e.g. a missing radar) return "
             "a valid all-zero feature map instead of crashing.\n"
             "2. `self.indexer` assigns pillars. If no point is inside the grid — or, during training, fewer than "
             "two points exist (BatchNorm1d cannot normalise a single sample) — the empty canvas is returned.\n"
             "3. `feats = pts[:, 1:]` drops the batch column; `xyz` are the first three feature columns.\n"
             "4. `mean = scatter_mean(xyz, inv, U)[inv]` broadcasts each pillar's mean back to its points.\n"
             "5. `centre_x/centre_y` are the metric centres of each point's cell.\n"
             "6. `raw = feats / normalizer` rescales raw columns (e.g. x/50 m, intensity/255) so inputs are O(1).\n"
             "7. The decorated point feature is `[raw, xyz − mean, x − centre_x, y − centre_y]` → `(P', F+5)`.\n"
             "8. For each PFN layer: Linear-BN-ReLU per point, then `scatter_max` per pillar → `pooled (U, C_i)`; "
             "for non-final layers the pooled feature is concatenated back to every point.\n"
             "9. The final `pooled (U, C)` is written into the canvas at `[b, :, iy, ix]` (recovered from `flat` by "
             "integer division / modulo). `.to(canvas.dtype)` avoids an fp16/fp32 mismatch under autocast.\n\n"
             "Output: `(B, C, ny, nx)`; cells without points stay zero."),
            ("def bev_raster_stats", "`bev_raster_stats` builds *hand-crafted* BEV channels: `log(1 + count) / 3` "
             "(point density) plus the per-cell mean of selected point columns (`value_cols`, 0-based feature "
             "indices; `+1` skips the batch column). Output `(B, 1 + len(value_cols), ny, nx)`. These explicit, "
             "physically meaningful channels (density, height, intensity, radar velocity, RCS, time lag) are appended "
             "to the learned encoder outputs so that the feature-selection channel gates can be interpreted "
             "per physical quantity."),
        ],
    },
    {
        "file": "models/camera_bev/image_backbone.py",
        "intro": "The image feature extractor of the camera branch: a torchvision ResNet trunk with a light "
                 "two-level FPN neck that outputs one feature map at stride 16.",
        "blocks": [
            ('"""Image backbone', "Imports. `torchvision` provides the ResNet definitions and ImageNet weights; "
             "`torch.utils.checkpoint.checkpoint` implements gradient checkpointing; `_WEIGHTS` maps the backbone "
             "name in the config (`model.camera_bev.backbone`) to the torchvision weights enum."),
            ("class ImageBackbone", "Constructor.\n\n"
             "- Loads `resnet18/34/50` with ImageNet weights when `pretrained=True`. The `try/except` keeps the "
             "program running offline: on any download error a warning is logged and the network is randomly "
             "initialised.\n"
             "- `stem` = conv1+bn1+relu+maxpool (stride 4); `layer1..layer4` have strides 4, 8, 16, 32.\n"
             "- `c3`/`c4` read the channel counts of layer3/layer4 from the network (256/512 for ResNet18/34, "
             "1024/2048 for ResNet50).\n"
             "- `neck`: 1×1 ConvBNReLU on `concat(c3, upsample(c4))` → `out_channels` (256), then a 3×3 ConvBNReLU.\n"
             "- `frozen_stages` / `norm_eval` / `grad_checkpoint` are stored and `_freeze()` is applied."),
            ("def _freeze", "`_freeze()` puts the first `frozen_stages + 1` stages (stem, layer1 for the default "
             "`frozen_stages = 1`) into eval mode and disables their gradients. Low-level ImageNet filters are "
             "generic; freezing them saves memory and reduces over-fitting on nuScenes-mini."),
            ("def train(self", "Overrides `nn.Module.train()` so that (a) frozen stages stay frozen after "
             "`model.train()` is called by the trainer and (b) with `norm_eval=True` every BatchNorm2d stays in eval "
             "mode — with batch size 1 × 6 images, batch statistics would be noisy, so the ImageNet running "
             "statistics are kept."),
            ("def _run", "`_run(layer, x)` executes a stage either normally or through `checkpoint(layer, x)` when "
             "gradient checkpointing is enabled (`train.grad_checkpoint`, on in the GTX 1650 profile). Checkpointing "
             "discards intermediate activations in the forward pass and recomputes them in the backward pass: "
             "less GPU memory, ~20–30 % more compute."),
            ("def forward", "Forward: `(B·N, 3, H, W)` → stem+layer1 → layer2 → `c3 (B·N, 256, H/16, W/16)` → "
             "`c4 (B·N, 512, H/32, W/32)`; `c4` is bilinearly upsampled to `c3`'s size, concatenated "
             "(`768` channels for ResNet18) and fused by the neck → `(B·N, 256, H/16, W/16)`. For 128×352 inputs "
             "this is `(6, 256, 8, 22)`."),
        ],
    },
    {
        "file": "models/camera_bev/lss.py",
        "intro": "Camera → BEV transformation (Lift-Splat-Shoot with a BEVDepth-style camera-aware depth network). "
                 "This is the most geometry-heavy module; the docstring at the top contains the exact equations "
                 "used by `get_geometry()`.",
        "blocks": [
            ('"""Camera -> BEV', "Docstring with the data flow and the frustum-to-reference-frame equations "
             "(see the *Camera → BEV* chapter for the derivation). Imports: `math` for the bin-centre helper, "
             "`torch.nn.functional` for one-hot encoding and BCE, the backbone and the shared blocks."),
            ("class DepthNet", "`DepthNet` predicts, for every image feature cell, a categorical depth distribution "
             "over D bins and a C-dimensional context feature.\n\n"
             "- `reduce`: 3×3 ConvBNReLU 256 → 256.\n"
             "- If `camera_aware`: `cam_bn` (BatchNorm1d over the 16-d camera descriptor) and `cam_mlp` "
             "(16 → 256 → 512) produce two sets of 256 SE-style gates — one for the depth branch, one for the "
             "context branch. This lets a single network serve six cameras with different intrinsics/extrinsics "
             "(BEVDepth idea).\n"
             "- `context`: ResBlock + 1×1 conv → C = 64 channels.\n"
             "- `depth`: ResBlock + dilated ResBlock (larger receptive field) + 1×1 conv → D = 59 logits."),
            ("    def forward(self, x: torch.Tensor, cam_params", "`DepthNet.forward(x (B·N, 256, h, w), cam_params "
             "(B·N, 16))`: reduce, compute sigmoid gates from the camera descriptor (cast to float for numerical "
             "stability, then back to the activation dtype), multiply them channel-wise into two copies of the "
             "features and return `(depth_logits (B·N, D, h, w), context (B·N, C, h, w))`."),
            ("class CameraBEVEncoder", "Constructor of the full camera branch. Reads from `model.camera_bev`:\n\n"
             "- `feature_stride` (16) and `depth_bins` `[1, 60, 1]` → `dbound`, `num_bins = 59`;\n"
             "- builds `ImageBackbone`, `DepthNet` and `bev_refine` (ConvBNReLU + `bev_refine_blocks` ResBlocks);\n"
             "- `register_buffer('frustum', ..., persistent=False)`: the frustum is a constant tensor that moves "
             "with `.to(device)` but is not saved in checkpoints (it is recomputed from the config)."),
            ("    def _create_frustum", "`_create_frustum()` → `(D, h, w, 3)`. For every depth bin and feature cell it "
             "stores `(u, v, d)`: `u, v` are the pixel coordinates of the cell centre in the *augmented* network "
             "input image (`(k + 0.5)·16 − 0.5`) and `d` the metric depth of the bin centre (`1.5, 2.5, … 59.5 m`). "
             "Using bin centres keeps the geometry consistent with the depth-supervision binning "
             "(`floor((d − d_min)/step)`)."),
            ("    def get_geometry", "`get_geometry()` lifts every frustum point to the BEV frame → `(B, N, D, h, w, 3)`.\n\n"
             "1. `pts = frustum − post_trans` and multiplication with `inv(post_rots)` undo the image augmentation "
             "(resize, crop, flip, rotation) — back to original 1600×900 pixel coordinates `(u', v', d)`.\n"
             "2. `cat([u'·d, v'·d], d)` forms the homogeneous image point scaled by depth.\n"
             "3. `combine = R_cam2ref · K⁻¹` back-projects into the camera frame and rotates into the reference "
             "frame in one matrix product; adding `t_cam2ref` translates. `cam2ref` already contains the "
             "extrinsics *and* the ego motion between the camera and LiDAR timestamps.\n"
             "4. `bda` (BEV data augmentation: flip/rotate/scale, identity at test time) is applied last so that "
             "camera features land in the same augmented frame as the LiDAR/radar points and GT boxes.\n\n"
             "Broadcasting via `.view(b, n, 1, 1, 1, 3, 3)` applies each camera's matrices to all D·h·w points."),
            ("    def voxel_pool", "`voxel_pool()` — the *splat*. Flattens geometry to `(B, N·D·h·w, 3)` and features to "
             "`(B, N·D·h·w, C)` (in fp32 to avoid overflow), computes integer cells `ix, iy`, keeps points inside the "
             "grid and z-range, builds a flat index `(b·ny + iy)·nx + ix` and sums all features falling into the "
             "same cell with `index_add_` → `(B·ny·nx, C)` → reshaped/permuted to `(B, C, ny, nx)`. The height "
             "dimension is collapsed (sum over all z). This replaces the CUDA \"cumsum trick\" of the original LSS "
             "with a single, deterministic-enough PyTorch op."),
            ("    @staticmethod", "`camera_params()` builds the 16-d camera descriptor for DepthNet: the augmented "
             "intrinsics `K_aug = post_rot · K` give fx/W, fy/H, cx/W, cy/H (normalised), plus the 12 entries of "
             "`cam2ref[:3, :4]`. Flattened to `(B·N, 16)`."),
            ("    def forward(self, images", "Forward pass:\n\n"
             "1. `images (B, N, 3, H, W)` → `flatten(0, 1)` → backbone → `feats (B·N, 256, h, w)`.\n"
             "2. DepthNet → `depth_logits (B·N, D, h, w)`, `context (B·N, C, h, w)`; `softmax(1)` over the depth "
             "bins gives a probability distribution per cell.\n"
             "3. `cam_valid` (False for a missing/unreadable image) zeroes that camera's depth so it contributes nothing.\n"
             "4. Lift: `depth.unsqueeze(1) * context.unsqueeze(2)` is the outer product `(B·N, C, D, h, w)`; it is "
             "reshaped and permuted to `(B, N, D, h, w, C)`.\n"
             "5. `get_geometry` + `voxel_pool` → `(B, C, ny, nx)`; `bev_refine` convolutions smooth the splatted map.\n\n"
             "Returns `bev`, `depth_logits` (for the loss) and `depth` (probabilities)."),
            ("    def depth_loss", "`depth_loss()` — BEVDepth supervision. `depth_gt (B, N, h, w)` holds the distance "
             "of the closest LiDAR point per feature cell (0 = none). Valid cells within `[d_min, d_max)` are "
             "converted to one-hot bin targets; the loss is the binary cross-entropy between the predicted "
             "probabilities and the one-hot vector, summed over bins and averaged over valid cells. Autocast is "
             "disabled because `F.binary_cross_entropy` is unsafe in fp16. If no cell is valid, `logits.sum()·0` "
             "returns a zero that is still connected to the graph."),
            ("def depth_bin_centers", "Helper returning the list of depth-bin centres for a `dbound`; not used by the "
             "model itself (kept for analysis / plotting)."),
        ],
    },
    {
        "file": "models/lidar_encoder/pillar_encoder.py",
        "intro": "LiDAR branch: dynamic pillars + explicit raster channels + a small residual CNN at full BEV resolution.",
        "blocks": [
            ('"""LiDAR encoder', "Docstring: input `(P, 1+5)` = batch, x, y, z, intensity, time-lag; output "
             "`(B, 64, ny, nx)` = 60 learned + 4 raster channels. `LIDAR_RASTER_NAMES` names the raster channels; "
             "these names appear in the dashboard's channel-gate chart."),
            ("class LidarEncoder", "Constructor. `in_features = 5`. `learned = out_channels − 4`. Builds:\n\n"
             "- `vfe`: `DynamicPillarVFE(5, (32, 64), grid)`;\n"
             "- `conv`: ConvBNReLU(64 + 4 → 60) followed by `num_res_blocks` ResBlocks (2);\n"
             "- `normalizer` buffer `[x_max, y_max, z_max, 255, 0.5]` — divides x, y, z, intensity and time-lag "
             "before the PointNet;\n"
             "- `channel_names`: 60 × `lidar_learned_i` + the 4 raster names (used for explainability)."),
            ("    def raster", "`raster()` builds the 4 explicit channels:\n\n"
             "- `bev_raster_stats(..., value_cols=[2, 3])` → density, mean z, mean intensity;\n"
             "- max height per cell via `scatter_reduce_(reduce='amax', include_self=False)` on the z column "
             "(column 3 of the batched point tensor) — `include_self=False` makes negative heights work;\n"
             "- scaling: density as is, heights / 3 m, intensity / 100.\n\nOutput `(B, 4, ny, nx)` ordered as "
             "`LIDAR_RASTER_NAMES` (density, max height, mean height, intensity)."),
            ("    def forward", "`forward(points, batch_size)`: pillar features `(B, 64, ny, nx)` + raster `(B, 4, ny, nx)` "
             "→ `conv(cat)` → `(B, 60, ny, nx)` → concatenated with the raster again → `(B, 64, ny, nx)`. Returning "
             "the raster unchanged in the last 4 channels guarantees that these physical quantities reach the "
             "feature-selection gates verbatim."),
        ],
    },
    {
        "file": "models/radar_encoder/radar_encoder.py",
        "intro": "Radar branch. Same dynamic-pillar idea as LiDAR, adapted to very sparse points (≈ 100× fewer) "
                 "and with the ego-motion-compensated Doppler velocity kept explicitly.",
        "blocks": [
            ('"""Radar encoder', "Docstring: the 10 radar point features are x, y, z, rcs, vx_comp, vy_comp, vx, vy, "
             "dyn_prop, time_lag (see `datasets/radar_loader.py: RADAR_POINT_FEATURES`). `RADAR_RASTER_NAMES` lists "
             "the 5 explicit channels."),
            ("class RadarEncoder", "Constructor. `learned = 64 − 5 = 59`. `conv` = ConvBNReLU(64 + 5 → 59) + one "
             "ResBlock per entry of `dilations` (1, 2, 4): dilated convolutions spread each isolated radar return "
             "over a larger neighbourhood (receptive field ≈ 31 cells ≈ 15 m) without downsampling. `normalizer` "
             "rescales the 10 raw columns (positions /50 m, RCS /30, velocities /10 m/s, dyn_prop /7, time-lag /0.5 s). "
             "`raster_scale` rescales the raster channels."),
            ("    def raster", "`raster()` → `(B, 5, ny, nx)`: density, mean vx_comp, mean vy_comp, mean RCS, mean "
             "time-lag (value columns 4, 5, 3, 9), divided by `[1, 10, 10, 30, 0.5]`. Velocities stay in the "
             "reference frame (rotated by the loader), so a cell's (vx, vy) is directly the object's motion over ground."),
            ("    def forward", "Same structure as the LiDAR encoder: VFE + raster → conv → concatenate raster → "
             "`(B, 64, ny, nx)`. The last five channels are exactly the radar raster (velocity preserved)."),
        ],
    },
    {
        "file": "models/feature_selection/selection.py",
        "intro": "**The research component.** Learned, camera-guided, cross-modal selection of LiDAR and radar "
                 "BEV features before fusion. See the *Deep-Learning Feature Selection* chapter for the method, "
                 "the losses and what the trained network actually learned.",
        "blocks": [
            ('"""Camera-Guided', "The long docstring is the design rationale: the problem (redundant / noisy "
             "channels when concatenating), the mechanism (channel gates `c_m` and spatial gates `s_m`, "
             "`F̃_m = F_m ⊙ c_m ⊙ s_m`), why SE/CBAM-style gating instead of global cross-attention (memory: "
             "40 000 BEV tokens), the optional hard-concrete binary masks, and the shapes."),
            ("from __future__", "Imports: `torch`, `nn` and the shared `ConvBNReLU` block."),
            ("class CrossModalFeatureSelection", "Class docstring: `channels` maps each *selectable* modality to its "
             "channel count (`{'lidar': 64, 'radar': 64}` for the full model; only one entry for camera+LiDAR or "
             "camera+radar), `guide_channels` is the camera BEV channel count or `None`."),
            ("    def __init__", "Constructor, part 1 — configuration and the **channel branch**:\n\n"
             "- stores the modality list, `gate_mode` (`soft` | `hard_concrete`), `temperature`, `threshold` "
             "(0.5; a gate above it counts as *selected*) and `init_open` (2.0);\n"
             "- `all_channels` = selectable modalities + `camera` if a guide is used;\n"
             "- `stat_dim = 2 · Σ channels` — each input contributes a mean and a max per channel "
             "(2 · 3 · 64 = 384 for model F);\n"
             "- `hidden = max(stat_dim // channel_reduction, 32)` = 96;\n"
             "- `channel_mlp`: Linear(384 → 96) → ReLU → Linear(96 → Σ selectable channels = 128). Because the input "
             "contains statistics of *all* sensors, the LiDAR gates can depend on the radar and camera content "
             "(cross-modal squeeze-and-excitation)."),
            ("        compress = 8", "Constructor, part 2 — the **spatial branch** and initialisation:\n\n"
             "- `compress`: per input a 1×1 ConvBNReLU 64 → 8 channels;\n"
             "- the spatial network input is, per input, 8 compressed channels + channel-mean + channel-max "
             "= 10 channels → `(compress + 2) · len(all_channels)` = 30 for model F;\n"
             "- `spatial`: 7×7 ConvBNReLU(30 → 32) → 3×3 ConvBNReLU(32 → 32) → 3×3 Conv(32 → #selectable) "
             "= one logit map per selectable sensor;\n"
             "- both final layers get weights `N(0, 0.01)` and bias `init_open = 2.0`, so at the start of training "
             "every gate is ≈ σ(2) = 0.88: the network starts from \"keep almost everything\" and learns what to "
             "suppress, instead of starting from random masks that would destroy the signal."),
            ("    def _gate", "`_gate(logits)` converts logits to gates:\n\n"
             "- `soft` (default): `sigmoid(logits)` — continuous importance in (0, 1).\n"
             "- `hard_concrete`, training: logistic noise `log u − log(1 − u)` is added, divided by the temperature "
             "and squashed (a Gumbel/Concrete relaxation); `hard = (soft > 0.5)`; the returned value "
             "`hard + soft − soft.detach()` equals the binary mask in the forward pass but has the gradient of "
             "`soft` (straight-through estimator).\n"
             "- `hard_concrete`, evaluation: deterministic `logits > 0`.\n"
             "- Any other mode raises `ValueError`."),
            ("    def forward(self, feats", "`forward(feats, guide)`:\n\n"
             "1. Collects the selectable inputs and, if configured, the camera guide (a `ValueError` is raised when "
             "a guide is expected but missing).\n"
             "2. **Channel statistics**: for every input, `mean` and `amax` over H and W (in fp32) → concatenated "
             "`(B, stat_dim)`; `channel_mlp` → `(B, Σ C_m)` logits, split per sensor into `(B, C_m)`.\n"
             "3. **Spatial descriptor**: for every input `[compress(x) (8 ch), mean_c(x), max_c(x)]` → concatenated "
             "`(B, 30, H, W)` → `spatial` → `sp_logits (B, M, H, W)`."),
            ("        selected, channel_gates", "4. For each selectable sensor `m` (index `i`): `c_gate = _gate(channel "
             "logits)` `(B, C_m)`, `s_gate = _gate(sp_logits[:, i:i+1])` `(B, 1, H, W)`, and the selected features "
             "`F_m · c_gate[:, :, None, None] · s_gate` `(B, C_m, H, W)` — the channel gate broadcasts over space, "
             "the spatial gate over channels. Gates are cast to the feature dtype for AMP.\n"
             "5. `info` returns gates and logits; the gates feed `feature_selection_loss`, the dashboard's "
             "explainability views and `selection_stats`."),
            ("    @torch.no_grad()", "`selection_stats(info)` (no gradients) summarises the gates per sensor: fraction "
             "of spatial cells / channels with gate > threshold, mean spatial / channel importance, and the "
             "*effective selected ratio* = (fraction of selected cells) × (fraction of selected channels), averaged "
             "over the batch. These numbers are logged during validation, stored in the ablation results and shown "
             "in the dashboard."),
        ],
    },
    {
        "file": "models/feature_selection/losses.py",
        "intro": "Training signal for the selection gates (`L_feature_selection` in the total loss).",
        "blocks": [
            ('"""Feature-selection losses', "Docstring with the three terms and their rationale."),
            ("def feature_selection_loss", "Implementation:\n\n"
             "- `spatial` stacks the spatial gates of all selectable sensors → `(B, M, H, W)`; "
             "`combined = max over sensors` (clamped to ≥ 1e-4 for the log).\n"
             "- **coverage** `Σ y · (−log combined) / Σ y` with `y = fg_heatmap (B, 1, H, W)` (class-agnostic Gaussian "
             "object-centre heatmap built by the dataset). Only positives are penalised: near objects, at least one "
             "sensor must keep its features; background is left to the sparsity term.\n"
             "- **sparsity** `Σ_m |mean(s_m) − ρ_s| + |mean(c_m) − ρ_c|` with `ρ_s = 0.35`, `ρ_c = 0.75` "
             "(`fusion.selection.target_keep_ratio`): forces the network to keep only a target fraction.\n"
             "- **binarization** `mean(4g(1 − g))` over all gates: 0 when every gate is 0 or 1, 1 when all are 0.5 → "
             "pushes towards crisp, interpretable decisions.\n"
             "- `total = w_cov·coverage + w_sp·sparsity + w_bin·binarization` with weights from "
             "`loss.feature_selection` (1.0 / 0.5 / 0.1). All components are returned for logging."),
        ],
    },
    {
        "file": "models/fusion/fusion_methods.py",
        "intro": "The interchangeable fusion modules. Both take a dict of BEV maps and return `(fused, info)`.",
        "blocks": [
            ('"""BEV fusion methods', "Docstring and imports (CBAM, ConvBNReLU, ResBlock)."),
            ("class ConcatFusion", "`ConcatFusion` (baseline, experiments A–C): concatenate all inputs along the "
             "channel axis (`Σ C_m` channels, e.g. 128 for LiDAR + radar), 3×3 ConvBNReLU to `out_channels` (128) "
             "and one ResBlock. `info` is empty."),
            ("class AttentionGatedFusion", "`AttentionGatedFusion` (experiments D–I).\n\n"
             "- `proj[m]`: 1×1 ConvBNReLU projecting each sensor to the common space `(B, 128, H, W)` (`P_m`).\n"
             "- `attention`: 3×3 ConvBNReLU over the concatenated raw inputs → 1×1 conv → `M` logits per cell.\n"
             "- `mix`: 1×1 ConvBNReLU over `128·(M+1)` channels → ResBlock → CBAM."),
            ("    def forward(self, feats",
             "Forward: `weights = softmax_m(attention(...))` `(B, M, H, W)` — a per-cell probability over sensors "
             "(computed in fp32); `weighted = Σ_m a_m · P_m`; the mixer sees `[weighted, P_1, …, P_M]` so it can "
             "still use complementary information that the attention weighting would average away. Returns the "
             "fused map `(B, 128, H, W)` and `info = {modality_weights, modality_order}` (visualised as the "
             "\"fusion attention\" RGB map in the dashboard)."),
        ],
    },
    {
        "file": "models/fusion/registry.py",
        "intro": "A tiny registry that decouples the fusion choice (`fusion.method` in YAML) from the model code.",
        "blocks": [
            ('"""Fusion-method registry', "Docstring: the contract a fusion class must satisfy."),
            ("FUSION_REGISTRY", "`FUSION_REGISTRY` maps names to classes. `register_fusion(name)` is a decorator to "
             "add new methods from any module; `build_fusion(method, in_channels, out_channels)` instantiates the "
             "configured class or raises `KeyError` listing the available names."),
        ],
    },
    {
        "file": "models/fusion/bev_backbone.py",
        "intro": "SECOND-FPN 2-D backbone on the fused BEV map (shared by every experiment).",
        "blocks": [
            ('"""SECOND-FPN', "Docstring with the default shapes: `(B, 128, 200, 200)` → block1 `(B, 128, 100, 100)` "
             "→ block2 `(B, 256, 50, 50)`; deblocks bring both to 100 × 100 and the outputs are concatenated → "
             "`(B, 256, 100, 100)`."),
            ("class SECONDFPN", "Constructor iterates over the zipped config lists (`layer_nums [3, 5]`, `layer_strides "
             "[2, 2]`, `num_filters [128, 256]`, `upsample_strides [1, 2]`, `num_upsample_filters [128, 128]`): each "
             "block = stride-s ConvBNReLU + n ConvBNReLU; each deblock = 1×1 ConvBNReLU (upsample 1) or "
             "ConvTranspose2d (upsample 2). `out_channels = Σ num_upsample_filters = 256`; `out_stride = first "
             "stride = 2` (checked against `detection_head.out_stride` in the model)."),
            ("    def forward", "Runs the blocks sequentially (optionally through gradient checkpointing), applies each "
             "deblock to the corresponding block output and concatenates along channels."),
        ],
    },
    {
        "file": "models/detection/targets.py",
        "intro": "CenterPoint training targets, generated with NumPy inside the DataLoader workers "
                 "(`NuScenesFusionDataset.__getitem__`).",
        "blocks": [
            ('"""CenterPoint training targets', "Docstring: what each regression channel means."),
            ("def gaussian_radius", "`gaussian_radius(height, width, min_overlap)` — the CornerNet/CenterPoint formula: "
             "solves three quadratic equations for the largest centre shift that still keeps an IoU of "
             "`min_overlap` (0.1) with the GT box, and returns the minimum radius (in cells)."),
            ("def draw_gaussian", "`draw_gaussian(heatmap, center, radius)` renders a 2-D Gaussian (σ = diameter / 6) "
             "with peak 1.0 at `center = (ix, iy)` into `heatmap[iy, ix]` with element-wise maximum (overlapping "
             "objects keep the larger value). Clipping handles boxes near the grid border."),
            ("def build_center_targets", "`build_center_targets(boxes (M, 9), labels, velocity_valid, grid, ...)` on the "
             "**head grid** (100 × 100, 1 m cells):\n\n"
             "- `heatmap (K=10, ny, nx)`: one Gaussian per object in its class channel; radius from the box "
             "length/width in cells, at least `min_radius` (2);\n"
             "- `ind (max_objects,)`: flat index `iy·nx + ix` of each object's centre cell;\n"
             "- `mask`: 1 for real objects; `vel_mask`: 1 if the annotation velocity is defined;\n"
             "- `reg (max_objects, 10)`: `[dx, dy, z, log l, log w, log h, sin yaw, cos yaw, vx, vy]` where "
             "`dx, dy` are the sub-cell offsets of the true centre.\n\nObjects outside the grid or with "
             "non-positive size are skipped; at most `max_objects` (500)."),
            ("def foreground_heatmap", "`foreground_heatmap(boxes, grid)` → `(1, ny, nx)` class-agnostic Gaussians at "
             "the **BEV** resolution (0.5 m, 200 × 200). Used by the feature-selection coverage loss and the fusion "
             "deep-supervision loss."),
        ],
    },
    {
        "file": "models/detection/center_head.py",
        "intro": "Detection head and its losses.",
        "blocks": [
            ('"""CenterPoint-style', "Docstring: output branches and the loss summary. `HEAD_BRANCHES` lists the "
             "regression branches and their channel counts."),
            ("class CenterHead", "`CenterHead`: a shared 3×3 ConvBNReLU (256 → 64) followed by one small head per "
             "branch (3×3 ConvBNReLU + 1×1 conv). The heatmap bias is initialised to −2.19 = logit(0.1) so the "
             "initial predicted object probability is 0.1 everywhere (CenterNet trick; prevents a huge initial "
             "focal loss). `forward` returns a dict of `(B, ch, 100, 100)` maps."),
            ("def gaussian_focal_loss", "`gaussian_focal_loss(pred, target)` — CornerNet focal loss on probabilities. "
             "Positives are cells where the target equals exactly 1 (Gaussian peaks): `−log(p)·(1 − p)^α`. "
             "Negatives: `−log(1 − p)·p^α·(1 − y)^γ` — the `(1 − y)^γ` term (γ = 4) reduces the penalty near object "
             "centres. Returns the summed loss and the number of positives (for normalisation)."),
            ("def _gather", "`_gather(feat (B, C, H, W), ind (B, K))` → `(B, K, C)`: picks the predictions at the "
             "object-centre cells using `gather` on the flattened spatial dimension."),
            ("def detection_loss", "`detection_loss()`:\n\n"
             "- classification: focal loss on `sigmoid(heatmap)` normalised by the number of positives;\n"
             "- regression: predictions gathered at `ind` and concatenated in the same order as the targets "
             "`(B, K, 10)`; L1 error weighted by `code_weights` (velocity 0.2);\n"
             "- split into `center` (dx, dy, z), `bbox` (log sizes), `orientation` (sin, cos) and `velocity` "
             "(masked by `vel_mask`), each normalised by the number of objects;\n"
             "- `detection` = weighted sum with `loss.detection` weights (1.0 heatmap, 0.25 for the others)."),
            ("def aux_heatmap_loss", "`aux_heatmap_loss(logits, target)` — class-agnostic heatmap loss used for fusion "
             "deep supervision; adaptive max-pooling resizes the target if the prediction has a different resolution."),
        ],
    },
    {
        "file": "models/detection/decoder.py",
        "intro": "Converts head outputs to metric 3-D boxes in the reference frame.",
        "blocks": [
            ('"""Decode CenterHead', "Imports; NumPy is used for the greedy NMS loop on the CPU."),
            ("def circle_nms", "`circle_nms(centers, scores, radius)` — CenterPoint's distance-based NMS: sort by "
             "score, compute all pairwise centre distances once (`torch.cdist` on CPU), then greedily keep a box "
             "and suppress every other box within `radius` metres. Running the loop on the CPU avoids one GPU "
             "synchronisation per iteration."),
            ("@torch.no_grad()", "`decode_detections()`:\n\n"
             "1. `heat = sigmoid(heatmap)`; a 3×3 max-pool keeps only local peaks (`heat == peak`) — the "
             "CenterNet replacement for NMS.\n"
             "2. `topk(max_detections)` over the flattened `C·H·W` scores → `labels = idx // (H·W)`, "
             "`pix = idx % (H·W)`, `iy, ix`.\n"
             "3. `gather` reads the regression branches at those pixels.\n"
             "4. Metric box: `x = (ix + dx)·r + x_min`, `y = (iy + dy)·r + y_min`, `z`, `l, w, h = exp(log size)` "
             "(log-size clamped at 4 → at most e⁴ ≈ 55 m, protects against an untrained head), `yaw = atan2(sin, cos)`, "
             "`vx, vy`.\n"
             "5. Per sample: score threshold, then class-wise circle NMS with radii from "
             "`model.detection_head.nms_radius`; results sorted by score.\n\nReturns a list (one per sample) of "
             "`{boxes (K, 9), scores (K,), labels (K,)}`."),
        ],
    },
    {
        "file": "models/trajectory/planning_head.py",
        "intro": "The learned planner (Stage 6) — predicts six future ego positions from the BEV features.",
        "blocks": [
            ('"""Learned trajectory planner', "Docstring with the architecture and the reason for residual prediction "
             "on top of a constant-velocity prior (trainable on mini)."),
            ("class PlanningHead", "Constructor:\n\n"
             "- `reduce`: three stride-2 ConvBNReLU layers on `[BEV features, x-coord, y-coord]` (two coordinate "
             "channels give the attention a sense of position): 100 → 50 → 25 → 13, so 169 tokens of 128 channels;\n"
             "- `query`: MLP over `[speed, yaw rate, accel] / state_scale` + one-hot command (3);\n"
             "- `attn`: 4-head cross-attention (query → BEV tokens);\n"
             "- `gru`: GRUCell over `[context, current residual]`;\n"
             "- `out`: Linear(128 → 2) **initialised to zero**, so an untrained head outputs exactly the "
             "constant-velocity prior."),
            ("    def forward", "Forward:\n\n"
             "1. Normalised coordinate channels are concatenated to the BEV map; `reduce` → tokens `(B, L=169, 128)`.\n"
             "2. The ego query `(B, 1, 128)` attends over the tokens → context `(B, 128)` and attention weights "
             "`(B, 1, 169)` (reshaped to 13 × 13 for visualisation).\n"
             "3. Prior: `prior_t = (speed · t·dt, 0)` for t = 1…6 (speed clamped ≥ 0).\n"
             "4. The GRU unrolls six steps; each step adds a predicted delta to the running residual.\n"
             "5. `waypoints = prior + residuals` `(B, 6, 2)` in the reference frame (x forward, y left)."),
        ],
    },
    {
        "file": "models/trajectory/losses.py",
        "intro": "Losses of the learned planner (`L_trajectory`).",
        "blocks": [
            ('"""Trajectory-planning losses', "Docstring describing the four terms."),
            ("def ego_footprint_points", "`ego_footprint_points(waypoints (B, T, 2), ego_cfg)` → `(B, T, 5, 2)`: the "
             "heading of each step is the direction from the previous waypoint (0 when standing still); five points "
             "are placed on the ego body (rear, centre, front, left, right) using the vehicle length/width and the "
             "rear-axle-to-centre offset — waypoints are rear-axle positions."),
            ("def sample_occupancy", "`sample_occupancy(occ (B, T, H, W), points (B, T, P, 2), grid)` converts metric "
             "points to normalised grid coordinates (x → width axis, y → height axis, `align_corners=False`) and "
             "bilinearly samples the occupancy with `grid_sample`. Differentiable w.r.t. the points → the collision "
             "penalty pushes waypoints out of occupied cells."),
            ("def trajectory_loss", "`trajectory_loss()`:\n\n"
             "- `reg`: L1 distance to the logged trajectory, masked where the future is unknown (end of a scene);\n"
             "- `collision`: GT future occupancy of other agents, softened with a 3×3 average pool, sampled at the "
             "5 footprint points; the max over points is averaged over valid steps;\n"
             "- `smooth`: squared second differences of `[−v·dt, 0, waypoints]` (includes the current velocity);\n"
             "- `goal`: Euclidean error of the last valid waypoint;\n"
             "- `trajectory` = weighted sum (`loss.trajectory`: 1.0 / 2.0 / 0.1 / 0.5)."),
        ],
    },
    {
        "file": "models/fusion_model.py",
        "intro": "`MultiSensorFusionNet` wires all sub-networks together according to the configuration and "
                 "implements the configurable total loss. Every training script, evaluator and the dashboard use "
                 "this class.",
        "blocks": [
            ('"""MultiSensorFusionNet', "Docstring: ASCII architecture and the total-loss formula."),
            ("from __future__", "Imports of every sub-module. `ALL_LOSSES` lists the loss groups a training stage "
             "can activate (`configs/training.yaml → stages.<name>.losses`)."),
            ("class MultiSensorFusionNet", "Constructor, part 1 — **sensor encoders**. Reads `fusion.use_camera_bev / "
             "use_lidar / use_radar` (set per experiment in `configs/ablation.yaml`). At least one must be on "
             "(`ValueError` otherwise). Each enabled encoder is created and its output channel count recorded in "
             "`channels` (insertion order camera, lidar, radar defines `self.modalities`). Disabled encoders are "
             "simply not created, so a LiDAR-only model has no camera parameters at all."),
            ("        selectable =", "Constructor, part 2 — **feature selection and fusion**:\n\n"
             "- `selectable` = LiDAR and/or radar channels; feature selection is built only if "
             "`fusion.feature_selection` is true *and* at least one of them is enabled; the camera is used as guide "
             "if enabled and `selection.camera_guided`;\n"
             "- `build_fusion(fusion.method, channels, 128)` creates the fusion module;\n"
             "- `aux_heads`: a 1×1 conv per sensor + one for the fused map (deep supervision, bias −2.19)."),
            ("        self.bev_backbone = SECONDFPN", "Constructor, part 3 — **backbone and heads**: SECOND-FPN, the "
             "head grid (`grid.downsample(out_stride)` = 100 × 100), a consistency check between the backbone "
             "stride and the head stride, `CenterHead` with 10 classes, and the optional `PlanningHead` (horizon 6, "
             "dt 0.5 s)."),
            ("    def set_grad_checkpoint", "Utilities. `set_grad_checkpoint` toggles checkpointing in the image backbone "
             "and the BEV backbone. `module_groups` lists top-level children. `freeze(names)` disables gradients "
             "for every parameter whose name starts with one of the prefixes (e.g. `camera_bev.backbone`) and "
             "returns their names — used by the staged training."),
            ("    def forward(self, batch", "Forward pass, part 1 — encoders. A `StageTimer` (or a no-op `NullTimer`) "
             "measures each stage (used by the dashboard's live pipeline strip). Each enabled encoder is run on its "
             "batch entries; the camera encoder also returns `depth_logits`/`depth` for the depth loss; LiDAR/radar "
             "rasters are kept in the output for analysis. `out['raw']` keeps the unselected maps."),
            ("        selected = dict(feats)", "Forward pass, part 2 — selection, fusion, heads:\n\n"
             "- `selected` starts as a copy of the raw maps; with feature selection, the LiDAR/radar entries are "
             "replaced by their selected versions and the gate info stored in `out['selection']`;\n"
             "- fusion → `out['fused'] (B, 128, 200, 200)` and `fusion_info`;\n"
             "- in training mode with deep supervision, auxiliary heatmap logits are computed for every (selected) "
             "sensor and for the fused map;\n"
             "- backbone → `bev_features (B, 256, 100, 100)` → `det` dict;\n"
             "- the planning head runs only if the batch contains `ego_state` (always true for dataset batches)."),
            ("    def compute_losses", "`compute_losses(out, batch, loss_cfg, active)` builds\n"
             "`L = λ_det·L_detection + λ_traj·L_trajectory + λ_fs·L_feature_selection + λ_fus·L_fusion + λ_depth·L_depth`, "
             "adding only the groups listed in `active` and only when their inputs exist (e.g. no trajectory loss "
             "without a planning head, no depth loss without a camera or depth targets). Every sub-loss is "
             "returned for logging together with `total` (fp32 scalar)."),
            ("    def decode", "`decode(out, score_threshold)` calls `decode_detections` with the head grid, class list, "
             "threshold, `max_detections` and per-class NMS radii from the config. `build_model(cfg)` is the "
             "factory used everywhere."),
        ],
    },
]
