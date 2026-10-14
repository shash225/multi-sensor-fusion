"""Block-by-block walkthroughs: utilities (utils/*) and the data layer (datasets/*)."""

WALKTHROUGHS = [
    {
        "file": "utils/config.py",
        "intro": "Layered YAML configuration. Every tunable value of the project flows through `load_config()`; "
                 "no path or hyper-parameter is hard-coded in Python.",
        "blocks": [
            ('"""Layered YAML', "Docstring: the merge order (base YAML files → hardware profile → experiment overrides "
             "→ `configs/local.yaml` → command-line `--set key=value`) and the `${env:NAME,default}` syntax."),
            ("from __future__", "Imports: `copy` (deep copies of nested dicts), `os` (environment variables, user "
             "paths), `re` (environment-variable interpolation), `pathlib.Path`, typing helpers and PyYAML."),
            ("PROJECT_ROOT =", "Module constants. `PROJECT_ROOT` is computed from this file's location "
             "(`utils/` → parent), so the project works from any working directory. `CONFIG_DIR` = `configs/`. "
             "`BASE_FILES` fixes the order in which the five base YAML files are merged. `_ENV_PATTERN` matches "
             "`${env:NAME}` or `${env:NAME,default}`."),
            ("class Config(dict)", "`Config` is a `dict` subclass with attribute access (`cfg.dataset.root` instead of "
             "`cfg['dataset']['root']`).\n\n"
             "- `__init__` recursively wraps nested dicts/lists via `_wrap`.\n"
             "- `__getattr__` converts a missing key into `AttributeError` (required for `hasattr`, `copy`, `pickle`).\n"
             "- `__setattr__` stores attributes as keys.\n"
             "- `get_path('a.b.c', default)` / `set_path('a.b.c', value)` read and write nested keys (used for "
             "command-line overrides and to patch machine-specific keys into checkpoint configs).\n"
             "- `to_dict()` returns plain dicts (stored inside checkpoints)."),
            ("def _wrap", "`_wrap` / `_unwrap`: recursive conversion between plain containers and `Config`."),
            ("def deep_merge", "`deep_merge(base, update)`: recursive in-place merge; nested mappings are merged key by "
             "key, everything else (including lists) is replaced by a deep copy. This is what lets a hardware "
             "profile change only `dataset.camera.image_size` without repeating the other camera settings."),
            ("def _interpolate_env", "`_interpolate_env` walks the merged structure and replaces `${env:NAME,default}` "
             "inside strings with the environment value (or the default). `dataset.root` defaults to "
             "`${env:NUSCENES_ROOT,/path/to/nuscenes}`."),
            ("def read_yaml", "`read_yaml(path)` → dict (empty dict for an empty file). `parse_override('a.b=3')` → "
             "`('a.b', 3)`; the value is parsed as YAML so `true`, `[128,352]`, `0.5` get their proper types. "
             "`list_experiments()` returns the `experiments` table of `configs/ablation.yaml`."),
            ("def load_config", "`load_config(experiment, hardware, overrides, config_dir, use_local)` — the central "
             "function:\n\n"
             "1. merge the base files in `BASE_FILES` order;\n"
             "2. read `local.yaml` (optional) and parse the CLI overrides into a nested dict;\n"
             "3. choose the hardware profile: explicit argument > CLI > local.yaml > base; merge "
             "`configs/hardware/<name>.yaml` (unknown name → `FileNotFoundError`);\n"
             "4. choose the experiment the same way and merge its `overrides` (unknown → `KeyError` listing the "
             "available names); record `runtime.experiment` and its description;\n"
             "5. merge local.yaml, then the CLI overrides (highest priority);\n"
             "6. interpolate environment variables and return a `Config`."),
            ("def resolve_path", "`resolve_path(p)`: absolute paths are returned unchanged, relative ones are anchored "
             "at the project root (used for `outputs/`, `checkpoints/`, the cache directory)."),
        ],
    },
    {
        "file": "utils/geometry.py",
        "intro": "NumPy rigid-body geometry. The docstring defines the coordinate conventions used everywhere "
                 "(quaternion order, matrix naming, reference frame, box layout).",
        "blocks": [
            ('"""Rigid-body geometry', "Conventions (must be respected by any new code): quaternions `[w, x, y, z]`; "
             "matrices named `<src>2<dst>`; reference frame = ego at the LIDAR_TOP key-frame; boxes "
             "`[x, y, z, l, w, h, yaw, vx, vy]` (nuScenes stores `[w, l, h]` — converted once in the info builder)."),
            ("def quat_to_rotmat", "`quat_to_rotmat(q)`: normalises the quaternion and returns the standard 3×3 rotation "
             "matrix. `rotmat_to_quat(R)`: inverse conversion (Shepperd's method with the four branches for "
             "numerical stability; returns w ≥ 0). Used to write rotations in the nuScenes submission format."),
            ("def transform_matrix", "`transform_matrix(t, q_or_R)` builds a 4×4 homogeneous transform; "
             "`invert_transform(T)` inverts a rigid transform analytically (`[Rᵀ, −Rᵀt]`), cheaper and more accurate "
             "than `np.linalg.inv`; `apply_transform(T, points)` transforms the first three columns of an `(N, 3+)` "
             "array and keeps extra columns (intensity, time-lag) untouched."),
            ("def yaw_from_rotmat", "`yaw_from_rotmat(R) = atan2(R[1,0], R[0,0])` — heading of the rotated x-axis "
             "projected onto the ground plane (identical to the devkit's `quaternion_yaw`). `wrap_angle` maps angles "
             "to [−π, π); `rot2d(θ)` is a 2×2 rotation."),
            ("def box_corners_bev", "`box_corners_bev(boxes)` → `(N, 4, 2)`: corners front-left, front-right, rear-right, "
             "rear-left in the local box frame `(±l/2, ±w/2)`, rotated by yaw (batched with `einsum`) and translated. "
             "`box_corners_3d` adds top (z + h/2) and bottom (z − h/2) faces → `(N, 8, 3)` (used to draw boxes in "
             "camera images)."),
            ("def points_in_boxes", "`points_in_boxes(points (P, 3), boxes (B, 7+))` → boolean `(P, B)`: each point is "
             "rotated into each box's frame by −yaw and tested against the half-extents. Used by the calibration "
             "validation (points inside GT boxes vs. `num_lidar_pts`) and the bike-rack rule of the evaluator."),
            ("def transform_boxes", "`transform_boxes(boxes, T)`: centres are transformed as points, the heading as a "
             "*vector* `(cos yaw, sin yaw, 0)` rotated by `R` (so pitch/roll in `T` cannot corrupt yaw), velocities as "
             "vectors (rotation only). Used to bring every annotation from the global frame into the reference frame."),
            ("def project_to_image", "`project_to_image(p_cam, K)` → pixels `(N, 2)` and depths `(N,)` (pinhole model, "
             "division by z; invalid divisions are silenced)."),
            ("def bev_rotation_matrix", "`bev_rotation_matrix(rot, scale, flip_x, flip_y)` builds the 4×4 BEV "
             "augmentation matrix `bda = scale · R_z(rot) · Flip`. `augment_boxes_bev(boxes, bda)` applies it to "
             "boxes: centres transformed, sizes multiplied by the scale (the cube root of |det|), heading and velocity "
             "transformed as vectors — so a mirror (`flip_y`) correctly negates yaw and vy."),
        ],
    },
    {
        "file": "utils/bev_grid.py",
        "intro": "The single BEV grid definition shared by every module — the key to keeping camera, LiDAR, radar, "
                 "targets, planning and visualisation aligned.",
        "blocks": [
            ('"""The single Bird', "Docstring: tensors are `(B, C, H = ny, W = nx)`; rows index y (left), columns index x "
             "(forward); cell `(iy, ix)` covers `[x_min + ix·r, x_min + (ix+1)·r) × [y_min + iy·r, …)`."),
            ("@dataclass(frozen=True)", "`BEVGrid` is an immutable dataclass with `x_range`, `y_range`, `z_range`, "
             "`resolution`. Properties `nx`, `ny`, `shape`; `downsample(stride)` returns the same extent with coarser "
             "cells (the detection-head grid: stride 2 → 100 × 100); `world_to_grid` / `grid_to_world` convert "
             "between metres and continuous cell coordinates (cell centres at k + 0.5); `in_range` masks points; "
             "`cell_centers` returns the metric centres; `from_config(cfg.bev)` builds the grid from YAML."),
        ],
    },
    {
        "file": "utils/raster.py",
        "intro": "Rasterisation helpers (OpenCV) for boxes and points on a `BEVGrid`.",
        "blocks": [
            ('"""Rasterisation', "Imports and `_SHIFT = 4`: OpenCV fixed-point precision (coordinates are multiplied "
             "by 2⁴ for sub-pixel accurate polygon filling)."),
            ("def rasterize_boxes", "`rasterize_boxes(boxes, grid, inflate, values, out)` fills oriented boxes into an "
             "`(ny, nx)` uint8 array.\n\n"
             "- `cv2.fillPoly` also fills pixels on the polygon boundary, which adds about half a cell on every "
             "side; the code therefore shrinks length and width by one cell (`− grid.resolution`, never below "
             "1 % of a cell) so the filled area matches the metric area (verified by `tests/test_geometry.py`).\n"
             "- Corners are converted to grid coordinates (x → column, y → row), shifted by −0.5 cell because "
             "OpenCV pixel centres sit on integer coordinates, and drawn with sub-pixel precision.\n"
             "- `inflate` grows the box by a safety margin on each side (used by the planner)."),
            ("def rasterize_points", "`rasterize_points(points, grid)` → `(ny, nx)` float32 point counts "
             "(`np.add.at` handles repeated cells)."),
        ],
    },
    {
        "file": "utils/device.py",
        "intro": "Device selection, the fp16 reliability self-test and GPU memory reporting.",
        "blocks": [
            ('"""Device selection', "Imports."),
            ("def get_device", "`get_device(preference)`: `auto` → CUDA if available else CPU; a requested CUDA device "
             "falls back to CPU when CUDA is unavailable (the program never crashes for lack of a GPU)."),
            ("_FP16_OK", "`fp16_is_reliable(device)` — added after a real failure on the development machine: with "
             "cuDNN 9.10 the GTX 1650 returns NaN for fp16 3×3 convolutions with 256 input channels. The function "
             "runs exactly that convolution type in fp16 and fp32 on three tensor shapes that occur in the model "
             "and compares the outputs (finite and < 2 % relative error). The result is cached per device; a "
             "warning is logged when fp16 is broken."),
            ("def amp_enabled", "`amp_enabled(device, requested)`: mixed precision is used only if requested, on CUDA, "
             "and when the self-test passes. Every training/evaluation/inference code path calls this function, so "
             "a broken GPU silently falls back to fp32 instead of producing NaN losses."),
            ("def gpu_memory_report", "`gpu_memory_report` returns allocated / peak / reserved / total MiB and the GPU "
             "name (empty on CPU). `move_to_device` recursively moves tensors inside dicts/lists/tuples (batches "
             "contain nested lists and metadata dicts)."),
        ],
    },
    {
        "file": "datasets/nuscenes_db.py",
        "intro": "A lightweight replacement for the devkit's `NuScenes` class: loads the 13 JSON tables and builds the "
                 "reverse indexes the rest of the code needs. The official devkit is only needed for the official "
                 "evaluation and the full train/val split lists.",
        "blocks": [
            ('"""Lightweight nuScenes', "Docstring and the list `TABLES` of the 13 nuScenes tables."),
            ("class NuScenesDB", "Constructor: checks that `<root>/<version>/` exists (clear `FileNotFoundError` "
             "telling the user to set `dataset.root`), loads every table with `json.load` into `self.tables[name]` "
             "(list of records) and `self._index[name]` (token → record dict), builds the reverse index and logs a "
             "summary. For v1.0-mini this takes ≈ 0.5 s."),
            ("    def _build_reverse_index", "`_build_reverse_index()` adds the same derived fields as the devkit:\n\n"
             "- `sample['data'] = {channel: sample_data_token}` for key-frames and `sample['anns'] = [...]`;\n"
             "- `sample_data['channel']` and `['sensor_modality']` (via calibrated_sensor → sensor);\n"
             "- `annotation['category_name']` (via instance → category);\n"
             "- `scene name → scene` and `log token → map` lookups."),
            ("    def get(self", "Accessors: `get(table, token)` (with a helpful `KeyError`), `has`, `scenes`, "
             "`scene_by_name`, `scene_sample_tokens(scene)` (follows the `next` chain from the first sample → "
             "key-frames in temporal order), `iter_sweeps(sd_token, 'prev'|'next')` (walks a sensor's sample_data "
             "chain including non-key-frame sweeps), `map_for_log`, `file_path`."),
        ],
    },
    {
        "file": "datasets/info_builder.py",
        "intro": "Pre-computes one self-contained *info* dictionary per key-frame: file paths, calibration and ego-pose "
                 "matrices, ground truth in the reference frame, ego state, ego future/past and future agents. The "
                 "result is cached as a pickle so training and the dashboard never parse the tables again.",
        "blocks": [
            ('"""Pre-compute per-sample', "Docstring: the transformation chain `sensor → ego(t_sensor) → global → ref` "
             "and why going through the global frame with each sensor's own ego pose performs the time "
             "synchronisation."),
            ("from __future__", "Imports, `INFO_FORMAT_VERSION` (bumped when the info layout changes → invalidates "
             "old caches), the official `CATEGORY_TO_DETECTION` mapping from nuScenes categories to the 10 detection "
             "classes, and `COMMANDS` (index 0 = RIGHT, 1 = LEFT, 2 = STRAIGHT)."),
            ("def _sd_matrices", "`_sd_matrices(db, sd)` returns `sensor2ego` (calibrated_sensor) and `ego2global` "
             "(ego_pose at this sample_data's timestamp) as 4×4 matrices."),
            ("def _sensor_record", "`_sensor_record()` builds the per-sweep record: relative file path (keeps the cache "
             "portable across machines), timestamp, `time_lag` = (t_ref − t_sweep) in seconds, both matrices and the "
             "combined `sensor2ref = inv(ref_ego2global) · ego2global · sensor2ego`. Cameras also get the intrinsic "
             "matrix and the image size."),
            ("def annotation_velocity", "`annotation_velocity()` reproduces `NuScenes.box_velocity`: central difference "
             "of the annotation positions using the previous and next annotation of the same instance (one-sided at "
             "the ends), `nan` when the instance has a single annotation or the time gap exceeds 1.5 s (3 s for a "
             "central difference)."),
            ("def _ann_box_global", "`_ann_box_global(ann)` converts a nuScenes annotation to `[x, y, z, l, w, h, yaw]` "
             "in the global frame — note the swap from nuScenes `size = [w, l, h]`."),
            ("def _boxes_for_sample", "`_boxes_for_sample()` collects all annotations of a sample that map to one of the "
             "configured classes: global boxes + velocity (or 0 with `velocity_valid = False`), label index, name, "
             "LiDAR/radar point counts, attribute name, instance and annotation tokens, visibility; finally the boxes "
             "are transformed to the reference frame with `transform_boxes`."),
            ("def _lidar_pose", "`_lidar_pose(db, sample, channel)` returns the LIDAR_TOP key-frame sample_data and "
             "its ego pose — this pose *defines* the reference frame of the sample."),
            ("class CanBus", "`CanBus`: optional reader for the CAN-bus expansion (`<root>/can_bus/<scene>_pose.json`). "
             "If the folder exists, `ego_state()` returns speed, yaw rate and longitudinal acceleration of the last "
             "message before the timestamp. The CAN-bus expansion is **not present** in the current dataset folder, "
             "so the pose-based estimate below is used."),
            ("def _ego_state_from_poses", "`_ego_state_from_poses()` — causal ego-state estimation from the 20 Hz LiDAR "
             "ego poses: speed along the current heading and yaw rate from a backward difference over ≥ 0.2 s, "
             "acceleration from the speed change over ≥ 0.5 s; a forward difference is used only for the very first "
             "sweeps of a log. Being causal (no future poses) keeps the planner input honest."),
            ("def driving_command", "`driving_command(future_xy, mask, threshold)`: the navigation command used by the "
             "planners — LEFT if the last valid future point is ≥ 2 m to the left, RIGHT if ≤ −2 m, else STRAIGHT "
             "(ST-P3/UniAD open-loop protocol; stands in for a route planner)."),
            ("def build_sample_info", "`build_sample_info()`, part 1: looks up sample, scene, log and map; computes the "
             "reference pose; records the LiDAR key-frame plus up to `max_cached_lidar_sweeps` previous sweeps, the six "
             "camera key-frames and, per radar, the key-frame plus `max_cached_radar_sweeps − 1` previous sweeps."),
            ("    # Ground truth in the reference frame.", "Part 2: ground truth (`_boxes_for_sample`) and the bicycle racks "
             "(category `static_object.bicycle_rack`) needed for the official evaluation rule that ignores bicycles "
             "and motorcycles parked inside racks; then the ego state (CAN bus if available, else poses)."),
            ("    # Ego future / past trajectory", "Part 3: walks the next `planning.horizon` (6) key-frames: ego position "
             "and yaw in the current reference frame (the logged trajectory, `mask` False past the scene end) and "
             "all future agent boxes (also in the current reference frame) for the collision metric/loss; the "
             "command is derived from the future; then up to `history_frames` past ego positions."),
            ("    # Future trajectories of the current GT objects", "Part 4: for every current GT object, its future "
             "positions are looked up by instance token in the future agent lists → `future_xy (M, T, 2)` and "
             "`future_mask (M, T)` (drawn as 'logged object futures' in the dashboard)."),
            ("def build_infos", "`build_infos(db, scene_names, cfg)` loops over scenes and key-frames, adds `frame_idx` and "
             "`num_frames`. `_cache_key()` hashes every setting that affects the infos (format version, version, split, "
             "classes, sensors, sweep caps, horizon, command threshold) so a config change automatically produces a "
             "new cache file."),
            ("def load_infos", "`load_infos(cfg, split, db, rebuild)` returns the cached infos for a split "
             "(`outputs/cache/infos_<version>_<split>_<hash>.pkl`) or builds them (loading the tables only when "
             "needed) and writes the cache."),
        ],
    },
    {
        "file": "datasets/camera_loader.py",
        "intro": "Camera image loading with augmentation and the matrices the LSS encoder needs.",
        "blocks": [
            ('"""Camera loading', "Docstring: the meaning of `intrinsic`, `post_rot/post_tran` and `cam2ref`, and the "
             "lifting equation."),
            ("from __future__", "Imports and the ImageNet normalisation constants."),
            ("@dataclass", "`ImageAug`: the sampled augmentation of one image — resize factor, resized size, crop box, "
             "horizontal flip, rotation (degrees)."),
            ("def _rot2", "`_rot2(θ)`: 2×2 rotation matching `PIL.Image.rotate` (counter-clockwise on screen, image y "
             "axis pointing down)."),
            ("def image_aug_matrices", "`image_aug_matrices(aug)` computes `post_rot (3×3)` and `post_tran (3,)` so that "
             "`p_aug = post_rot·p + post_tran` for pixel coordinates (LSS convention): scale by the resize factor, "
             "subtract the crop offset, mirror around the crop width if flipped, rotate around the crop centre. "
             "Verified numerically (round trip error 6·10⁻¹⁴ m)."),
            ("def apply_image_aug", "`apply_image_aug(img, aug)` applies the same operations to the PIL image in the same "
             "order (resize → crop → flip → rotate)."),
            ("class CameraLoader", "`CameraLoader.__init__` stores the dataset root, camera list, final network input "
             "size (H, W) and the `dataset.camera` config."),
            ("    def sample_aug", "`sample_aug(W, H, train, rng)`: base resize = target width / original width. In training "
             "a random resize jitter, a random bottom-biased crop (keeps the road, drops sky), a random flip and a "
             "random rotation (±5.4°) are drawn; at test time the resize is the base, the crop is centred "
             "horizontally and placed at the mean bottom ratio, no flip/rotation."),
            ("    def load(self", "`load(info, train, rng)` → arrays for the six cameras: normalised images "
             "`(6, 3, H, W)`, intrinsics, post_rots, post_trans, cam2ref and `cam_valid`. A missing camera record or "
             "an unreadable file is logged once (`WarnOnce`) and leaves a zero image with `cam_valid = False`, which "
             "the camera encoder uses to ignore that camera."),
            ("def lidar_depth_maps", "`lidar_depth_maps(points_ref, cams, image_size, stride, depth_range)` → `(6, h, w)` "
             "depth targets: each LiDAR point is transformed into the camera (`inv(cam2ref)`), projected with K, "
             "moved into the augmented image with post_rot/post_tran, binned into feature cells (÷16), and the "
             "**closest** depth per cell is kept (points are written far-to-near). 0 means no supervision."),
        ],
    },
    {
        "file": "datasets/lidar_loader.py",
        "intro": "Multi-sweep LiDAR accumulation in the reference frame.",
        "blocks": [
            ('"""LiDAR loading', "Docstring with the processing chain; `LIDAR_POINT_FEATURES` = x, y, z, intensity, "
             "time_lag."),
            ("def read_lidar_bin", "`read_lidar_bin(path)` reads a nuScenes `.pcd.bin` (float32 × 5: x, y, z, intensity, "
             "ring index) and validates the size."),
            ("class LidarLoader", "Constructor: root, grid, number of sweeps (`dataset.lidar.sweeps`), optional "
             "`max_points`, `remove_close_radius` (1.5 m: drops returns from the ego vehicle itself)."),
            ("    def load_sweep", "`load_sweep(rec)`: read, remove close points (sensor frame), transform xyz with "
             "`sensor2ref` (motion-compensated through the sweep's own ego pose), keep intensity, set the time-lag "
             "column. Errors are logged once and return `None`."),
            ("    def load(self", "`load(info, num_sweeps, rng, filter_range)`: concatenates the first *n* sweeps "
             "(key-frame first), filters to the BEV x/y range and z range, and randomly subsamples above "
             "`max_points` (seeded rng → deterministic at evaluation). Returns `(P, 5)` or an empty `(0, 5)` array."),
            ("    def load_keyframe", "`load_keyframe(info)`: only the key-frame sweep (depth supervision, calibration "
             "checks, planner obstacle layer)."),
        ],
    },
    {
        "file": "datasets/radar_loader.py",
        "intro": "Radar loading: own binary-PCD parser, the devkit's default filters, multi-sweep accumulation, and "
                 "rotation of the velocity vectors into the reference frame.",
        "blocks": [
            ('"""Radar loading', "Docstring: why velocities must be rotated (the devkit's multisweep loader leaves them "
             "in the sensor frame). `RADAR_POINT_FEATURES` defines the 10 output columns; `_PCD_TYPES` maps PCD "
             "(TYPE, SIZE) pairs to NumPy dtypes."),
            ("def read_radar_pcd", "`read_radar_pcd(path)`: parses the ASCII header up to `DATA binary` (FIELDS, SIZE, "
             "TYPE, POINTS), builds a *packed* structured dtype (43 bytes per point for nuScenes radar) and reads "
             "exactly `POINTS` records with `np.frombuffer` (the files contain trailing padding). Raises "
             "`ValueError` for non-binary or truncated files."),
            ("class RadarLoader", "Constructor: the three filter lists (`None` disables a filter). Defaults from "
             "`configs/dataset.yaml` = devkit defaults: `invalid_state ∈ {0}`, `dyn_prop ∈ {0..6}`, `ambig_state ∈ {3}`."),
            ("    def _filter", "`_filter(pts)` keeps points whose states are in the allowed sets (`np.isin`)."),
            ("    def load_sweep", "`load_sweep(rec)` → `(N, 10)`: positions transformed with `sensor2ref`; RCS copied; "
             "`(vx_comp, vy_comp)` (ego-motion compensated) and raw `(vx, vy)` rotated with the 2×2 rotation part of "
             "`sensor2ref`; dynamic property; time-lag."),
            ("    def load(self", "`load(info, num_sweeps)`: loops over the 5 radars and their first *n* sweeps, concatenates, "
             "and filters to the BEV range."),
        ],
    },
    {
        "file": "datasets/nuscenes_dataset.py",
        "intro": "The PyTorch `Dataset` used for training, evaluation and inference, plus the batch collation.",
        "blocks": [
            ('"""PyTorch dataset', "Docstring: table of every key in a dataset item with its shape and meaning "
             "(reproduced in the *Tensor Dimension Reference* chapter)."),
            ("from __future__", "Imports and the key groups used by `collate_batch` (`POINT_KEYS` are concatenated "
             "with a batch-index column, `LIST_KEYS` stay Python lists)."),
            ("def enabled_modalities", "`enabled_modalities(cfg)` → `{camera, lidar, radar}: bool` from `fusion.use_*`."),
            ("class NuScenesFusionDataset", "Constructor: loads (or receives) the infos, stores modality flags, classes, "
             "the BEV grid and the head grid (`downsample(out_stride)`), creates the three loaders from the config, "
             "precomputes the label indices of the collision classes and optionally truncates the dataset "
             "(`train.max_train_samples` / `max_val_samples`) for quick runs."),
            ("    def _sample_bda", "`_sample_bda(rng)`: identity unless training with augmentation enabled; otherwise a "
             "random rotation, scale and flips → `bev_rotation_matrix`."),
            ("    @staticmethod", "`_apply_bda_points(points, bda, vel_cols)`: applies the augmentation to xyz and to the "
             "listed velocity column pairs (radar `vx_comp, vy_comp` and `vx, vy`)."),
            ("    def _gt_mask", "`_gt_mask(gt)`: keeps GT boxes with ≥ `gt_filter.min_points` LiDAR+radar returns "
             "(nuScenes evaluation rule) and centres inside the grid."),
            ("    def __getitem__", "`__getitem__(idx)`, part 1 — sensors: a per-item RNG (random in training, seeded by "
             "the index otherwise), the augmentation matrix, LiDAR (if used or needed for display), cameras with "
             "depth targets from the key-frame LiDAR, radar; augmentation is applied to point clouds. "
             "`load_all_sensors` (dashboard) also returns the raw clouds for drawing."),
            ("        # Simulated sensor failure", "Part 2 — optional modality dropout (training only): empties a "
             "point cloud or invalidates all cameras with the configured probabilities (0 by default)."),
            ("        gt = info[\"gt\"]", "Part 3 — targets: GT boxes filtered and augmented; ego future transformed by "
             "`bda`, command recomputed from the augmented future, ego state (yaw rate negated when mirrored); "
             "CenterPoint targets on the head grid, the class-agnostic foreground heatmap on the BEV grid, and the "
             "future occupancy `(T, ny, nx)` of collision-class agents; `meta` carries token, scene, frame index."),
            ("def collate_batch", "`collate_batch(samples)`: point clouds are concatenated with a leading batch-index "
             "column (`(Σ P_b, 1 + F)`), `LIST_KEYS` stay lists (variable-length GT), all other arrays are stacked "
             "with a new batch dimension; `batch_size` is added."),
        ],
    },
]
