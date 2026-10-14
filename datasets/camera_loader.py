"""Camera loading, image-space augmentation and calibration matrices.

For every camera the loader returns the matrices that the Lift-Splat camera BEV
encoder needs to place image features in the reference BEV frame:

* ``intrinsic`` ``K`` (3x3) of the original 1600x900 image,
* ``post_rot`` / ``post_tran``: the affine image augmentation (resize, crop, flip,
  rotate) so that ``p_aug = post_rot @ p_orig + post_tran`` in pixel space,
* ``cam2ref`` (4x4): camera -> ego(t_cam) -> global -> ego(t_ref), i.e. the calibrated
  extrinsics *and* the ego-motion between the camera and LiDAR capture times.

A pixel ``(u, v)`` with depth ``d`` in the augmented image maps to the reference frame as
``cam2ref · K⁻¹ · [d·u', d·v', d]`` with ``(u', v') = post_rot⁻¹ ((u, v) − post_tran)``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from utils.geometry import invert_transform
from utils.logging_utils import WarnOnce

logger = logging.getLogger(__name__)
_warn = WarnOnce(logger)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ImageAug:
    resize: float
    resize_dims: tuple[int, int]          # (W, H) after resizing
    crop: tuple[int, int, int, int]       # (x0, y0, x1, y1) in the resized image
    flip: bool
    rotate: float                         # degrees, counter-clockwise


def _rot2(theta: float) -> np.ndarray:
    # Image-space rotation matching PIL.Image.rotate (counter-clockwise on screen, y down).
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


def image_aug_matrices(aug: ImageAug) -> tuple[np.ndarray, np.ndarray]:
    """3x3 ``post_rot`` and 3-vector ``post_tran`` of an :class:`ImageAug` (LSS convention)."""
    rot = np.eye(2) * aug.resize
    tran = -np.asarray(aug.crop[:2], dtype=np.float64)
    width, height = aug.crop[2] - aug.crop[0], aug.crop[3] - aug.crop[1]
    if aug.flip:
        a = np.array([[-1.0, 0.0], [0.0, 1.0]])
        rot, tran = a @ rot, a @ tran + np.array([width, 0.0])
    a = _rot2(np.deg2rad(aug.rotate))
    centre = np.array([width, height]) / 2.0
    b = a @ -centre + centre
    rot, tran = a @ rot, a @ tran + b
    post_rot, post_tran = np.eye(3), np.zeros(3)
    post_rot[:2, :2], post_tran[:2] = rot, tran
    return post_rot, post_tran


def apply_image_aug(img: Image.Image, aug: ImageAug) -> Image.Image:
    img = img.resize(aug.resize_dims, Image.BILINEAR).crop(aug.crop)
    if aug.flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if aug.rotate:
        img = img.rotate(aug.rotate, resample=Image.BILINEAR)
    return img


class CameraLoader:
    """Load the surround-view images of one sample with consistent augmentation matrices."""

    def __init__(self, root: Path, cameras: list[str], image_size: tuple[int, int], camera_cfg: Any) -> None:
        self.root = Path(root)
        self.cameras = list(cameras)
        self.final_h, self.final_w = int(image_size[0]), int(image_size[1])
        self.cfg = camera_cfg

    def sample_aug(self, orig_w: int, orig_h: int, train: bool, rng: np.random.Generator) -> ImageAug:
        fh, fw = self.final_h, self.final_w
        base = fw / orig_w
        bot = self.cfg.bottom_crop_jitter
        if train:
            resize = base * rng.uniform(*self.cfg.resize_jitter)
            dims = (int(orig_w * resize), int(orig_h * resize))
            crop_h = int((1 - rng.uniform(*bot)) * dims[1]) - fh
            crop_w = int(rng.uniform(0, max(0, dims[0] - fw)))
            flip = bool(self.cfg.flip) and rng.random() < 0.5
            rotate = float(rng.uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg))
        else:
            resize = base
            dims = (int(orig_w * resize), int(orig_h * resize))
            crop_h = int((1 - float(np.mean(bot))) * dims[1]) - fh
            crop_w = int(max(0, dims[0] - fw) / 2)
            flip, rotate = False, 0.0
        crop_h = max(crop_h, 0)
        return ImageAug(resize, dims, (crop_w, crop_h, crop_w + fw, crop_h + fh), flip, rotate)

    def load(self, info: dict[str, Any], train: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n = len(self.cameras)
        imgs = np.zeros((n, 3, self.final_h, self.final_w), dtype=np.float32)
        intrinsics = np.tile(np.eye(3), (n, 1, 1))
        post_rots = np.tile(np.eye(3), (n, 1, 1))
        post_trans = np.zeros((n, 3))
        cam2ref = np.tile(np.eye(4), (n, 1, 1))
        valid = np.zeros(n, dtype=bool)
        for i, cam in enumerate(self.cameras):
            rec = info["cams"].get(cam)
            if rec is None:
                _warn(f"{info['token']}:{cam}", f"Camera {cam} missing in sample {info['token']}")
                continue
            intrinsics[i] = rec["intrinsic"]
            cam2ref[i] = rec["sensor2ref"]
            path = self.root / rec["path"]
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")
            except OSError as exc:
                _warn(str(path), f"Unreadable image {path}: {exc} (camera disabled for this sample)")
                aug = self.sample_aug(rec["width"], rec["height"], False, rng)
                post_rots[i], post_trans[i] = image_aug_matrices(aug)
                continue
            aug = self.sample_aug(img.width, img.height, train, rng)
            img = apply_image_aug(img, aug)
            post_rots[i], post_trans[i] = image_aug_matrices(aug)
            arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            imgs[i] = arr.transpose(2, 0, 1)
            valid[i] = True
        return {
            "images": imgs,
            "intrinsics": intrinsics.astype(np.float32),
            "post_rots": post_rots.astype(np.float32),
            "post_trans": post_trans.astype(np.float32),
            "cam2ref": cam2ref.astype(np.float32),
            "cam_valid": valid,
        }


def lidar_depth_maps(points_ref: np.ndarray, cams: dict[str, np.ndarray], image_size: tuple[int, int],
                     stride: int, depth_range: tuple[float, float]) -> np.ndarray:
    """Sparse LiDAR depth per camera at feature resolution ``(N, H/stride, W/stride)``.

    Each feature cell keeps the *closest* projected LiDAR point (0 = no supervision),
    exactly the BEVDepth depth-supervision target.
    """
    h, w = image_size[0] // stride, image_size[1] // stride
    n = cams["images"].shape[0]
    depth = np.zeros((n, h, w), dtype=np.float32)
    if points_ref.shape[0] == 0:
        return depth
    xyz1 = np.concatenate([points_ref[:, :3], np.ones((points_ref.shape[0], 1), np.float32)], 1).astype(np.float64)
    for i in range(n):
        if not cams["cam_valid"][i]:
            continue
        pc = (xyz1 @ invert_transform(cams["cam2ref"][i].astype(np.float64)).T)[:, :3]
        d = pc[:, 2]
        keep = (d > depth_range[0]) & (d < depth_range[1])
        pc, d = pc[keep], d[keep]
        uv = (pc @ cams["intrinsics"][i].T)[:, :2] / d[:, None]
        uv = uv @ cams["post_rots"][i][:2, :2].T + cams["post_trans"][i][:2]
        col, row = np.floor(uv[:, 0] / stride).astype(np.int64), np.floor(uv[:, 1] / stride).astype(np.int64)
        ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        col, row, d = col[ok], row[ok], d[ok]
        order = np.argsort(-d)  # far first so the nearest point is written last
        depth[i, row[order], col[order]] = d[order]
    return depth
