"""nuScenes raster map prior (``maps/*.png``, category ``semantic_prior``).

The original masks are 0.1 m/pixel and up to 32k x 37k pixels (>1 GB decoded), so they
are decoded once, reduced 5x to 0.5 m/pixel and cached as ``.npy``
(``python scripts/preprocess.py --maps`` or lazily on first use).

Global -> map pixel (devkit ``MapMask``): ``col = x / res``, ``row = H - y / res``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
NATIVE_RES = 0.1


class MapPrior:
    def __init__(self, root: Path, cache_dir: Path, reduce: int = 5) -> None:
        self.root, self.cache_dir, self.reduce = Path(root), Path(cache_dir) / "maps", int(reduce)
        self.res = NATIVE_RES * self.reduce
        self._maps: dict[str, np.ndarray] = {}

    def _cache_path(self, filename: str) -> Path:
        return self.cache_dir / f"{Path(filename).stem}_r{self.reduce}.npy"

    def load(self, filename: str) -> Optional[np.ndarray]:
        if filename in self._maps:
            return self._maps[filename]
        cache = self._cache_path(filename)
        if cache.exists():
            arr = np.load(cache)
        else:
            src = self.root / filename
            if not src.exists():
                logger.warning("Map file %s not found", src)
                return None
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None  # the official masks exceed PIL's decompression-bomb limit
            logger.info("Decoding map %s once (large file, ~10-30 s) ...", src.name)
            with Image.open(src) as im:
                small = im.reduce(self.reduce)
                arr = (np.asarray(small) > 127).astype(np.uint8) * 255
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)
        self._maps[filename] = arr
        return arr

    def crop_to_ego(self, filename: str, ego2global: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float],
                    ppm: float) -> Optional[np.ndarray]:
        """Map patch in *display* layout (forward up, left left) at ``ppm`` pixels per metre (uint8 0/255)."""
        arr = self.load(filename)
        if arr is None:
            return None
        h_map = arr.shape[0]
        h = int(round((x_range[1] - x_range[0]) * ppm))
        w = int(round((y_range[1] - y_range[0]) * ppm))
        # display pixel (c, r) -> ego (x, y): x = x_max - r/ppm, y = y_max - c/ppm
        disp_to_ego = np.array([[0.0, -1.0 / ppm, x_range[1]], [-1.0 / ppm, 0.0, y_range[1]], [0.0, 0.0, 1.0]])
        ego_to_global = np.array([[ego2global[0, 0], ego2global[0, 1], ego2global[0, 3]],
                                  [ego2global[1, 0], ego2global[1, 1], ego2global[1, 3]], [0.0, 0.0, 1.0]])
        global_to_map = np.array([[1.0 / self.res, 0.0, 0.0], [0.0, -1.0 / self.res, h_map], [0.0, 0.0, 1.0]])
        mat = (global_to_map @ ego_to_global @ disp_to_ego)[:2]  # display pixel -> map pixel (dst -> src)
        return cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def drivable_on_grid(self, filename: str, ego2global: np.ndarray, grid) -> Optional[np.ndarray]:
        """Map prior on a model-layout grid ``(ny, nx)`` (bool), e.g. for the planner's map cost."""
        ppm = 1.0 / grid.resolution
        disp = self.crop_to_ego(filename, ego2global, grid.x_range, grid.y_range, ppm)
        if disp is None:
            return None
        # display -> model layout (inverse of visualization.common.bev_to_display)
        return (np.ascontiguousarray(np.swapaxes(disp, 0, 1)[::-1, ::-1]) > 0)
