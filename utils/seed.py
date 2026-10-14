"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    ``deterministic=True`` additionally requests deterministic cuDNN kernels (slower).
    ``index_add_``/``scatter_reduce_`` on CUDA use atomics and are not bit-exact even then.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible NumPy/random stream."""
    base = torch.initial_seed() % 2**31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
