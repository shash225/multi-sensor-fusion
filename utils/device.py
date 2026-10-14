"""Device selection and GPU memory reporting."""
from __future__ import annotations

import logging
from typing import Any

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Use CUDA when available (or requested) and fall back to CPU gracefully."""
    pref = (preference or "auto").lower()
    if pref.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


_FP16_OK: dict[str, bool] = {}


def fp16_is_reliable(device: torch.device) -> bool:
    """Numerically test fp16 convolutions on this GPU (cached).

    Some GPU / cuDNN combinations return NaN for fp16 convolutions: on a GTX 1650 with
    cuDNN 9.10 every 3x3 conv with 256 input channels does (measured in this project,
    see README "Hardware notes"). The test runs exactly that layer type in fp16 and fp32.
    """
    key = str(device)
    if key not in _FP16_OK:
        g = torch.Generator(device="cpu").manual_seed(0)
        conv = torch.nn.Conv2d(256, 256, 3, padding=1).to(device)
        ok = True
        # Shapes that occur in this model (image backbone layer3, BEV backbone block2, head input);
        # cuDNN picks different kernels per shape, so all of them are probed.
        for shape in ((6, 256, 8, 22), (1, 256, 50, 50), (1, 256, 100, 100)):
            x = torch.randn(*shape, generator=g).to(device)
            with torch.no_grad():
                ref = conv(x)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = conv(x).float()
            ok &= bool(torch.isfinite(y).all()) and float((y - ref).abs().max() / ref.abs().max()) < 0.02
        _FP16_OK[key] = ok
        if not ok:
            logging.getLogger(__name__).warning(
                "fp16 convolutions are numerically broken on %s (cuDNN %s); mixed precision disabled, using fp32.",
                torch.cuda.get_device_name(device), torch.backends.cudnn.version())
    return _FP16_OK[key]


def amp_enabled(device: torch.device, requested: bool) -> bool:
    """Use fp16 mixed precision only on CUDA devices where fp16 convolutions are verified correct."""
    return bool(requested) and device.type == "cuda" and fp16_is_reliable(device)


def gpu_memory_report(device: torch.device) -> dict[str, Any]:
    """Current / peak allocated memory in MiB (empty dict on CPU)."""
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
        "total_mb": torch.cuda.get_device_properties(device).total_memory / 2**20,
        "name": torch.cuda.get_device_name(device),
    }


def move_to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Recursively move tensors inside dicts/lists/tuples to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device, non_blocking) for v in obj)
    return obj
