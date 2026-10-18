"""Device/seed/thread plumbing for direct training (WP-INFRA1, §11.5).

``resolve_device`` fails loudly when CUDA is required but unavailable
(no silent CPU fallback). ``seed_everything`` covers python/numpy/torch;
``set_cpu_threads`` sets intra/inter-op threads per process (the learner
starts at 2/1 per the infra plan, never the torch defaults of 16/32).
Batch movement pins nothing here — the loader owns pinning policy.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Final, Mapping

import numpy as np

from pes2ts_core.pesn.models import require_torch

DEVICE_SCHEMA_VERSION: Final[str] = "pes2ts_direct_device_v1"


class DeviceResolutionError(RuntimeError):
    """Typed refusal when a required device is unavailable."""


def resolve_device(*, device: str = "cuda:0", require_cuda: bool = True):
    """Return ``(torch, torch.device)``; explicit failure when CUDA is missing."""
    torch = require_torch()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            if require_cuda:
                raise DeviceResolutionError(
                    f"device {device!r} requested with require_cuda=True but "
                    "torch.cuda.is_available() is False; refusing CPU fallback"
                )
            return torch, torch.device("cpu")
        if torch.cuda.device_count() <= int(device.split(":")[-1] or 0):
            raise DeviceResolutionError(f"device index out of range: {device}")
    return torch, torch.device(device)


def set_cpu_threads(*, intra: int = 2, inter: int = 1) -> None:
    torch = require_torch()
    torch.set_num_threads(int(intra))
    try:
        torch.set_num_interop_threads(int(inter))
    except RuntimeError:
        pass  # already initialized in this process; the setting is process-wide


def seed_everything(seed: int) -> dict:
    """Seed python/numpy/torch (CPU) and return the seeds used."""
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {"seed": int(seed), "rng": ["python", "numpy", "torch_cpu"]}


def move_batch(batch: Mapping, device) -> dict:
    """Recursively move tensor dict leaves to ``device`` (non-tensors pass)."""
    out = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device, non_blocking=False)
        elif isinstance(value, dict):
            out[key] = move_batch(value, device)
        else:
            out[key] = value
    return out


def module_summary(module) -> dict:
    """Device/parameter summary used in run records (P2 evidence)."""
    torch = require_torch()
    params = list(module.parameters())
    devices = sorted({str(p.device) for p in params})
    return {
        "n_parameters": int(sum(p.numel() for p in params)),
        "parameter_devices": devices,
        "trainable": int(sum(p.numel() for p in params if p.requires_grad)),
        "torch": torch.__version__,
    }


__all__ = [
    "DEVICE_SCHEMA_VERSION",
    "DeviceResolutionError",
    "module_summary",
    "move_batch",
    "resolve_device",
    "seed_everything",
    "set_cpu_threads",
]
