"""Low-level training utilities shared by the trainer, callbacks and the runner.

This module deliberately imports only the standard library, NumPy and Torch so
that it can be used by :mod:`pes2ts_v5.training.callbacks` without creating an
import cycle through :mod:`pes2ts_v5.training` itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Determinism / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA when available)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str | None = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``cuda:0`` into a :class:`torch.device`."""
    if name is None or str(name).strip().lower() in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name).strip())


# ---------------------------------------------------------------------------
# Identity / hashing
# ---------------------------------------------------------------------------

def weights_sha256(model: torch.nn.Module) -> str:
    """SHA256 over ``name -> tensor bytes`` of a model state dict (stable)."""
    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(param.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def param_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _git(args: list[str], repo_root: str | Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(repo_root), capture_output=True, text=True,
            timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_commit_sha(repo_root: str | Path) -> str:
    """Current ``HEAD`` sha, or ``""`` when git/repository is unavailable."""
    return _git(["rev-parse", "HEAD"], repo_root) or ""


def git_is_dirty(repo_root: str | Path) -> bool | None:
    """``True``/``False`` when git reports status, ``None`` when unavailable."""
    out = _git(["status", "--porcelain"], repo_root)
    if out is None:
        return None
    return bool(out)


# ---------------------------------------------------------------------------
# Time / runtime provenance
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


# ---------------------------------------------------------------------------
# JSON writing (strict: NaN/Inf are mapped to null, never emitted)
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Recursively convert ``value`` into strict-JSON-safe primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def json_dumps(doc: Any) -> str:
    """Canonical, strict JSON text (sorted keys, no NaN/Inf)."""
    return json.dumps(json_safe(doc), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def write_json(path: str | Path, doc: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(doc), ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    return target


def append_jsonl(path: str | Path, row: Any) -> Path:
    """Append one strict-JSON line; creates parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json_dumps(row) + "\n")
    return target


# ---------------------------------------------------------------------------
# Memory observability (graceful on CPU / when psutil is absent)
# ---------------------------------------------------------------------------

def peak_memory_mb(device: str | torch.device | None = None) -> float | None:
    """Peak memory in MiB for ``device``; ``None`` when not observable.

    * CUDA: ``torch.cuda.max_memory_allocated`` (per-device peak).
    * CPU/other: process RSS peak via psutil, or ``None`` without psutil.
    """
    try:
        dev = device if isinstance(device, torch.device) else torch.device(
            str(device or "cpu"))
    except (RuntimeError, ValueError):
        dev = torch.device("cpu")
    if dev.type == "cuda":  # pragma: no cover - CPU-only CI
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2))
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is present in CI
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except (OSError, psutil.Error):  # pragma: no cover - defensive
        return None


def device_total_mb(device: str | torch.device | None = None) -> float | None:
    """Total device memory in MiB when known (CUDA); ``None`` on CPU."""
    try:
        dev = device if isinstance(device, torch.device) else torch.device(
            str(device or "cpu"))
    except (RuntimeError, ValueError):
        return None
    if dev.type != "cuda" or not torch.cuda.is_available():  # pragma: no cover
        return None
    return float(torch.cuda.get_device_properties(dev).total_memory / (1024 ** 2))
