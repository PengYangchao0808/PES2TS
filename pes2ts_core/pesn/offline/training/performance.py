"""Performance measurement utilities (WP-INFRA1, plan v3 §11.7).

CUDA-correct step timing (explicit synchronization around the measured
region) plus a light resource sampler that records CPU/RSS/GPU/disk lines
at a fixed interval to JSONL. Unavailable readings are recorded as null
with a reason — never skipped silently (§11.7 telemetry rule).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PERFORMANCE_SCHEMA_VERSION: Final[str] = "pes2ts_direct_performance_v1"


@dataclass
class StepTimer:
    """Context manager measuring synchronized wall time per step."""

    sync_cuda: bool = True
    samples_ms: list[float] = field(default_factory=list)
    _t0: float = 0.0

    def _sync(self) -> None:
        if not self.sync_cuda:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def __enter__(self) -> "StepTimer":
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._sync()
        self.samples_ms.append((time.perf_counter() - self._t0) * 1e3)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {"n": 0}
        ordered = sorted(self.samples_ms)

        def pct(p: float) -> float:
            idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
            return round(ordered[idx], 3)

        return {
            "n": len(ordered),
            "mean_ms": round(sum(ordered) / len(ordered), 3),
            "p50_ms": pct(0.50), "p95_ms": pct(0.95), "max_ms": round(ordered[-1], 3),
        }


def _read_proc_status() -> dict:
    try:
        text = Path("/proc/self/status").read_text()
    except OSError as exc:
        return {"rss_kib": None, "rss_reason": f"{type(exc).__name__}: {exc}"}
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return {"rss_kib": int(line.split()[1])}
    return {"rss_kib": None, "rss_reason": "VmRSS not found"}


def _read_gpu() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_allocated_mib": None, "gpu_reason": "cuda unavailable"}
        return {
            "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        }
    except Exception as exc:
        return {"gpu_allocated_mib": None, "gpu_reason": f"{type(exc).__name__}: {exc}"}


def sample_resources(output_path: str | Path, *, interval_seconds: float = 2.0,
                     duration_seconds: float = 60.0) -> int:
    """Append resource samples as JSONL; returns the number of samples."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    end = time.monotonic() + max(0.0, duration_seconds)
    while True:
        row = {
            "schema": PERFORMANCE_SCHEMA_VERSION,
            "t": round(time.monotonic(), 3),
            "pid": os.getpid(),
        }
        row.update(_read_proc_status())
        row.update(_read_gpu())
        try:
            import shutil
            usage = shutil.disk_usage(os.environ.get("PES2TS_DISK_ROOT", "."))
            row.update({"disk_free_gib": round(usage.free / 2**30, 3)})
        except OSError as exc:
            row.update({"disk_free_gib": None, "disk_reason": str(exc)})
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        n += 1
        if time.monotonic() >= end:
            break
        time.sleep(interval_seconds)
    return n


__all__ = [
    "PERFORMANCE_SCHEMA_VERSION",
    "StepTimer",
    "sample_resources",
]
