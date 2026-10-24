"""Pinned-buffer H2D staging for the V5.3-MP training loop (plan §6 / I3).

``to_torch_batch`` (the sha-bound ``key_atom_graph_v3``, NEVER edited)
moves each collated numpy array to the device with ``torch.as_tensor(arr,
device=...)`` — a PAGEABLE host-to-device copy: the driver stages through
an internal pinned bounce buffer, synchronously.  :func:
`to_torch_batch_pinned` mirrors the exact output contract (same keys,
dtypes, shapes, values; non-array values pass through untouched) but
stages each array through a REUSABLE PINNED CPU tensor from
:class:`PinnedBufferPool` and issues a ``non_blocking=True`` device copy
followed by an CUDA event sync BEFORE returning — correctness over
async: when the function returns, every tensor is fully materialised on
the device and the pinned buffer may be recycled on the next batch.

The pool is keyed by ``(field name, shape, dtype)`` and capped by total
bytes (LRU eviction; default 256 MB).  A shape miss (the common case for
node/edge fields whose batch totals vary) allocates a fresh pinned tensor
and caches it subject to the cap.  ANY error (pinning unavailable, CUDA
absent, allocation failure) logs ONE stderr line and falls back to the
direct ``to_torch_batch`` path — a pinned-buffer bug costs performance
only, never a run and never numerics (the staged copy is a plain
elementwise copy; no dtype or precision handling differs).

This is the "B3 里实测一次" row of plan §6: the honest win on this loop is
the pinned DMA bandwidth for the larger tensors (node positions, edge
attrs) plus skipping the pageable bounce; the micro-benchmark in
``tests/pesn/test_h2d_pinned_v53mp.py`` measures it (informational, never
a gate).

Torch-only module (mirrors the trainer's dependency); numpy for the
array contract.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from pes2ts_core.pesn.features.key_atom_graph_v3 import to_torch_batch

__all__ = [
    "DEFAULT_PINNED_CAPACITY_BYTES",
    "PinnedBufferPool",
    "to_torch_batch_pinned",
]

#: Default cap for the pinned buffer pool (plan §6: the loop moves
#: ~32-sample batches of ~34 KB features; a few shape variants dominate).
DEFAULT_PINNED_CAPACITY_BYTES = 256 << 20

_LOGGED_ISSUES: set = set()


def log_h2d_issue(message: str) -> None:
    """ONE stderr line per distinct h2d problem (never raises)."""
    try:
        if message in _LOGGED_ISSUES:
            return
        _LOGGED_ISSUES.add(message)
        print("[v53mp pinned h2d] {}".format(message), file=sys.stderr,
              flush=True)
    except Exception:                           # pragma: no cover - stderr gone
        pass


def _torch_dtype_of(arr: np.ndarray) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=arr.dtype)).dtype


class PinnedBufferPool:
    """Reusable pinned CPU tensors keyed by ``(field, shape, dtype)``.

    LRU-capped by TOTAL BYTES (default 256 MB): an acquire that would
    exceed the cap evicts the least-recently-used buffer; a single tensor
    larger than the whole cap is allocated but never cached.  Thread-safe
    (one lock); the intended caller is the single training thread.
    ``acquire`` returns a buffer whose CONTENTS ARE UNDEFINED — the caller
    fully overwrites it before use.
    """

    def __init__(self, capacity_bytes: int = DEFAULT_PINNED_CAPACITY_BYTES) -> None:
        self._capacity = max(int(capacity_bytes), 1)
        self._lock = threading.Lock()
        self._bufs: "OrderedDict[Tuple[str, Tuple[int, ...], torch.dtype], torch.Tensor]" = \
            OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return int(self._bytes)

    @property
    def n_buffers(self) -> int:
        with self._lock:
            return len(self._bufs)

    def acquire(self, field: str, shape: Tuple[int, ...],
                dtype: torch.dtype) -> torch.Tensor:
        """A pinned tensor of ``shape``/``dtype`` (contents undefined)."""
        key = (str(field), tuple(int(s) for s in shape), dtype)
        with self._lock:
            buf = self._bufs.pop(key, None)
            if buf is not None:
                self._bufs[key] = buf        # re-insert as most-recent
                self.hits += 1
                return buf
            self.misses += 1
        buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        nbytes = int(buf.numel()) * int(buf.element_size())
        if nbytes > self._capacity:
            return buf                        # too big to pool: unpooled use
        with self._lock:
            self._bufs[key] = buf
            self._bytes += nbytes
            while self._bytes > self._capacity and self._bufs:
                _k, old = self._bufs.popitem(last=False)
                self._bytes -= int(old.numel()) * int(old.element_size())
        return buf

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": int(self.hits),
                "misses": int(self.misses),
                "used_bytes": int(self._bytes),
                "capacity_bytes": int(self._capacity),
                "n_buffers": len(self._bufs),
            }


def to_torch_batch_pinned(
    batch: Mapping[str, Any],
    device: torch.device | str = "cpu",
    pool: Optional[PinnedBufferPool] = None,
) -> Dict[str, Any]:
    """Device batch via pooled pinned staging — SAME output as ``to_torch_batch``.

    Contract mirror: every ``np.ndarray`` becomes a contiguous device
    tensor with identical dtype/shape/values; every non-array value is
    passed through unchanged.  For non-CUDA devices, or when ``pool`` is
    None, this IS ``to_torch_batch``.  On the CUDA path each array is
    copied numpy → pooled pinned buffer → ``non_blocking`` device copy,
    and an CUDA event recorded after all copies is synchronised BEFORE
    returning (correctness over async: buffers are recycled next call, so
    the DMA must have completed).  ANY error logs one line and falls back
    to the direct path for the whole batch.
    """
    if pool is None or not str(device).startswith("cuda") \
            or not torch.cuda.is_available():
        return to_torch_batch(batch, device=device)
    try:
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                arr = np.ascontiguousarray(value)
                pinned = pool.acquire(key, arr.shape, _torch_dtype_of(arr))
                pinned.copy_(torch.from_numpy(arr))
                dev_t = torch.empty(arr.shape, dtype=pinned.dtype,
                                    device=device)
                dev_t.copy_(pinned, non_blocking=True)
                out[key] = dev_t
            else:
                out[key] = value
        # correctness barrier: all non_blocking DMAs complete before the
        # pinned buffers are recycled on the next batch
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        return out
    except Exception as exc:                    # noqa: BLE001 - fail-open
        log_h2d_issue(
            "pinned staging failed ({}: {}); falling back to direct "
            "to_torch_batch".format(type(exc).__name__, exc))
        return to_torch_batch(batch, device=device)
