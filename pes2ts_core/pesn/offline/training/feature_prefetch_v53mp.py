"""Optional fork-prefetch of ``build_sample_features_v3`` (V5.3-MP perf lane).

Feature building (``pes2ts_core/pesn/features/key_atom_graph_v3.py``,
sha-bound, NEVER modified here) is ~60% of the per-update wall clock of a
V5.3-MP run and is a pure function of ``KeyAtomSampleV3``.  This module
runs it across a small ``fork`` process pool so a 32-sample training batch
builds in parallel.

Design constraints (S3 campaign discipline):

* **opt-in via flag file** — ``outputs/pesn/v53mp/reports/feature_workers.on``
  containing an int in [2, 12] enables the pool; anything else (missing
  file, bad value) keeps the trainer on today's serial path with zero
  behavioural change.  The gate is read ONCE at trainer start.
* **fork BEFORE any CUDA init** — the trainer constructs the pool at the
  very top of ``train_run_v53mp`` (before ``load_model_v53mp``), so the
  children never inherit a CUDA context; they only ever run numpy code.
* **fail-open** — every prefetch call is wrapped; the FIRST failure
  (pickling error, worker death, broken pool, timeout) logs one line to
  stderr, closes the pool, and the run continues on the serial path.  A
  prefetch bug costs performance only, never a run.
* **bit-identical numerics** — the same ``build_sample_features_v3`` is
  called with the same arguments (level/intent_off are pass-throughs); no
  dtype or precision handling differs from the serial path.
"""

from __future__ import annotations

import multiprocessing
import sys
from functools import partial
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from pes2ts_core.pesn.features.key_atom_graph_v3 import (
    KeyAtomSampleV3,
    build_sample_features_v3,
)

__all__ = [
    "FEATURE_WORKERS_FLAG",
    "FEATURE_WORKERS_MAX",
    "FEATURE_WORKERS_MIN",
    "FeaturePrefetcher",
    "maybe_create_prefetcher",
    "read_feature_workers",
    "split_contiguous",
]

#: Activation gate (repo-relative; read from the trainer's CWD).  Default
#: OFF: the file does not exist on the campaign host, so every run today
#: takes the serial path.
FEATURE_WORKERS_FLAG = Path("outputs/pesn/v53mp/reports/feature_workers.on")
FEATURE_WORKERS_MIN = 2
FEATURE_WORKERS_MAX = 12

#: Safety net: one map() call beyond this is treated as a hung pool and
#: degrades to serial.  A healthy call is ~0.1 s at batch scale; this only
#: ever trips on a wedged worker, and tripping must NEVER hang the run.
_PREFETCH_TIMEOUT_S = 120.0


def log_prefetch_issue(message: str) -> None:
    """ONE stderr line per prefetch problem (never raises)."""
    try:
        print("[v53mp feature prefetch] {}".format(message),
              file=sys.stderr, flush=True)
    except Exception:                       # pragma: no cover - stderr gone
        pass


def read_feature_workers(flag_path: Optional[Path] = None) -> int:
    """Effective worker count from the flag file; 0 = disabled (serial).

    Returns the file's int iff it lies in [FEATURE_WORKERS_MIN,
    FEATURE_WORKERS_MAX]; a missing file or any unparsable/out-of-range
    content yields 0 (serial path, no behavioural change).
    """
    path = Path(flag_path) if flag_path is not None else FEATURE_WORKERS_FLAG
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return 0
    try:
        workers = int(text)
    except ValueError:
        return 0
    if FEATURE_WORKERS_MIN <= workers <= FEATURE_WORKERS_MAX:
        return workers
    return 0


def split_contiguous(items: Sequence[Any], n_chunks: int) -> List[List[Any]]:
    """Split ``items`` into <= ``n_chunks`` CONTIGUOUS, order-preserving slices.

    Sizes differ by at most one; empty inputs yield no chunks; the
    concatenation of the result equals ``list(items)`` exactly.
    """
    n = len(items)
    if n == 0:
        return []
    n_chunks = max(1, min(int(n_chunks), n))
    base, remainder = divmod(n, n_chunks)
    chunks: List[List[Any]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(list(items[start:start + size]))
        start += size
    return chunks


def _worker_build_chunk(
    samples: Sequence[KeyAtomSampleV3],
    level: str,
    intent_off: bool,
) -> List[dict]:
    """Pool worker (module-level, picklable): serially build one chunk.

    Calls the SAME ``build_sample_features_v3`` the serial path calls, with
    the same ``level``/``intent_off`` — numerics are bit-identical.
    """
    return [build_sample_features_v3(sample, level=level,
                                     intent_off=intent_off)
            for sample in samples]


class FeaturePrefetcher:
    """Fork pool that builds ``build_sample_features_v3`` dicts in parallel.

    Construction forks immediately (``multiprocessing.get_context("fork")``)
    — callers MUST construct it before any CUDA initialisation.  Use as a
    context manager or call :meth:`close`/:meth:`terminate` explicitly; both
    are idempotent and never leave orphans.  After ANY failure the instance
    flips to inactive permanently and every later call degrades to the
    caller's serial path (returning ``None`` from :meth:`build_many`).
    """

    def __init__(
        self,
        workers: int,
        level: str,
        intent_off: bool,
        *,
        timeout_s: float = _PREFETCH_TIMEOUT_S,
    ) -> None:
        workers = int(workers)
        if not (FEATURE_WORKERS_MIN <= workers <= FEATURE_WORKERS_MAX):
            raise ValueError(
                "workers must lie in [{}, {}], got {}".format(
                    FEATURE_WORKERS_MIN, FEATURE_WORKERS_MAX, workers))
        self.workers = workers
        self.level = str(level)
        self.intent_off = bool(intent_off)
        self._timeout_s = float(timeout_s)
        self._closed = False
        self._broken = False
        ctx = multiprocessing.get_context("fork")
        self._pool = ctx.Pool(workers)

    @property
    def active(self) -> bool:
        """True while the pool is usable (never closed, never broken)."""
        return self._pool is not None and not self._closed and not self._broken

    def build_many(
        self, samples: Sequence[KeyAtomSampleV3],
    ) -> Optional[List[dict]]:
        """Build features for ``samples`` (input order preserved).

        Returns ``None`` when the pool is unavailable or this call (or any
        earlier call) failed — the caller MUST then fall back to building
        serially.  ``samples`` is split into ``self.workers`` contiguous
        chunks (the measured sweet spot: 8 workers, ~4-sample chunks).
        """
        if not self.active:
            return None
        if not samples:
            return []
        chunks = split_contiguous(samples, self.workers)
        worker = partial(_worker_build_chunk,
                         level=self.level, intent_off=self.intent_off)
        try:
            parts = self._pool.map_async(worker, chunks).get(
                timeout=self._timeout_s)
        except Exception as exc:            # pickling error, worker death,
            self._disable(                 # broken pool, timeout, ...
                "build failed ({}: {}); disabling pool, run continues "
                "serial".format(type(exc).__name__, exc))
            return None
        built: List[dict] = []
        for part in parts:
            built.extend(part)
        return built

    def _disable(self, message: str) -> None:
        """First failure: one stderr line, then TERMINATE; stay dead.

        terminate() (not close()): a pool that just timed out or lost a
        worker may have wedged children, and ``close()+join()`` would wait
        on them forever.  Hard-coded safety net — never hang a run.
        """
        if not self._broken:
            log_prefetch_issue(message)
        self._broken = True
        self.terminate()

    def close(self) -> None:
        """close()+join() the pool; idempotent, never raises."""
        pool, self._pool = self._pool, None
        self._closed = True
        if pool is None:
            return
        try:
            pool.close()
            pool.join()
        except Exception:                   # pragma: no cover - defensive
            self._terminate_quiet(pool)

    def terminate(self) -> None:
        """terminate()+join() the pool; idempotent, never raises."""
        pool, self._pool = self._pool, None
        self._closed = True
        if pool is None:
            return
        self._terminate_quiet(pool)

    @staticmethod
    def _terminate_quiet(pool: Any) -> None:
        try:
            pool.terminate()
            pool.join()
        except Exception:                   # pragma: no cover - defensive
            pass

    def __enter__(self) -> "FeaturePrefetcher":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False


def maybe_create_prefetcher(
    level: str,
    intent_off: bool,
    *,
    flag_path: Optional[Path] = None,
) -> Optional[FeaturePrefetcher]:
    """Gate + construct: a pool iff the flag file enables it, else ``None``.

    Any construction failure logs one stderr line and returns ``None`` —
    the caller proceeds serially.  Callers MUST close() the result (the
    V5.3-MP trainer does so in a ``finally``).
    """
    workers = read_feature_workers(flag_path)
    if workers <= 0:
        return None
    try:
        return FeaturePrefetcher(workers, level, intent_off)
    except Exception as exc:
        log_prefetch_issue(
            "pool construction failed ({}: {}); running serial".format(
                type(exc).__name__, exc))
        return None
