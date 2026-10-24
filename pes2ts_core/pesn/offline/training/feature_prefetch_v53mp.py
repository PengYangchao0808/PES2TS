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

I3 (plan §4.3 / §5.5 / §7-B3/B4): :func:`workers_for_concurrency`
implements the plan's worker-scaling formula for B4 concurrent runs, and
:class:`DoubleBufferedPrefetcher` adds the plan's double-buffer — a
background thread prefetches update k+1's feature build while the main
thread runs forward on batch k.  The background thread NEVER touches the
feature LRU or any non-thread-safe state (it orchestrates
``dataset.records`` → ``to_sample_v3`` → ``pool.build_many`` and hands
the built list back; the main thread does the LRU put + collate serially
after popping).  Any thread/queue error logs one line and permanently
falls back to the synchronous path (same fail-open discipline as the
pool).  The ``(seed, update)`` sampling sequence and the feature bytes
are IDENTICAL to the sync path — pinned by
``tests/pesn/test_double_buffer_v53mp.py``.  All of it is opt-in via the
trainer's ``double_buffer=False`` default (B0–B2 reproducibility).
"""

from __future__ import annotations

import gc
import multiprocessing
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from pes2ts_core.pesn.features.key_atom_graph_v3 import (
    KeyAtomSampleV3,
    build_sample_features_v3,
)

__all__ = [
    "FEATURE_WORKERS_FLAG",
    "FEATURE_WORKERS_MAX",
    "FEATURE_WORKERS_MIN",
    "DoubleBufferedPrefetcher",
    "FeaturePrefetcher",
    "maybe_create_prefetcher",
    "read_feature_workers",
    "split_contiguous",
    "workers_for_concurrency",
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


def _worker_init() -> None:
    """Worker initializer: never run a GC cycle in the child.

    The child inherits the parent's whole heap (the 20k dataset is ~20 GiB).
    A collection would touch every tracked container's header, COW-copying
    those pages into the child — measured as a slow anon climb in the
    parent's cgroup.  Workers only build acyclic dicts/arrays, so the
    cycle collector has nothing to do anyway.
    """
    gc.disable()


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
        self._pool = ctx.Pool(workers, initializer=_worker_init)

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
    workers: Optional[int] = None,
) -> Optional[FeaturePrefetcher]:
    """Gate + construct: a pool iff enabled, else ``None``.

    ``workers`` (I3): explicit override of the flag-file gate (the run
    driver's ``--feature-workers``).  ``None`` (default) = today's
    flag-file behaviour, byte-identical; an int in [FEATURE_WORKERS_MIN,
    FEATURE_WORKERS_MAX] forces that many workers; ``0`` forces the serial
    path (``None``); out-of-range values are clamped (never a refusal —
    the orchestrator computes them from the concurrency formula).

    Any construction failure logs one stderr line and returns ``None`` —
    the caller proceeds serially.  Callers MUST close() the result (the
    V5.3-MP trainer does so in a ``finally``).
    """
    if workers is not None:
        if int(workers) <= 0:
            return None
        n_workers = max(FEATURE_WORKERS_MIN,
                        min(FEATURE_WORKERS_MAX, int(workers)))
    else:
        n_workers = read_feature_workers(flag_path)
        if n_workers <= 0:
            return None
    # BEFORE the fork: move everything already loaded (dataset light index,
    # label tables — the ~20 GiB that never dies) into the permanent
    # generation.  Children then inherit a heap the cycle collector never
    # walks, so forking does not COW-copy the dataset page by page.
    try:
        gc.freeze()
    except Exception:                       # pragma: no cover - defensive
        pass
    try:
        return FeaturePrefetcher(n_workers, level, intent_off)
    except Exception as exc:
        log_prefetch_issue(
            "pool construction failed ({}: {}); running serial".format(
                type(exc).__name__, exc))
        return None


def workers_for_concurrency(n_ways: int) -> int:
    """Plan §4.3 worker formula for ``n_ways`` concurrent runs.

    ``max(FEATURE_WORKERS_MIN, min(FEATURE_WORKERS_MAX, 32 // n_ways - 1))``
    — 12 workers single-way (cap), 12 at 2 ways (32//2-1=15, capped),
    9 at 3 ways, 7 at 4 ways, floor 2.  Guarded against n_ways < 1.
    """
    n = max(1, int(n_ways))
    return max(FEATURE_WORKERS_MIN,
               min(FEATURE_WORKERS_MAX, 32 // n - 1))


# ---------------------------------------------------------------------------
# double buffer (plan §4.3 / I3): prefetch update k+1 during forward on k
# ---------------------------------------------------------------------------
class DoubleBufferedPrefetcher:
    """Background-thread prefetch of the NEXT update's feature build.

    The (seed, update) sampler is a PURE function, so update k+1's
    positions can be computed while the main thread runs forward/backward
    on batch k.  The background thread ONLY:

    1. computes positions for ``update`` (caller-supplied pure ``compute``
       callable),
    2. calls ``dataset.records`` + ``dataset.to_sample_v3`` (read-only
       dataset access),
    3. submits ``pool.build_many`` and waits for the built list.

    It NEVER touches the feature LRU (no ``put_built``), the optimizer, or
    any trainer state — it returns ``{"positions", "records", "built"}``
    through a one-slot buffer guarded by a ``threading.Condition``.  The
    MAIN thread pops the result, does ``lru.put_built`` + collate
    serially.

    Thread-safety of the dataset overlap (e.g. inner-val running while the
    k+1 job loads geometry): all dataset reads are PURE functions of
    immutable files — concurrent geometry loads can only waste work
    (redundant decompress), never return different bytes; with the
    geometry pack (I2) there is no shared mutable state at all.  The
    feature LRU is the only shared MUTABLE structure and the thread never
    touches it.  Any exception in the thread logs ONE stderr line
    (``log_prefetch_issue``) and permanently disables the buffer — every
    later update takes the existing synchronous path (fail-open; a
    double-buffer bug costs performance only, never a run, never
    numerics: the positions for update k are ALWAYS recomputed by the
    caller's own ``_update_rng(seed, k)`` draw, so a stale/failed prefetch
    cannot alter the sampling sequence).

    One job in flight at a time (the trainer submits k+1 after popping k);
    ``submit`` on a busy buffer returns False and the caller falls back.
    ``close()`` is idempotent, joins the thread, never raises.
    """

    def __init__(
        self,
        pool: FeaturePrefetcher,
        dataset: Any,
        compute: Callable[[int], Sequence[int]],
        materialize: Callable[[Sequence[int]], Dict[str, Any]],
        *,
        level: str,
        intent_off: bool = False,
    ) -> None:
        """``compute(update) -> positions`` (pure); ``materialize(positions)
        -> {"records": [...], "samples": [...]}`` (dataset access)."""
        self._pool = pool
        self._dataset = dataset
        self._compute = compute
        self._materialize = materialize
        self._level = str(level)
        self._intent_off = bool(intent_off)
        self._cond = threading.Condition()
        self._pending: Optional[int] = None      # update queued for the thread
        self._inflight: bool = False             # thread working on a job
        self._result: Optional[Dict[str, Any]] = None
        self._broken = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="v53mp-dbuffer", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        return (not self._broken and not self._closed
                and self._pool is not None and self._pool.active)

    def submit(self, update: int) -> bool:
        """Queue the prefetch of ``update`` (main thread).  False = busy/off."""
        if not self.active:
            return False
        with self._cond:
            if self._result is not None and \
                    int(self._result["update"]) < int(update):
                self._result = None      # stale slot: never block the new job
            if self._pending is not None or self._inflight \
                    or self._result is not None:
                return False          # one slot: caller builds sync this step
            self._pending = int(update)
            self._cond.notify()
            return True

    def pop(self, update: int) -> Optional[Dict[str, Any]]:
        """Take the finished job for ``update`` (main thread) or None.

        A result for a DIFFERENT update is a legitimate timing artifact
        (the sync path already built that batch while the thread was still
        working) — it is dropped silently; the buffer stays live.  Pure
        ``(seed, update)`` functions make stale results harmless waste.
        """
        with self._cond:
            result = self._result
            self._result = None
        if result is None:
            return None
        if int(result["update"]) != int(update):
            return None
        return result

    def _disable(self) -> None:
        if not self._broken:
            log_prefetch_issue(
                "double buffer disabled; run continues on the synchronous "
                "prefetch path")
        self._broken = True

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait(timeout=0.5)
                if self._closed:
                    return
                update = self._pending
                self._pending = None
                self._inflight = True
            try:
                positions = [int(p) for p in self._compute(update)]
                packed = self._materialize(positions)
                built_list = self._pool.build_many(packed["samples"])
                if built_list is None or len(built_list) != len(positions):
                    # pool degraded mid-job — fall back (the caller's sync
                    # path re-checks lru misses and builds serially)
                    self._disable()
                    return
                result = {
                    "update": int(update),
                    "positions": positions,
                    "records": packed["records"],
                    "built": built_list,
                }
                with self._cond:
                    self._inflight = False
                    if self._closed:
                        return
                    self._result = result
            except Exception as exc:            # noqa: BLE001 - fail-open
                log_prefetch_issue(
                    "double-buffer job failed ({}: {}); disabling".format(
                        type(exc).__name__, exc))
                self._disable()
                return

    def close(self) -> None:
        """Stop the thread; idempotent, never raises."""
        self._closed = True
        with self._cond:
            self._cond.notify()
        thread, self._thread = self._thread, None
        if thread is None:
            return
        try:
            thread.join(timeout=30.0)
        except Exception:                       # pragma: no cover - defensive
            pass

    def __enter__(self) -> "DoubleBufferedPrefetcher":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False
