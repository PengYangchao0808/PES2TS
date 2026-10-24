"""Pinned H2D staging tests (S4 plan §6 / lane I3 / B3).

Coverage:

* **CPU delegation** — ``to_torch_batch_pinned`` with a CPU device (or
  no pool) IS ``to_torch_batch`` (same tensors, same dtypes);
* **CUDA equivalence** (skipif no GPU) — the same collated batch through
  both paths yields identical tensors (values via ``torch.equal`` + dtypes
  + non-array passthrough);
* **pool LRU** — keyed reuse returns the SAME pinned buffer (hits), shape
  misses allocate fresh, the byte cap evicts LRU, an oversized buffer is
  allocated but never pooled;
* **fail-open** — a failing acquire logs ONE line and falls back to the
  direct path with identical output;
* **micro-benchmark** (informational, GPU) — H2D ms for ~50 batches old
  vs new; logged, never a gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    to_torch_batch,
)
from pes2ts_core.pesn.offline.training.h2d_pinned_v53mp import (  # noqa: E402
    PinnedBufferPool,
    to_torch_batch_pinned,
)


def _sample_batch() -> dict:
    """A NUMPY collated batch mirroring ``collate_v3``'s output dtypes.

    ``make_v3_batch`` yields torch tensors (which ``to_torch_batch`` passes
    through untouched) — the staging path only touches np.ndarray values,
    so we build the numpy form directly (the real loop's input).
    """
    rng = np.random.RandomState(0)
    n_nodes_total, n_edges_total, n_keys_total = 40, 24, 8
    return {
        "node_pos": rng.randn(n_nodes_total, 3).astype(np.float32),
        "node_z": rng.randint(1, 119, n_nodes_total).astype(np.int64),
        "edge_index": rng.randint(0, n_nodes_total,
                                  (2, n_edges_total)).astype(np.int64),
        "edge_attr": rng.randn(n_edges_total, 6).astype(np.float32),
        "pair_index": np.zeros((2, 3), dtype=np.int64),
        "pair_edit": rng.randn(3).astype(np.float32),
        "frame_weight": np.array([1.0], dtype=np.float32),
        "y_teacher": rng.randn(n_keys_total, 3).astype(np.float32),
        "n_nodes": np.array([40] * 4, dtype=np.int64),
        "sample_ids": ("a:0:i:0.08", "b:1:j:0.08",
                       "c:2:k:0.08", "d:3:l:0.08"),
        "ptr": np.array([0, 10, 20, 30, 40], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# CPU delegation + no-pool passthrough
# ---------------------------------------------------------------------------
def test_cpu_device_delegates_to_direct_path() -> None:
    batch = _sample_batch()
    pool = PinnedBufferPool()
    got = to_torch_batch_pinned(batch, "cpu", pool)
    want = to_torch_batch(batch, "cpu")
    assert set(got) == set(want)
    for key in want:
        va, vb = want[key], got[key]
        if isinstance(va, torch.Tensor):
            assert va.dtype == vb.dtype and va.shape == vb.shape
            assert torch.equal(va, vb), key
        else:
            assert va == vb, key
    assert pool.hits == 0 and pool.misses == 0   # never touched on CPU


def test_none_pool_delegates() -> None:
    batch = _sample_batch()
    got = to_torch_batch_pinned(batch, "cpu", None)
    want = to_torch_batch(batch, "cpu")
    for key in want:
        va, vb = want[key], got[key]
        if isinstance(va, torch.Tensor):
            assert torch.equal(va, vb) and va.dtype == vb.dtype
        else:
            assert va == vb


# ---------------------------------------------------------------------------
# CUDA equivalence
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA required for the pinned H2D path")
def test_cuda_equivalence_values_and_dtypes() -> None:
    batch = _sample_batch()
    pool = PinnedBufferPool()
    got = to_torch_batch_pinned(batch, "cuda", pool)
    want = to_torch_batch(batch, "cuda")
    assert set(got) == set(want), "key sets differ"
    for key in want:
        va, vb = want[key], got[key]
        if isinstance(va, torch.Tensor):
            assert va.dtype == vb.dtype, key
            assert va.shape == vb.shape, key
            assert torch.equal(va.cpu(), vb.cpu()), key
        else:
            assert va == vb, key
    # the pinned path actually staged through the pool (arrays present)
    assert pool.misses > 0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA required for the pinned H2D path")
def test_cuda_repeated_batches_reuse_buffers() -> None:
    """Two identical-shape batches reuse the pooled pinned buffers."""
    pool = PinnedBufferPool()
    batch = _sample_batch()
    to_torch_batch_pinned(batch, "cuda", pool)
    misses_after_first = pool.misses
    to_torch_batch_pinned(batch, "cuda", pool)
    assert pool.hits >= 1, "second identical batch must hit the pool"
    assert pool.misses == misses_after_first


# ---------------------------------------------------------------------------
# pool LRU mechanics (CPU-pinned buffers work without a GPU? pin_memory
# requires CUDA; guard the pool tests on CUDA availability)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="pin_memory requires CUDA")
def test_pool_keyed_reuse_and_lru_cap() -> None:
    pool = PinnedBufferPool(capacity_bytes=1 << 20)   # 1 MiB
    a1 = pool.acquire("node_pos", (100, 3), torch.float32)
    a2 = pool.acquire("node_pos", (100, 3), torch.float32)
    assert a1.data_ptr() == a2.data_ptr()          # SAME buffer (hit)
    assert pool.hits == 1 and pool.misses == 1
    # different shape = miss + fresh alloc
    b = pool.acquire("node_pos", (200, 3), torch.float32)
    assert b.data_ptr() != a1.data_ptr()
    assert pool.misses == 2
    # different field = miss
    c = pool.acquire("edge_attr", (100, 3), torch.float32)
    assert c.data_ptr() != a1.data_ptr()
    # capacity eviction: fill beyond the cap
    big = PinnedBufferPool(capacity_bytes=64 << 10)
    for i in range(20):
        big.acquire("f{}".format(i % 3), (512, 3), torch.float32)
    assert big.used_bytes <= (64 << 10)
    assert big.n_buffers <= 20
    # oversized single buffer is allocated but never pooled
    huge = PinnedBufferPool(capacity_bytes=1 << 10)
    h = huge.acquire("huge", (1 << 20, 3), torch.float32)
    assert h.numel() == (1 << 20) * 3
    assert huge.n_buffers == 0 and huge.used_bytes == 0
    h2 = huge.acquire("huge", (1 << 20, 3), torch.float32)
    assert h2.data_ptr() != h.data_ptr()          # never cached


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="pin_memory requires CUDA")
def test_pool_stats_and_capacity_accessor() -> None:
    pool = PinnedBufferPool(capacity_bytes=1234)
    assert pool.capacity_bytes == 1234
    pool.acquire("x", (10, 3), torch.float32)
    stats = pool.stats()
    assert stats["capacity_bytes"] == 1234
    assert stats["misses"] == 1 and stats["hits"] == 0
    assert stats["used_bytes"] > 0 and stats["n_buffers"] == 1


# ---------------------------------------------------------------------------
# fail-open
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the pinned branch (and its fallback) is CUDA")
def test_fail_open_on_acquire_error(capsys) -> None:
    """A failing acquire logs once and falls back to the direct path."""

    class _BrokenPool(PinnedBufferPool):
        def acquire(self, field, shape, dtype):
            raise RuntimeError("synthetic acquire failure")

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

    real_cuda = torch.cuda
    try:
        torch.cuda = _FakeCuda()          # force the pinned branch
        batch = _sample_batch()
        got = to_torch_batch_pinned(batch, "cuda", _BrokenPool())
        want = to_torch_batch(batch, "cuda")   # the fallback's own target
        for key in want:
            va, vb = want[key], got[key]
            if isinstance(va, torch.Tensor):
                assert va.dtype == vb.dtype and va.shape == vb.shape
                assert torch.equal(va.cpu(), vb.cpu()), key
            else:
                assert va == vb
        err = capsys.readouterr().err
        assert "pinned staging failed" in err
        assert "synthetic acquire failure" in err
    finally:
        torch.cuda = real_cuda


def test_log_deduplicates_messages(capsys) -> None:
    from pes2ts_core.pesn.offline.training.h2d_pinned_v53mp import (
        _LOGGED_ISSUES,
        log_h2d_issue,
    )
    _LOGGED_ISSUES.clear()
    log_h2d_issue("unique-message-xyz")
    log_h2d_issue("unique-message-xyz")
    assert capsys.readouterr().err.count("unique-message-xyz") == 1


# ---------------------------------------------------------------------------
# informational micro-benchmark (never a gate)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA required for the H2D benchmark")
def test_h2d_microbenchmark_informational(capsys) -> None:
    """~50 batches: log H2D ms old vs new (informational, no assertion)."""
    import time
    n = 50
    batches = [_sample_batch() for _ in range(n)]
    # warmup
    to_torch_batch(batches[0], "cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in batches:
        to_torch_batch(b, "cuda")
    torch.cuda.synchronize()
    old_ms = (time.perf_counter() - t0) * 1000.0 / n
    pool = PinnedBufferPool()
    to_torch_batch_pinned(batches[0], "cuda", pool)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in batches:
        to_torch_batch_pinned(b, "cuda", pool)
    torch.cuda.synchronize()
    new_ms = (time.perf_counter() - t0) * 1000.0 / n
    print("[h2d benchmark] direct={:.3f} ms/batch  pinned={:.3f} ms/batch "
          "over {} batches".format(old_ms, new_ms, n))
    # informational: no gate on speed; the equivalence tests are the gate
    assert old_ms >= 0.0 and new_ms >= 0.0
