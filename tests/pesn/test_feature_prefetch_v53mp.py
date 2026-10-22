"""Feature-prefetch tests (feature_prefetch_v53mp + trainer wiring).

Gates for the opt-in fork-parallel feature build (S3 perf lane):

* equality: 40 synthetic ``KeyAtomSampleV3`` built serially vs via
  ``FeaturePrefetcher(8, "F2", False)`` are BIT-identical — same keys,
  ``np.array_equal`` INCLUDING dtype for ndarrays, ``==`` for
  scalars/tuples;
* fallback: a raising worker (or an unpicklable payload) disables the pool
  with ONE stderr line, every later call degrades to serial, and the
  serial path still returns the correct features (``_FeatureLRU`` /
  ``_prefetch_fill`` integration included);
* gate: with no flag file (or an invalid one) no pool is ever constructed
  — even when ``Pool`` construction is monkeypatched to explode — so the
  default campaign behaviour is the untouched serial path;
* LRU integration: ``put_built`` (prefetch path) and ``get`` (serial path)
  are numerics-equivalent, including the y_teacher zero-pad.

Fixture style mirrors tests/pesn/test_r3_v53mp_trainer.py (SN2 v3_fixtures
samples), but builds ``KeyAtomSampleV3`` directly — no CSV round-trip
needed for pure feature-build equality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
)
from pes2ts_core.pesn.offline.training import feature_prefetch_v53mp as fp  # noqa: E402
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (  # noqa: E402
    FeaturePrefetcher,
    maybe_create_prefetcher,
    read_feature_workers,
    split_contiguous,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    _FeatureLRU,
    _prefetch_fill,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample  # noqa: E402

LEVEL = "F2"
N_WORKERS = 8


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int = 40):
    """n deterministic, DISTINCT synthetic SN2 samples."""
    samples = []
    for i in range(n):
        direction = "forward" if i % 2 == 0 else "reverse"
        samples.append(make_sn2_sample(
            direction,
            node_index=i % 7,
            curve_id="cv{}".format(i % 5),
            d0=0.05 + 0.001 * i,
            jitter_seed=1000 + i,
        ))
    return samples


def _serial_build(samples):
    return [build_sample_features_v3(s, level=LEVEL, intent_off=False)
            for s in samples]


def _assert_features_identical(built, reference):
    assert len(built) == len(reference)
    for b, r in zip(built, reference):
        assert set(b) == set(r), "feature key sets differ"
        for key in r:
            rv, bv = r[key], b[key]
            if isinstance(rv, np.ndarray):
                assert isinstance(bv, np.ndarray), key
                assert bv.dtype == rv.dtype, \
                    "{}: dtype {} != {}".format(key, bv.dtype, rv.dtype)
                assert np.array_equal(bv, rv), "{}: values differ".format(key)
            else:
                assert type(bv) is type(rv), key
                assert bv == rv, "{}: values differ".format(key)


class _StubDataset:
    """Minimal dataset surface for _FeatureLRU/_prefetch_fill (samples as
    their own records; to_sample_v3 is the identity)."""

    def __init__(self, samples) -> None:
        self._samples = list(samples)

    def records(self, indices=None):
        if indices is None:
            return list(self._samples)
        return [self._samples[int(i)] for i in indices]

    def to_sample_v3(self, rec):
        return rec


# ---------------------------------------------------------------------------
# equality: serial vs 8-worker prefetch
# ---------------------------------------------------------------------------


def test_serial_vs_prefetcher_bit_identical():
    samples = _make_samples(40)
    reference = _serial_build(samples)
    with FeaturePrefetcher(N_WORKERS, LEVEL, False) as pool:
        assert pool.active
        built = pool.build_many(samples)
    assert built is not None and len(built) == 40
    _assert_features_identical(built, reference)


def test_split_contiguous_is_exact_partition():
    items = list(range(33))
    for n_chunks in (1, 4, 8, 12, 64):
        chunks = split_contiguous(items, n_chunks)
        assert len(chunks) <= min(n_chunks, len(items))
        assert [x for c in chunks for x in c] == items   # order-preserving
        sizes = [len(c) for c in chunks]
        assert max(sizes) - min(sizes) <= 1              # balanced
    assert split_contiguous([], 8) == []


# ---------------------------------------------------------------------------
# fallback: worker failure degrades to serial, never raises
# ---------------------------------------------------------------------------


def _raising_worker(samples, level, intent_off):
    raise RuntimeError("worker boom")


def test_worker_failure_disables_pool_and_degrades(monkeypatch, capsys):
    samples = _make_samples(8)
    reference = _serial_build(samples)
    # patch BEFORE construction so the forked children inherit the raiser
    monkeypatch.setattr(fp, "_worker_build_chunk", _raising_worker)
    pool = FeaturePrefetcher(N_WORKERS, LEVEL, False)
    try:
        assert pool.build_many(samples) is None
        assert not pool.active
        # every later call degrades silently to None (caller goes serial)
        assert pool.build_many(samples) is None
    finally:
        pool.close()
        pool.terminate()                    # idempotent, no orphans
    err = capsys.readouterr().err
    assert "v53mp feature prefetch" in err
    assert "worker boom" in err
    # serial path still returns the correct features
    _assert_features_identical(_serial_build(samples), reference)


def test_unpicklable_payload_disables_pool(capsys):
    samples = _make_samples(4)
    with FeaturePrefetcher(N_WORKERS, LEVEL, False) as pool:
        payload = list(samples)
        payload.append(object())            # not picklable
        assert pool.build_many(payload) is None
        assert not pool.active
        assert pool.build_many(samples) is None
    assert "v53mp feature prefetch" in capsys.readouterr().err


def test_prefetch_fill_fallback_still_correct_features(capsys):
    """Trainer-level integration: broken pool => _prefetch_fill is a no-op
    and the LRU's serial path yields bit-identical features."""
    samples = _make_samples(16)
    reference = _serial_build(samples)
    ds = _StubDataset(samples)

    broken = FeaturePrefetcher(N_WORKERS, LEVEL, False)
    broken._disable("forced failure for test")
    lru = _FeatureLRU(ds, capacity=64, level=LEVEL, intent_off=False)
    positions = list(range(16))
    _prefetch_fill(lru, ds, positions, broken)   # must not raise
    assert all(not lru.contains(p) for p in positions)
    _assert_features_identical(
        [lru.get(p) for p in positions], reference)
    assert "v53mp feature prefetch" in capsys.readouterr().err
    broken.close()


def test_prefetch_fill_populates_lru_identically():
    """Happy path: prefetched LRU entries == serial LRU entries."""
    samples = _make_samples(16)
    ds = _StubDataset(samples)
    positions = list(range(16))
    with FeaturePrefetcher(N_WORKERS, LEVEL, False) as pool:
        lru = _FeatureLRU(ds, capacity=64, level=LEVEL, intent_off=False)
        _prefetch_fill(lru, ds, positions, pool)
    assert all(lru.contains(p) for p in positions)
    serial_lru = _FeatureLRU(ds, capacity=64, level=LEVEL, intent_off=False)
    _assert_features_identical(
        [lru.get(p) for p in positions],
        [serial_lru.get(p) for p in positions])


def test_lru_put_built_matches_get_including_zero_pad():
    """put_built (prefetch path) replays get's y_teacher zero-pad exactly."""
    samples = _make_samples(3)
    samples[1] = samples[1].__class__(
        **{**{f: getattr(samples[1], f) for f in (
            "curve_id", "node_index", "intent", "atomic_numbers", "xyz",
            "d0", "endpoint_names")},
           "y_teacher": None, "frame_weight": 1.0, "label_status": "ok",
           "sample_id": "no-teacher", "extras": {}})
    ds = _StubDataset(samples)
    ref_lru = _FeatureLRU(ds, capacity=8, level=LEVEL, intent_off=False)
    reference = [ref_lru.get(p) for p in range(3)]
    assert reference[1]["y_teacher"].shape == (3, 3)
    assert not reference[1]["y_teacher"].any()

    lru = _FeatureLRU(ds, capacity=8, level=LEVEL, intent_off=False)
    for p in range(3):
        rec = ds.records([p])[0]
        built = build_sample_features_v3(
            ds.to_sample_v3(rec), level=LEVEL, intent_off=False)
        lru.put_built(p, rec, built)
    _assert_features_identical([lru.get(p) for p in range(3)], reference)


# ---------------------------------------------------------------------------
# gate: default OFF, no pool ever constructed
# ---------------------------------------------------------------------------


def test_read_feature_workers_boundaries(tmp_path, monkeypatch):
    cases = {"2": 2, "8": 8, "12": 12, " 8 \n": 8,
             "1": 0, "13": 0, "0": 0, "-3": 0, "abc": 0, "8.5": 0, "": 0}
    for text, expected in cases.items():
        flag = tmp_path / "feature_workers.on"
        flag.write_text(text, encoding="utf-8")
        assert read_feature_workers(flag) == expected, repr(text)
    # missing file => 0
    assert read_feature_workers(tmp_path / "missing.on") == 0
    # default path resolution (monkeypatched so the test does not depend on
    # whether the campaign host currently has the gate enabled)
    import pes2ts_core.pesn.offline.training.feature_prefetch_v53mp as _pf
    monkeypatch.setattr(_pf, "FEATURE_WORKERS_FLAG", tmp_path / "missing.on")
    assert read_feature_workers() == 0


def test_no_flag_file_never_constructs_pool(tmp_path, monkeypatch, capsys):
    def _explode(*a, **kw):
        raise AssertionError("Pool must not be constructed when gated off")

    monkeypatch.setattr(fp.multiprocessing, "get_context", _explode)
    flag = tmp_path / "feature_workers.on"      # absent => gate OFF
    assert maybe_create_prefetcher(LEVEL, False, flag_path=flag) is None
    flag.write_text("bogus", encoding="utf-8")  # invalid => gate OFF
    assert maybe_create_prefetcher(LEVEL, False, flag_path=flag) is None
    assert capsys.readouterr().err == ""        # silent when gated off


def test_valid_flag_creates_and_closes_cleanly(tmp_path):
    flag = tmp_path / "feature_workers.on"
    flag.write_text("4", encoding="utf-8")
    pool = maybe_create_prefetcher(LEVEL, False, flag_path=flag)
    try:
        assert pool is not None and pool.active and pool.workers == 4
        built = pool.build_many(_make_samples(6))
        assert built is not None and len(built) == 6
    finally:
        pool.close()
    assert not pool.active


def test_construction_failure_returns_none(tmp_path, monkeypatch, capsys):
    flag = tmp_path / "feature_workers.on"
    flag.write_text("4", encoding="utf-8")

    class _BrokenCtx:
        def Pool(self, *a, **kw):
            raise RuntimeError("cannot fork")

    monkeypatch.setattr(fp.multiprocessing, "get_context",
                        lambda *a, **kw: _BrokenCtx())
    assert maybe_create_prefetcher(LEVEL, False, flag_path=flag) is None
    assert "cannot fork" in capsys.readouterr().err


def test_invalid_worker_count_rejected():
    with pytest.raises(ValueError):
        FeaturePrefetcher(1, LEVEL, False)
    with pytest.raises(ValueError):
        FeaturePrefetcher(13, LEVEL, False)
