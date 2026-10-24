"""Double-buffer prefetch tests (S4 plan §4.3 / lane I3).

Coverage:

* **200-update positional + collate equality** (the I3 hard gate (c)) —
  for a tiny synthetic run, the position draws AND the collated batch
  bytes of 200 consecutive updates are IDENTICAL between the synchronous
  path and the double-buffered path (a real fork pool + real background
  thread), proving the ``(seed, update)`` sampling sequence and the
  feature bytes cannot drift;
* **fail-open** — a broken pool disables the buffer with ONE stderr line
  and later submits return False; a materialize error disables the
  buffer; a stale result for another update is dropped silently (timing
  artifact, never a bug-disable);
* **workers_for_concurrency** — the plan §4.3 formula, boundary-clamped;
* **maybe_create_prefetcher(workers=)** — explicit override vs the flag
  gate (workers=0 → serial; None → flag-file behaviour unchanged);
* **trainer integration** — a smoke run with ``double_buffer=True``
  produces metrics EXACTLY equal to the same run without it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
)
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (  # noqa: E402
    FEATURE_WORKERS_MAX,
    FEATURE_WORKERS_MIN,
    DoubleBufferedPrefetcher,
    FeaturePrefetcher,
    maybe_create_prefetcher,
    workers_for_concurrency,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    FEATURE_LEVEL_V53MP,
    V53MPTrainerConfig,
    _curve_batch_positions,
    _CurveDirIndex,
    _update_rng,
    v53mp_pool_split,
)

LEVEL = FEATURE_LEVEL_V53MP
N_WORKERS = 2


# ---------------------------------------------------------------------------
# workers_for_concurrency (plan §4.3 formula)
# ---------------------------------------------------------------------------
def test_workers_for_concurrency_formula() -> None:
    # max(2, min(12, 32//n - 1))
    assert workers_for_concurrency(1) == 12      # 32//1-1=31 -> cap 12
    assert workers_for_concurrency(2) == 12      # 32//2-1=15 -> cap 12
    assert workers_for_concurrency(3) == 9       # 32//3-1=9
    assert workers_for_concurrency(4) == 7       # 32//4-1=7
    assert workers_for_concurrency(32) == 2      # 32//32-1=0 -> floor 2
    assert workers_for_concurrency(100) == 2     # floor
    assert workers_for_concurrency(0) == 12      # guarded to n>=1
    assert workers_for_concurrency(-3) == 12


# ---------------------------------------------------------------------------
# maybe_create_prefetcher(workers=...) override (driver --feature-workers)
# ---------------------------------------------------------------------------
def test_maybe_create_prefetcher_explicit_workers(tmp_path: Path) -> None:
    # workers=0 forces serial (no pool) even with a valid flag file
    flag = tmp_path / "feature_workers.on"
    flag.write_text("8", encoding="utf-8")
    assert maybe_create_prefetcher(
        LEVEL, False, flag_path=flag, workers=0) is None
    # explicit int overrides the flag file value
    pool = maybe_create_prefetcher(
        LEVEL, False, flag_path=flag, workers=5)
    try:
        assert pool is not None and pool.workers == 5
        assert pool.active
    finally:
        pool.close()
    # out-of-range explicit values are CLAMPED, never refused
    pool = maybe_create_prefetcher(
        LEVEL, False, flag_path=flag, workers=99)
    try:
        assert pool.workers == FEATURE_WORKERS_MAX
    finally:
        pool.close()
    pool = maybe_create_prefetcher(LEVEL, False, flag_path=flag, workers=1)
    try:
        assert pool.workers == FEATURE_WORKERS_MIN
    finally:
        pool.close()
    # workers=None keeps the flag-file behaviour byte-identical
    pool = maybe_create_prefetcher(LEVEL, False, flag_path=flag)
    try:
        assert pool is not None and pool.workers == 8
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# DoubleBufferedPrefetcher: fail-open disciplines
# ---------------------------------------------------------------------------
class _FakePool:
    """Minimal FeaturePrefetcher stand-in for pure dbuf unit tests."""

    def __init__(self) -> None:
        self.active = True
        self.n_calls = 0

    def build_many(self, samples):
        self.n_calls += 1
        return [{"built_from": s} for s in samples]

    def terminate(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


class _DeadPool:
    """A pool that LOOKS active but fails on build_many (mid-flight death)."""

    active = True

    def build_many(self, samples):
        return None

    def terminate(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


def _trivial_compute(n: int):
    return lambda update: list(range(int(update), int(update) + n))


def _trivial_materialize(positions):
    return {"records": list(positions),
            "samples": [{"p": p} for p in positions]}


def _wait_result(dbuf, update, timeout_s: float = 5.0):
    import time as _t
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        got = dbuf.pop(update)
        if got is not None:
            return got
        _t.sleep(0.01)
    return None


def test_dbuf_returns_built_list_for_submitted_update() -> None:
    pool = _FakePool()
    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=_trivial_compute(4),
        materialize=_trivial_materialize, level=LEVEL)
    try:
        assert dbuf.submit(10)
        got = _wait_result(dbuf, 10)
        assert got is not None and got["update"] == 10
        assert got["positions"] == [10, 11, 12, 13]
        assert got["records"] == [10, 11, 12, 13]
        assert len(got["built"]) == 4
        assert got["built"][0] == {"built_from": {"p": 10}}
    finally:
        dbuf.close()


def test_dbuf_broken_pool_disables_with_one_log(capsys) -> None:
    pool = _DeadPool()
    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=_trivial_compute(2),
        materialize=_trivial_materialize, level=LEVEL)
    try:
        assert dbuf.submit(1)
        import time as _t
        _t.sleep(0.4)
        assert not dbuf.active
        assert dbuf.pop(1) is None
        # later submits fall back (False) — the caller builds synchronously
        assert not dbuf.submit(2)
        err = capsys.readouterr().err
        assert "double buffer disabled" in err
    finally:
        dbuf.close()


def test_dbuf_materialize_error_disables_fail_open(capsys) -> None:
    def bad_materialize(positions):
        raise RuntimeError("synthetic materialize failure")
    pool = _FakePool()
    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=_trivial_compute(2),
        materialize=bad_materialize, level=LEVEL)
    try:
        assert dbuf.submit(1)
        import time as _t
        _t.sleep(0.4)
        assert not dbuf.active
        err = capsys.readouterr().err
        assert "double-buffer job failed" in err
        assert "synthetic materialize failure" in err
        assert not dbuf.submit(2)              # permanently fell back
    finally:
        dbuf.close()


def test_dbuf_compute_error_disables_fail_open(capsys) -> None:
    def bad_compute(update):
        raise ValueError("synthetic compute failure")
    pool = _FakePool()
    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=bad_compute,
        materialize=_trivial_materialize, level=LEVEL)
    try:
        assert dbuf.submit(1)
        import time as _t
        _t.sleep(0.4)
        assert not dbuf.active
        err = capsys.readouterr().err
        assert "double-buffer job failed" in err
    finally:
        dbuf.close()


def test_dbuf_stale_result_dropped_silently(capsys) -> None:
    """A result for update k when k+1 is requested is a timing artifact:
    dropped silently, buffer stays live."""
    pool = _FakePool()
    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=_trivial_compute(2),
        materialize=_trivial_materialize, level=LEVEL)
    try:
        assert dbuf.submit(5)
        got = _wait_result(dbuf, 5)
        assert got is not None
        # push a result for update 5 into the slot, then pop for 6
        with dbuf._cond:
            dbuf._result = {"update": 5, "positions": [5, 6],
                            "records": [5, 6], "built": [{}, {}]}
        assert dbuf.pop(6) is None          # stale -> silent drop
        assert dbuf.active                   # NOT disabled
        err = capsys.readouterr().err
        assert "double buffer" not in err    # no log for a timing artifact
        # and a stale slot never blocks a NEW submit
        assert dbuf.submit(7)
        got7 = _wait_result(dbuf, 7)
        assert got7 is not None and got7["update"] == 7
    finally:
        dbuf.close()


def test_dbuf_submit_busy_returns_false() -> None:
    """While a job is pending/in-flight, submit returns False (one slot)."""
    pool = _FakePool()
    gate = {"open": False}
    import threading
    release = threading.Event()

    def slow_materialize(positions):
        release.wait(timeout=5.0)
        return _trivial_materialize(positions)

    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=_trivial_compute(2),
        materialize=slow_materialize, level=LEVEL)
    try:
        assert dbuf.submit(1)         # pending; thread picks it up and blocks
        import time as _t
        _t.sleep(0.1)
        assert not dbuf.submit(2)     # busy: caller builds sync this step
        release.set()
        got = _wait_result(dbuf, 1)
        assert got is not None and got["update"] == 1
    finally:
        release.set()
        dbuf.close()
    del gate


def test_dbuf_close_idempotent() -> None:
    pool = _FakePool()
    dbuf = DoubleBufferedPrefetcher(
        pool, dataset=None, compute=_trivial_compute(2),
        materialize=_trivial_materialize, level=LEVEL)
    dbuf.close()
    dbuf.close()                     # idempotent, never raises
    assert not dbuf.active
    assert dbuf.submit(1) is False


# ---------------------------------------------------------------------------
# THE I3 hard gate (c): 200-update position + collate equality
# ---------------------------------------------------------------------------
def _collate_fingerprint(positions: list, built_list: list) -> bytes:
    """Bitwise fingerprint of one collated batch (positions + every array)."""
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps([int(p) for p in positions]).encode())
    batch = collate_v3(built_list)
    for key in sorted(batch):
        value = batch[key]
        h.update(str(key).encode())
        if isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(value.tobytes())
        else:
            h.update(repr(value).encode())
    return h.digest()


@pytest.fixture()
def env(tmp_path):
    torch = pytest.importorskip("torch")
    del torch
    from tests.pesn.test_r3_v53mp_trainer import _build_env
    folds = {"cv0": 1, "cv1": 1, "cv2": 1, "cv_dev0": 0, "cv_dev1": 0}
    return _build_env(tmp_path, folds)


def test_200update_positions_and_collate_equal_sync_vs_dbuf(env) -> None:
    """200 updates: position draws AND collated batch bytes are IDENTICAL
    between the synchronous path and the double-buffered path."""
    ds = env.ds
    samples = ds.samples
    config = V53MPTrainerConfig(arm="A", seed=17, backbone_init="scratch")
    train_pos, _inner = v53mp_pool_split(
        samples, pool_curve_ids=["cv0", "cv1", "cv2"], seed=17)
    index = _CurveDirIndex.build(samples, train_pos)
    n_updates = 200

    # --- sync path: positions + serial feature build + collate --------------
    sync_fingerprints = []
    for update in range(1, n_updates + 1):
        positions = _curve_batch_positions(
            samples, train_pos, config=config,
            rng=_update_rng(17, update, salt=0), index=index)
        recs = ds.records(positions)
        built = [build_sample_features_v3(
            ds.to_sample_v3(r), level=LEVEL, intent_off=False)
            for r in recs]
        sync_fingerprints.append(_collate_fingerprint(positions, built))

    # --- double-buffered path: real pool + real background thread -----------
    with FeaturePrefetcher(N_WORKERS, LEVEL, False) as pool:
        def compute(u):
            return _curve_batch_positions(
                samples, train_pos, config=config,
                rng=_update_rng(17, int(u), salt=0), index=index)

        def materialize(positions):
            recs = ds.records([int(p) for p in positions])
            return {"records": recs,
                    "samples": [ds.to_sample_v3(r) for r in recs]}

        dbuf = DoubleBufferedPrefetcher(
            pool, ds, compute, materialize, level=LEVEL)
        try:
            dbuf_fingerprints = []
            for update in range(1, n_updates + 1):
                positions = compute(update)          # caller's own draw
                if update < n_updates:
                    dbuf.submit(update + 1)
                got = dbuf.pop(update)
                if got is not None and got["positions"] == positions:
                    built = got["built"]
                else:                                 # sync fallback
                    packed = materialize(positions)
                    built = [build_sample_features_v3(
                        s, level=LEVEL, intent_off=False)
                        for s in packed["samples"]]
                dbuf_fingerprints.append(
                    _collate_fingerprint(positions, built))
            assert dbuf.active                       # never disabled
            assert dbuf_fingerprints == sync_fingerprints, \
                "double-buffered collate bytes drifted from the sync path"
        finally:
            dbuf.close()


# ---------------------------------------------------------------------------
# trainer integration: double_buffer=True == double_buffer=False
# ---------------------------------------------------------------------------
def test_trainer_double_buffer_metrics_identical(env) -> None:
    """A tiny synthetic run with double_buffer=True equals the run without.

    Requires the feature pool (the dbuf is pool-gated); monkeypatch the
    trainer's gate so the pool exists in the test.
    """
    torch = pytest.importorskip("torch")
    del torch
    from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr
    from tests.pesn.test_r3_v53mp_trainer import _smoke_cfg

    tmp = env.ds.tables_dir.parent

    def _fake_prefetcher(level, intent_off, *, flag_path=None, workers=None):
        return FeaturePrefetcher(2, level, intent_off)

    monkey_target = tr.maybe_create_prefetcher
    tr.maybe_create_prefetcher = _fake_prefetcher
    try:
        cfg = _smoke_cfg(arm="A")
        out_off = tmp / "run_off"
        metrics_off = tr.train_run_v53mp(
            env.ds, env.labels, env.endpoint, cfg, out_off,
            dev_positions=env.dev_pos, resume=False, double_buffer=False)
        out_on = tmp / "run_on"
        metrics_on = tr.train_run_v53mp(
            env.ds, env.labels, env.endpoint, cfg, out_on,
            dev_positions=env.dev_pos, resume=False, double_buffer=True)
    finally:
        tr.maybe_create_prefetcher = monkey_target

    assert metrics_on["double_buffer"] is True
    assert metrics_on["double_buffer_active"] is True
    assert metrics_off["double_buffer"] is False
    # eval-derived blocks EXACTLY equal (all floats bitwise)
    for block in ("move", "move_guard_vs_r3", "q_head", "ordering",
                  "endpoint_head", "tau_end_calibration"):
        _assert_exact_equal(metrics_off[block], metrics_on[block], block)
    assert metrics_off["best_inner_val_loss"] == metrics_on["best_inner_val_loss"]
    assert metrics_off["last_train_loss"] == metrics_on["last_train_loss"]
    # training loop itself identical (elapsed_s is wall-clock, excluded)
    def _log_no_time(path: Path) -> list:
        rows = []
        for raw in path.read_text("utf-8").splitlines():
            if not raw.strip():
                continue
            doc = json.loads(raw)
            doc.pop("elapsed_s", None)
            rows.append(doc)
        return rows
    assert _log_no_time(out_off / "train_log.jsonl") == \
        _log_no_time(out_on / "train_log.jsonl")


def _assert_exact_equal(a, b, path: str) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), path
        for key in a:
            _assert_exact_equal(a[key], b[key], "{}.{}".format(path, key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_exact_equal(x, y, "{}[{}]".format(path, i))
    elif isinstance(a, float) and isinstance(b, float):
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), \
            (path, a, b)
    else:
        assert a == b, (path, a, b)
