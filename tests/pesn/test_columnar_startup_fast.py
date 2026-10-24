"""Columnar startup fast-path equality tests (S4 plan §7-B2 / lane I4).

Every trainer/driver startup pass that I4 vectorizes is pinned here on the
synthetic mini fixture, BOTH branches:

* ``v53mp_pool_split`` — fast vs per-item: EXACT train/inner-val position
  lists for seeds 17/29/43 (the seeded curve-level shuffle is replayed
  verbatim at curve level over the STRING-sorted curve list, so the rng
  draw sequence cannot drift);
* ``_CurveDirIndex.build`` — same STRING-sorted curve list + same pools
  (each group's positions ascending);
* ``_assert_no_confirm`` — fast accepts the same sets / refuses the same
  confirm folds with the same typed error;
* ``_curve_id_set_of`` / ``estimate_endpoint_pos_weight`` /
  ``_train_q_mean`` / ``_assert_label_coverage`` — identical results;
* ``V53MP_DISABLE_COLUMNAR_FAST=1`` (module flag) forces the per-item
  path — the certificate's reference arm;
* the v1 loader (no ``_ds``) never takes the fast path, and its results
  equal the v2 fast path (cross-loader equality).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3Error,
)
from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from tests.pesn.test_r3_v53mp_geometry_pack import _build_stack  # noqa: E402
from tests.pesn.test_r3_v53mp_dataset import (  # noqa: E402
    ALL_STATUSES,
    N_FRAMES,
    _write_fixture,
)

SEEDS = (17, 29, 43)


@pytest.fixture()
def stack(tmp_path: Path) -> dict:
    tables = _write_fixture(tmp_path / "data")
    out = _build_stack(tables)
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    out["tables"] = tables
    out["v2"] = ColumnarKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    return out


@pytest.fixture(autouse=True)
def fast_on():
    """Every test runs with the fast path ON unless it flips it off."""
    tr._COLUMNAR_FAST = True
    yield
    tr._COLUMNAR_FAST = True


def _pool_ids_for(env: dict) -> list:
    """All curve ids of the mini fixture (the 20k-tier analogue)."""
    return list(env["v1"]._curves.keys())


# ---------------------------------------------------------------------------
# v53mp_pool_split: fast == per-item, multi-seed (rng-order replication)
# ---------------------------------------------------------------------------
def test_pool_split_fast_equals_per_item_all_seeds(stack) -> None:
    pool_ids = _pool_ids_for(stack)
    v1_samples = stack["v1"].samples        # per-item path (no _ds)
    v2_samples = stack["v2"].samples        # fast path
    for seed in SEEDS:
        for frac in (0.2, 0.34):
            ref_train, ref_val = tr.v53mp_pool_split(
                v1_samples, pool_curve_ids=pool_ids, seed=seed,
                inner_val_fraction=frac)
            got_train, got_val = tr.v53mp_pool_split(
                v2_samples, pool_curve_ids=pool_ids, seed=seed,
                inner_val_fraction=frac)
            assert got_train == ref_train, (seed, frac, "train")
            assert got_val == ref_val, (seed, frac, "val")
            assert sorted(got_train + got_val) == sorted(
                list(range(len(v1_samples))))


def test_pool_split_fast_disabled_flag_same_result(stack) -> None:
    """The env/module flag forces the per-item path; results unchanged."""
    pool_ids = _pool_ids_for(stack)
    v2_samples = stack["v2"].samples
    ref_train, ref_val = tr.v53mp_pool_split(
        v2_samples, pool_curve_ids=pool_ids, seed=17)
    tr._COLUMNAR_FAST = False
    try:
        got_train, got_val = tr.v53mp_pool_split(
            v2_samples, pool_curve_ids=pool_ids, seed=17)
    finally:
        tr._COLUMNAR_FAST = True
    assert got_train == ref_train and got_val == ref_val


def test_pool_split_rng_draw_sequence_replicated(stack) -> None:
    """The fast path replays the EXACT curve-level rng draws.

    The per-item and fast paths must construct the identical STRING-sorted
    curve list (the rng.shuffle input) — pin the list equality directly,
    which is what makes every downstream position identical.
    """
    pool_ids = _pool_ids_for(stack)
    v1_samples = stack["v1"].samples
    v2_samples = stack["v2"].samples
    st = v2_samples._ds.startup
    pool_ref = [i for i, s in enumerate(v1_samples)
                if str(s.curve_id) in set(pool_ids)]
    curves_ref = sorted({str(v1_samples[i].curve_id) for i in pool_ref})
    pool_fast = st.positions_in_curves(pool_ids)
    curves_fast = st.unique_sorted_curve_ids(pool_fast)
    assert curves_fast == curves_ref       # identical rng-shuffle input
    assert pool_fast.tolist() == pool_ref  # identical pool order


# ---------------------------------------------------------------------------
# _CurveDirIndex.build: fast == per-item
# ---------------------------------------------------------------------------
def test_curve_dir_index_fast_equals_per_item(stack) -> None:
    v1_samples = stack["v1"].samples
    v2_samples = stack["v2"].samples
    positions = list(range(len(v1_samples)))
    ref = tr._CurveDirIndex.build(v1_samples, positions)
    got = tr._CurveDirIndex.build(v2_samples, positions)
    assert got.curves == ref.curves
    assert set(got.pools.keys()) == set(ref.pools.keys())
    for key in ref.pools:
        assert got.pools[key] == ref.pools[key], key


def test_curve_dir_index_fast_on_subset(stack) -> None:
    """A non-contiguous position subset (train-like) matches per-item."""
    v1_samples = stack["v1"].samples
    v2_samples = stack["v2"].samples
    subset = [0, 2, 5, 7, 8, 11, 15]
    ref = tr._CurveDirIndex.build(v1_samples, subset)
    got = tr._CurveDirIndex.build(v2_samples, subset)
    assert got.curves == ref.curves
    assert {k: list(v) for k, v in got.pools.items()} == \
        {k: list(v) for k, v in ref.pools.items()}


# ---------------------------------------------------------------------------
# _assert_no_confirm: fast accepts / refuses identically
# ---------------------------------------------------------------------------
def test_assert_no_confirm_fast_accepts_and_refuses(stack) -> None:
    v1_samples = stack["v1"].samples
    v2_samples = stack["v2"].samples
    clean = list(range(len(v1_samples)))
    tr._assert_no_confirm(v1_samples, clean, "test")
    tr._assert_no_confirm(v2_samples, clean, "test")     # no raise
    # a confirm-fold position must refuse on BOTH paths with the same code
    # (the mini fixture folds are 0/1; synthesize a fold-3 refusal by
    # monkeypatching the fold column)
    st = v2_samples._ds.startup
    original = v2_samples._ds._columns["fold"]
    tampered = np.asarray(original).copy()
    tampered[3] = 3
    v2_samples._ds._columns["fold"] = tampered
    try:
        with pytest.raises(tr.V53MPTrainerError) as exc_fast:
            tr._assert_no_confirm(v2_samples, clean, "test")
        assert exc_fast.value.code == "CONFIRM_POSITIONS"
        tr._COLUMNAR_FAST = False
        try:
            with pytest.raises(tr.V53MPTrainerError) as exc_slow:
                tr._assert_no_confirm(v2_samples, clean, "test")
        finally:
            tr._COLUMNAR_FAST = True
        assert exc_slow.value.code == "CONFIRM_POSITIONS"
        assert str(exc_fast.value) == str(exc_slow.value)
    finally:
        v2_samples._ds._columns["fold"] = original
    del st


# ---------------------------------------------------------------------------
# _curve_id_set_of / estimate_endpoint_pos_weight / _train_q_mean
# ---------------------------------------------------------------------------
def test_curve_id_set_fast_equals_per_item(stack) -> None:
    v1_samples = stack["v1"].samples
    v2_samples = stack["v2"].samples
    for positions in (list(range(len(v1_samples))), [0, 3, 8, 15], []):
        assert tr._curve_id_set_of(v2_samples, positions) == \
            tr._curve_id_set_of(v1_samples, positions)


def _mini_endpoint_table(env: dict):
    """Self-contained endpoint table over the v1 fixture rows."""
    rows = {}
    for rec in env["v1"].samples:
        node = int(rec.node_index)
        fwd = rec.intent.direction == "forward"
        end_node = N_FRAMES - 1 if fwd else 0
        label = 1 if node == end_node else (0 if node != 2 else -1)
        rows[rec.sample_id] = SimpleNamespace(
            endpoint_label=label, d_end_angstrom=0.1,
            bond_edit_complete=(label == 1), geometry_legal=True,
            ts_row=False, start_row=(node == 0), terminal_row=False,
            degenerate_curve=False)
    return SimpleNamespace(by_sample=rows)


def test_estimate_endpoint_pos_weight_fast_equals_per_item(stack) -> None:
    env = stack
    ep = _mini_endpoint_table(env)
    v1_samples = env["v1"].samples
    v2_samples = env["v2"].samples
    train_pos = list(range(0, len(v1_samples), 2))
    # per-item reference on v1
    tr._COLUMNAR_FAST = False
    try:
        ref = tr.estimate_endpoint_pos_weight(
            v1_samples, ep, train_pos, seed=17)
    finally:
        tr._COLUMNAR_FAST = True
    # fast path needs the ep_label_col + covers_all_positions on the table
    ep_col = np.asarray([ep.by_sample[sid].endpoint_label
                         for sid in (v2_samples[p].sample_id
                                     for p in range(len(v2_samples)))],
                        dtype=np.int8)
    ep_fast = SimpleNamespace(
        by_sample=ep.by_sample,
        ep_label_col=ep_col,
        covers_all_positions=lambda n: len(ep.by_sample) == n)
    got = tr.estimate_endpoint_pos_weight(
        v2_samples, ep_fast, train_pos, seed=17)
    assert got == ref
    # and the v1 per-item path with the fast-capable table also agrees
    got_v1 = tr.estimate_endpoint_pos_weight(
        v1_samples, ep_fast, train_pos, seed=17)
    assert got_v1 == ref


def test_train_q_mean_fast_equals_per_item(stack) -> None:
    env = stack
    v1_samples = env["v1"].samples
    v2_samples = env["v2"].samples
    # build a q/q_mask table aligned to sample order
    q = np.asarray([0.1 * (i % 5) - 0.2
                    for i in range(len(v1_samples))], dtype=np.float64)
    q_mask = np.asarray([i % 3 != 0 for i in range(len(v1_samples))],
                        dtype=bool)
    sids = [v1_samples[i].sample_id for i in range(len(v1_samples))]
    rows = {sid: SimpleNamespace(q=float(q[i]), q_mask=bool(q_mask[i]))
            for i, sid in enumerate(sids)}
    labels = SimpleNamespace(
        by_sample=rows, q_col=q, q_mask_col=q_mask,
        covers_all_positions=lambda n: len(rows) == n)
    train_pos = list(range(0, len(v1_samples), 2))
    tr._COLUMNAR_FAST = False
    try:
        ref = tr._train_q_mean(v1_samples, labels, train_pos)
    finally:
        tr._COLUMNAR_FAST = True
    got = tr._train_q_mean(v2_samples, labels, train_pos)
    assert got is not None and ref is not None
    assert np.float64(got).tobytes() == np.float64(ref).tobytes()
    # all-masked → None on both paths
    none_labels = SimpleNamespace(
        by_sample={sid: SimpleNamespace(q=0.0, q_mask=False)
                   for sid in sids},
        q_col=np.zeros(len(sids)), q_mask_col=np.zeros(len(sids), bool),
        covers_all_positions=lambda n: True)
    tr._COLUMNAR_FAST = False
    try:
        ref_none = tr._train_q_mean(v1_samples, none_labels, train_pos)
    finally:
        tr._COLUMNAR_FAST = True
    assert tr._train_q_mean(v2_samples, none_labels, train_pos) is None
    assert ref_none is None


# ---------------------------------------------------------------------------
# _assert_label_coverage
# ---------------------------------------------------------------------------
def test_label_coverage_fast_and_per_item(stack) -> None:
    env = stack
    v1_samples = env["v1"].samples
    v2_samples = env["v2"].samples
    n = len(v1_samples)
    sids = [v1_samples[i].sample_id for i in range(n)]
    full = SimpleNamespace(by_sample={sid: object() for sid in sids},
                           covers_all_positions=lambda m: m == n,
                           _n_rows=n)
    groups = [list(range(0, n, 2)), list(range(1, n, 2))]
    # fast path: structural coverage + probe → no raise
    tr._assert_label_coverage(v2_samples, full, full, groups, n)
    # per-item path: identical outcome
    tr._COLUMNAR_FAST = False
    try:
        tr._assert_label_coverage(v1_samples, full, full, groups, n)
    finally:
        tr._COLUMNAR_FAST = True
    # a MISSING label row must refuse on both paths
    partial = SimpleNamespace(
        by_sample={sid: object() for sid in sids[:-1]},
        covers_all_positions=lambda m: False,   # forces per-item fallback
        _n_rows=n - 1)
    tr._COLUMNAR_FAST = False
    try:
        with pytest.raises(tr.V53MPTrainerError) as exc_slow:
            tr._assert_label_coverage(v1_samples, partial, full,
                                      [[n - 1]], n)
    finally:
        tr._COLUMNAR_FAST = True
    assert exc_slow.value.code == "LABEL_MISSING"
    # structural coverage mismatch (row count) refuses on the fast path
    # with the structural code (fail-closed: never silently under-checked)
    short = SimpleNamespace(by_sample=full.by_sample,
                            covers_all_positions=lambda m: False,
                            _n_rows=n - 1)
    with pytest.raises(tr.V53MPTrainerError) as exc_fast:
        tr._assert_label_coverage(v2_samples, short, full, groups, n)
    assert exc_fast.value.code == "LABEL_COVERAGE"


# ---------------------------------------------------------------------------
# cross-loader: v1 per-item == v2 fast (the certificate's premise)
# ---------------------------------------------------------------------------
def test_cross_loader_pool_split_and_index_equal(stack) -> None:
    pool_ids = _pool_ids_for(stack)
    v1_samples = stack["v1"].samples
    v2_samples = stack["v2"].samples
    t1_train, t1_val = tr.v53mp_pool_split(
        v1_samples, pool_curve_ids=pool_ids, seed=17)
    t2_train, t2_val = tr.v53mp_pool_split(
        v2_samples, pool_curve_ids=pool_ids, seed=17)
    assert t1_train == t2_train and t1_val == t2_val
    i1 = tr._CurveDirIndex.build(v1_samples, t1_train)
    i2 = tr._CurveDirIndex.build(v2_samples, t2_train)
    assert i1.curves == i2.curves
    assert {k: list(v) for k, v in i1.pools.items()} == \
        {k: list(v) for k, v in i2.pools.items()}
