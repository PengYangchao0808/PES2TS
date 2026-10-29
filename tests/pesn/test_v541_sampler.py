"""V5.4.1 WP4a stratified sampler tests (``v541_strata.py`` draws).

Covers the frozen ``v541_curve_stratified_v1`` draw contract:

* batch shape 8×4 = 32 frames, per curve four slots = all four strata
  with direction balance exactly 2+2 (forward slots first, fixed strata
  order);
* slot sources: ``stratum`` slots come from their own pool, ``fallback``
  slots from the SAME curve's full pool of the ASSIGNED direction, with
  fallback counts recorded and the batch never shrinking;
* missing direction pool → typed refusal, never a silent direction
  rebalance (the legacy ``or forward or reverse`` trick is forbidden);
* same (seed, update) reproducible INDEPENDENTLY ACROSS PROCESSES;
* RNG consumption order pinned (structural recorder + an independent
  manual replica of the spec's draw sequence);
* stratum→direction assignment is redrawn per curve/update (no fixed
  confounding);
* ``BatchScheduleHasher``: continuous chain == checkpoint/resumed chain,
  position sensitivity, contiguous-update + identity-checked resume;
* pool SHA behaviour and the empty-index / bad-batch refusals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training import v541_strata as vs  # noqa: E402
from tests.pesn.test_v541_strata import (  # noqa: E402
    duck_specs_degenerate,
    duck_specs_forward_only,
    duck_specs_full,
    make_duck_pair,
)

SEED = 17


@pytest.fixture(autouse=True)
def columnar_fast_on():
    tr._COLUMNAR_FAST = True
    yield
    tr._COLUMNAR_FAST = True


def _duck_index(specs: Dict[str, Dict[str, Any]]
                ) -> Tuple[List[Any], vs.StrataIndex]:
    v1, _v2, meta = make_duck_pair(specs)
    index = vs.StrataIndex.build(v1, list(range(len(v1))),
                                 curve_meta=meta)
    return v1, index


@pytest.fixture()
def duck_full() -> SimpleNamespace:
    v1, index = _duck_index(duck_specs_full())
    return SimpleNamespace(samples=v1, index=index)


@pytest.fixture()
def duck_degenerate() -> SimpleNamespace:
    v1, index = _duck_index(duck_specs_degenerate())
    return SimpleNamespace(samples=v1, index=index)


# ---------------------------------------------------------------------------
# batch shape: 8 curves × 4 frames, direction 2+2, four strata per curve
# ---------------------------------------------------------------------------
def test_batch_shape_8x4_two_plus_two(duck_full) -> None:
    index = duck_full.index
    n_samples = len(duck_full.samples)
    assert len(index.curves) >= vs.BATCH_CURVES_V541
    for update in range(1, 25):
        batch = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        assert len(batch.chosen_curves) == vs.BATCH_CURVES_V541 == 8
        assert len(set(batch.chosen_curves)) == 8        # 12 curves: no dup
        assert len(batch.positions) == 32 == 8 * 4       # never shrinks
        assert len(batch.slots) == 32
        assert all(0 <= p < n_samples for p in batch.positions)
        assert set(batch.fallback_counts) == {
            "forward", "reverse", "total"}
        for g, curve in enumerate(batch.chosen_curves):
            group = batch.slots[4 * g:4 * g + 4]
            assert len(group) == 4
            assert all(s.curve_id == curve for s in group)
            # frozen slot order: forward pair first, reverse pair second
            assert [s.direction for s in group] == [
                "forward", "forward", "reverse", "reverse"]
            fwd = [s.stratum for s in group[:2]]
            rev = [s.stratum for s in group[2:]]
            # fixed strata order inside each direction ...
            assert fwd == [s for s in vs.STRATA_ORDER_V541 if s in set(fwd)]
            assert rev == [s for s in vs.STRATA_ORDER_V541 if s in set(rev)]
            # ... and the four slots cover the four strata exactly once
            assert set(fwd).isdisjoint(rev)
            assert set(fwd) | set(rev) == set(vs.STRATA_ORDER_V541)
            for slot in group:
                assert slot.position in index.dir_pools[
                    (slot.curve_id, slot.direction)]


# ---------------------------------------------------------------------------
# slot sources + fallback counting (degenerate curves, no batch shrink)
# ---------------------------------------------------------------------------
def test_slot_sources_fallback_counts_and_no_shrink(duck_degenerate) -> None:
    index = duck_degenerate.index
    # n=4, ts=2 → the FORWARD post_TS stratum pool never exists
    assert not any(
        (c, "forward", "post_TS") in index.pools for c in index.curves)
    total_fallback = 0
    for update in range(1, 17):
        batch = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        assert len(batch.positions) == 32          # 不缩batch
        counted = {"forward": 0, "reverse": 0, "total": 0}
        for slot in batch.slots:
            key = (slot.curve_id, slot.direction, slot.stratum)
            pool = index.pools.get(key)
            if pool:
                assert slot.source == "stratum"
                assert slot.position in pool
            else:
                # same curve, SAME assigned direction (never the other one)
                assert slot.source == "fallback"
                assert slot.position in index.dir_pools[
                    (slot.curve_id, slot.direction)]
                counted[slot.direction] += 1
                counted["total"] += 1
            # the declared stratum of a non-fallback slot is its own
            if slot.source == "stratum":
                assert vs.strata_of_position_v541(
                    duck_degenerate.samples, index, slot.position) == (
                    slot.curve_id, slot.direction, slot.stratum)
        assert dict(batch.fallback_counts) == counted
        total_fallback += counted["total"]
        # every forward/post_TS assignment must have taken the fallback
        for slot in batch.slots:
            if slot.direction == "forward" and slot.stratum == "post_TS":
                assert slot.source == "fallback"
    assert total_fallback > 0


def test_full_strata_fixture_never_falls_back(duck_full) -> None:
    """All four pools exist on every curve/direction → zero fallbacks."""
    index = duck_full.index
    for cid in index.curves:
        for direction in vs.DIRECTIONS_V541:
            for stratum in vs.STRATA_ORDER_V541:
                assert index.pools[(cid, direction, stratum)]
    for update in range(1, 9):
        batch = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        assert batch.fallback_counts == {
            "forward": 0, "reverse": 0, "total": 0}
        assert all(s.source == "stratum" for s in batch.slots)


def test_fallback_counters_accumulate(duck_degenerate) -> None:
    index = duck_degenerate.index
    counters: Dict[str, int] = {}
    positions = vs.curve_batch_positions_v541(
        index, rng=vs.batch_rng_v541(SEED, 4), counters=counters)
    batch = vs.draw_curve_batch_v541(
        index, rng=vs.batch_rng_v541(SEED, 4))
    assert positions == list(batch.positions)
    assert counters == dict(batch.fallback_counts)
    totals = {"forward": 0, "reverse": 0, "total": 0}
    counters2: Dict[str, int] = {}
    for update in (1, 2, 3):
        vs.curve_batch_positions_v541(
            index, rng=vs.batch_rng_v541(SEED, update), counters=counters2)
        one = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        for key in totals:
            totals[key] += one.fallback_counts[key]
    assert counters2 == totals


# ---------------------------------------------------------------------------
# missing direction pool → typed refusal (no silent rebalance)
# ---------------------------------------------------------------------------
def test_missing_direction_pool_typed_refusal() -> None:
    v1, _v2, meta = make_duck_pair(duck_specs_forward_only(n_curves=2))
    index = vs.StrataIndex.build(v1, list(range(len(v1))),
                                 curve_meta=meta)
    # forward pools exist; the reverse direction is wholly absent
    assert index.curves and all(d == "forward"
                                for (_c, d) in index.dir_pools)
    assert not any(d == "reverse" for (_c, d, _s) in index.pools)
    with pytest.raises(vs.V541StrataError) as exc:
        vs.draw_curve_batch_v541(index, rng=vs.batch_rng_v541(SEED, 1))
    assert exc.value.code == "SAMPLER_DIRECTION_POOL_EMPTY"
    assert "no silent direction rebalance" in str(exc.value)
    # the legacy sampler would have borrowed the forward pool — this one
    # must not have mutated anything either
    assert not any(d == "reverse" for (_c, d) in index.dir_pools)


# ---------------------------------------------------------------------------
# reproducibility (same process / cross process) + seed separation
# ---------------------------------------------------------------------------
def test_deterministic_same_seed_update() -> None:
    v1, index = _duck_index(duck_specs_full())
    del v1
    for update in (1, 5, 9):
        a = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        b = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        assert a.positions == b.positions
        assert a.chosen_curves == b.chosen_curves
        assert a.fallback_counts == b.fallback_counts
    same_seed = [vs.draw_curve_batch_v541(
        index, rng=vs.batch_rng_v541(SEED, u)).positions
        for u in range(1, 13)]
    assert any(p != same_seed[0] for p in same_seed[1:])
    other_seed = [vs.draw_curve_batch_v541(
        index, rng=vs.batch_rng_v541(29, u)).positions
        for u in range(1, 13)]
    assert other_seed != same_seed


def _subprocess_draw() -> List[Dict[str, Any]]:
    """Deterministic draw slice used by BOTH arms of the process test."""
    v1, _v2, meta = make_duck_pair(duck_specs_full())
    index = vs.StrataIndex.build(v1, list(range(len(v1))),
                                 curve_meta=meta)
    out: List[Dict[str, Any]] = []
    for update in range(1, 6):
        batch = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(29, update))
        out.append({
            "update": update,
            "positions": [int(p) for p in batch.positions],
            "chosen": [str(c) for c in batch.chosen_curves],
            "fallback": {k: int(v)
                         for k, v in batch.fallback_counts.items()},
            "pool_sha": index.pool_sha256,
        })
    return out


def test_same_seed_update_reproducible_across_processes() -> None:
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from tests.pesn.test_v541_sampler import _subprocess_draw\n"
        "print('V541JSON:' + json.dumps(_subprocess_draw()))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=str(REPO), env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [l for l in proc.stdout.splitlines()
             if l.startswith("V541JSON:")]
    assert lines, proc.stdout[-2000:]
    got = json.loads(lines[-1][len("V541JSON:"):])
    assert got == _subprocess_draw()


# ---------------------------------------------------------------------------
# RNG consumption order pinned
# ---------------------------------------------------------------------------
class _RecordingRng:
    """Duck rng: records ``(size, replace, n_items)`` of every choice."""

    def __init__(self, inner: np.random.RandomState) -> None:
        self._inner = inner
        self.calls: List[Tuple[Any, bool, int]] = []

    def choice(self, a, size=None, replace=True, p=None):
        result = self._inner.choice(a, size=size, replace=replace, p=p)
        self.calls.append((None if size is None else int(size),
                           bool(replace), len(a)))
        return result


def _manual_replica(
    index: vs.StrataIndex, seed: int, update: int,
) -> List[int]:
    """Independent restatement of the SPEC's consumption order.

    If the implementation ever reorders/inserts an rng call, its rng
    state diverges from this replica and the positions differ.
    """
    rng = vs.batch_rng_v541(seed, update)
    curves = index.curves
    chosen = [str(c) for c in np.atleast_1d(rng.choice(
        curves, size=8, replace=len(curves) < 8))]
    positions: List[int] = []
    for cid in chosen:
        pick = rng.choice(list(vs.STRATA_ORDER_V541), size=2,
                          replace=False)
        fwd = {str(s) for s in np.atleast_1d(pick)}
        for direction in ("forward", "reverse"):
            assigned = [s for s in vs.STRATA_ORDER_V541
                        if (s in fwd) == (direction == "forward")]
            for stratum in assigned:
                pool = index.pools.get((cid, direction, stratum))
                if not pool:
                    pool = index.dir_pools[(cid, direction)]
                positions.append(int(np.atleast_1d(
                    rng.choice(pool, size=1, replace=False))[0]))
    return positions


@pytest.mark.parametrize("fixture_name", ["duck_full", "duck_degenerate"])
def test_rng_consumption_order_pinned(request, fixture_name) -> None:
    scope = request.getfixturevalue(fixture_name)
    index = scope.index
    for update in range(1, 9):
        recorder = _RecordingRng(vs.batch_rng_v541(SEED, update))
        batch = vs.draw_curve_batch_v541(index, rng=recorder)
        plain = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        assert batch.positions == plain.positions
        calls = recorder.calls
        # 1 curve draw + per curve (1 strata-split + 4 frame draws)
        assert len(calls) == 1 + 8 * 5
        assert calls[0] == (8, len(index.curves) < 8, len(index.curves))
        for g in range(8):
            base = 1 + g * 5
            assert calls[base] == (2, False, 4)          # strata split
            for k in range(1, 5):
                size, replace, _n = calls[base + k]
                assert (size, replace) == (1, False)     # one frame each
        # the exact sequence equals the spec replica (order + arguments)
        assert list(batch.positions) == _manual_replica(index, SEED, update)
    # chosen curves of a fresh draw equal the very first rng call
    batch = vs.draw_curve_batch_v541(index, rng=vs.batch_rng_v541(SEED, 7))
    assert list(batch.chosen_curves) == [
        str(c) for c in np.atleast_1d(
            vs.batch_rng_v541(SEED, 7).choice(
                index.curves, size=8,
                replace=len(index.curves) < 8))]


def test_stratum_direction_assignment_not_fixed(duck_full) -> None:
    """No fixed stratum×direction binding can emerge from the draws."""
    index = duck_full.index
    forward = {s: 0 for s in vs.STRATA_ORDER_V541}
    reverse = {s: 0 for s in vs.STRATA_ORDER_V541}
    for update in range(1, 25):
        batch = vs.draw_curve_batch_v541(
            index, rng=vs.batch_rng_v541(SEED, update))
        for slot in batch.slots:
            table = forward if slot.direction == "forward" else reverse
            table[slot.stratum] += 1
    assert all(v > 0 for v in forward.values()), forward
    assert all(v > 0 for v in reverse.values()), reverse


# ---------------------------------------------------------------------------
# schedule hash chain: continuous == resumed; position-sensitive
# ---------------------------------------------------------------------------
def _schedule_positions(index: vs.StrataIndex) -> Dict[int, List[int]]:
    return {u: [int(p) for p in vs.draw_curve_batch_v541(
        index, rng=vs.batch_rng_v541(SEED, u)).positions]
        for u in range(1, 11)}


def test_schedule_hash_continuous_equals_resumed(duck_full) -> None:
    positions = _schedule_positions(duck_full.index)
    continuous = vs.BatchScheduleHasher()
    chain_at_5 = None
    state_at_5 = None
    for update in range(1, 11):
        continuous.record(update, positions[update])
        if update == 5:
            chain_at_5 = continuous.chain_sha256
            state_at_5 = continuous.state_dict()
    assert continuous.last_update == 10 and continuous.records == 10
    assert state_at_5 == {
        "sampler_id": vs.SAMPLER_ID_V541,
        "spec_sha256": vs.sampler_spec_sha256(),
        "last_update": 5,
        "chain_sha256": chain_at_5,
        "records": 5,
    }
    # JSON round-trip (checkpoint files are JSON-safe)
    resumed = vs.BatchScheduleHasher.from_state(
        json.loads(json.dumps(state_at_5)))
    assert resumed.chain_sha256 == chain_at_5
    for update in range(6, 11):
        resumed.record(update, positions[update])
    assert resumed.chain_sha256 == continuous.chain_sha256
    assert resumed.records == 10 and resumed.last_update == 10
    # state of the resumed hasher equals the continuous one at the end
    assert resumed.state_dict() == continuous.state_dict()


def test_schedule_hash_sensitivity_and_state_refusals(duck_full) -> None:
    positions = _schedule_positions(duck_full.index)
    a = vs.BatchScheduleHasher()
    b = vs.BatchScheduleHasher()
    for update in range(1, 7):
        a.record(update, positions[update])
        if update == 3:
            drifted = list(positions[3])
            drifted[0] = int(drifted[0]) + 1          # ONE frame changed
            b.record(update, drifted)
        else:
            b.record(update, positions[update])
    assert a.chain_sha256 != b.chain_sha256
    # chain label differs from a bare empty hasher
    assert vs.BatchScheduleHasher().chain_sha256 != a.chain_sha256
    # non-contiguous updates refuse
    c = vs.BatchScheduleHasher()
    c.record(1, [0, 1])
    with pytest.raises(vs.V541StrataError) as exc:
        c.record(3, [0, 1])
    assert exc.value.code == "SCHEDULE_UPDATE_NONCONTIGUOUS"
    with pytest.raises(vs.V541StrataError) as exc:
        c.record(-1, [0, 1])
    assert exc.value.code == "SCHEDULE_UPDATE_INVALID"
    # identity-checked resume
    good = a.state_dict()
    for tampered in (
        dict(good, sampler_id="v53mp_legacy"),
        dict(good, spec_sha256="0" * 64),
        dict(good, chain_sha256="zz"),
        dict(good, records=-1),
        {"last_update": 1},
    ):
        with pytest.raises(vs.V541StrataError) as exc:
            vs.BatchScheduleHasher.from_state(tampered)
        assert exc.value.code == "SCHEDULE_STATE_INVALID"


# ---------------------------------------------------------------------------
# hashes + entry-point refusals
# ---------------------------------------------------------------------------
def test_pool_sha_tracks_pool_content(duck_full) -> None:
    v1 = duck_full.samples
    index = duck_full.index
    positions = list(range(len(v1)))
    again = vs.StrataIndex.build(v1, positions, curve_meta=index.meta)
    assert again.pool_sha256 == index.pool_sha256
    subset = vs.StrataIndex.build(v1, positions[:37],
                                  curve_meta=index.meta)
    assert subset.pool_sha256 != index.pool_sha256
    assert vs.sampler_spec_sha256() == vs.sampler_spec_sha256()


def test_empty_index_and_batch_arg_refusals() -> None:
    v1, _v2, meta = make_duck_pair(duck_specs_full(n_curves=2))
    empty = vs.StrataIndex.build(v1, [], curve_meta=meta)
    assert empty.curves == []
    with pytest.raises(vs.V541StrataError) as exc:
        vs.draw_curve_batch_v541(empty, rng=vs.batch_rng_v541(SEED, 1))
    assert exc.value.code == "SAMPLER_CURVES_EMPTY"
    index = vs.StrataIndex.build(v1, list(range(len(v1))),
                                 curve_meta=meta)
    with pytest.raises(vs.V541StrataError) as exc:
        vs.draw_curve_batch_v541(index, rng=vs.batch_rng_v541(SEED, 1),
                                 batch_curves=0)
    assert exc.value.code == "SAMPLER_BATCH_INVALID"
