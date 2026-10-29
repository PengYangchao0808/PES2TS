"""V5.4.1 WP4a strata index tests (``v541_strata.py``).

Covers the frozen path-A stratification contract:

* machine-readable ``SAMPLER_SPEC_V541`` + rule SHA (canonical JSON);
* classification: exactly-one-stratum per position, TS-row rule,
  start/endpoint priority on short/degenerate curves, descending-path
  layout, forward/reverse mirror (with the frozen TS-row exception);
* ``StrataIndex.build``: both arms (v1 per-item / v2 startup columnar
  fast path) produce identical pools; the fast arm NEVER materialises a
  record (6M-sample discipline); ``V53MP_DISABLE_COLUMNAR_FAST``-style
  flag-off reference arm agrees; canonical ascending pools independent
  of the caller's indices order;
* metadata sources and typed refusals (source/curve/row/range/position);
* no label import and no label/teacher/q field access (AST scan + poison
  duck records);
* rebuild (double-buffer discipline) reproduces pools, pool SHA and the
  seeded draws;
* real loader fixture: v1 dataset vs columnar fast loader equivalence and
  the declared four-strata layout on real records (record gate poisoned
  on the v2 arm to prove no record construction).
"""
from __future__ import annotations

import ast
import hashlib
import json
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


# ---------------------------------------------------------------------------
# fixtures: duck loaders (v1 record list + v2 columnar startup)
# ---------------------------------------------------------------------------

#: Fields stratification must NEVER read (label/teacher side).
FORBIDDEN_LABEL_NAMES = frozenset({
    "label_status", "label_weight", "labels", "labels_v52", "endpoint_labels",
    "label_cache", "teacher", "q", "q_mask", "q_true", "q_canonical",
    "ts_row", "start_row", "terminal_row", "ep_label", "d_end", "stop_label",
    "curve_weight", "sample_weight", "y_teacher",
})


class DuckIntent:
    __slots__ = ("direction",)

    def __init__(self, direction: str) -> None:
        self.direction = str(direction)


class DuckRecord:
    """Structural-only sample record; label-side reads are poisoned."""

    __slots__ = ("curve_id", "node_index", "intent")

    def __init__(self, curve_id: str, node_index: int, direction: str) -> None:
        self.curve_id = str(curve_id)
        self.node_index = int(node_index)
        self.intent = DuckIntent(direction)

    def __getattribute__(self, name: str) -> Any:
        if name in FORBIDDEN_LABEL_NAMES:
            raise AssertionError(
                f"forbidden label/teacher field read: {name}")
        return object.__getattribute__(self, name)


class DuckStartup:
    """ColumnarStartup-shaped fast-path surface over duck columns."""

    __slots__ = ("_ds",)

    def __init__(self, ds: Any) -> None:
        self._ds = ds

    @property
    def curve_idx(self) -> Any:
        return self._ds._columns["curve_idx"]

    @property
    def curve_ids(self) -> List[str]:
        return self._ds._curve_ids

    def direction_int(self, positions: Any) -> Any:
        return self._ds._columns["dir01"][
            np.asarray(positions, dtype=np.int64)]


class DuckSamplesV2:
    """Lazy v2 sequence: ``_ds.startup`` fast path; counted records."""

    def __init__(self, ds: Any, *, forbid_reads: bool = False) -> None:
        self._ds = ds
        self.record_reads = 0
        self.forbid_reads = forbid_reads

    def __len__(self) -> int:
        return int(self._ds._n)

    def __getitem__(self, p: int) -> DuckRecord:
        if self.forbid_reads:
            raise AssertionError("v2 path materialised a record")
        self.record_reads += 1
        cols = self._ds._columns
        p = int(p)
        cid = self._ds._curve_ids[int(cols["curve_idx"][p])]
        direction = ("forward" if int(cols["dir01"][p]) == 0
                     else "reverse")
        return DuckRecord(cid, int(cols["node_index"][p]), direction)


def make_duck_pair(
    curve_specs: Dict[str, Dict[str, Any]],
    *,
    forbid_v2_reads: bool = False,
) -> Tuple[List[DuckRecord], DuckSamplesV2, Dict[str, Dict[str, Any]]]:
    """``(v1 record list, v2 lazy sequence, structural meta rows)``.

    ``curve_specs[cid]``: ``n_nodes``, ``ts_index``, optional
    ``endpoint_a_index``/``endpoint_b_index``/``directions``.  Both arms
    share one layout so pool equality is a real cross-path statement.
    """
    curve_ids = sorted(curve_specs)
    curve_row = {cid: i for i, cid in enumerate(curve_ids)}
    v1: List[DuckRecord] = []
    cols: Dict[str, List[Any]] = {"curve_idx": [], "node_index": [],
                                  "dir01": []}
    meta_rows: Dict[str, Dict[str, Any]] = {}
    for cid in curve_ids:
        spec = curve_specs[cid]
        n_nodes = int(spec["n_nodes"])
        meta_rows[cid] = {
            "n_nodes": n_nodes,
            "ts_index": int(spec["ts_index"]),
            "endpoint_a_index": int(spec.get("endpoint_a_index", 0)),
            "endpoint_b_index": int(
                spec.get("endpoint_b_index", n_nodes - 1)),
        }
        wanted = tuple(spec.get("directions", ("forward", "reverse")))
        for node in range(n_nodes):
            for direction in ("forward", "reverse"):
                if direction not in wanted:
                    continue
                v1.append(DuckRecord(cid, node, direction))
                cols["curve_idx"].append(curve_row[cid])
                cols["node_index"].append(node)
                cols["dir01"].append(0 if direction == "forward" else 1)
    arrays = {
        "curve_idx": np.asarray(cols["curve_idx"], dtype=np.int32),
        "node_index": np.asarray(cols["node_index"], dtype=np.int32),
        "dir01": np.asarray(cols["dir01"], dtype=np.int8),
    }
    ds = SimpleNamespace(_columns=arrays, _curve_ids=curve_ids,
                         _curves=meta_rows, _n=len(v1))
    ds.startup = DuckStartup(ds)
    v2 = DuckSamplesV2(ds, forbid_reads=forbid_v2_reads)
    return v1, v2, meta_rows


def duck_specs_full(n_curves: int = 12, n: int = 6, ts: int = 2
                    ) -> Dict[str, Dict[str, Any]]:
    """All four strata present in both directions."""
    return {f"cv_{c:02d}": {"n_nodes": n, "ts_index": ts}
            for c in range(n_curves)}


def duck_specs_degenerate(n_curves: int = 12
                          ) -> Dict[str, Dict[str, Any]]:
    """n=4, ts=2 → forward post_TS is EMPTY on every curve."""
    return {f"cv_{c:02d}": {"n_nodes": 4, "ts_index": 2}
            for c in range(n_curves)}


def duck_specs_forward_only(n_curves: int = 3
                            ) -> Dict[str, Dict[str, Any]]:
    """Curves carrying forward samples only → reverse pools missing."""
    return {f"cv_{c:02d}": {"n_nodes": 6, "ts_index": 2,
                            "directions": ("forward",)}
            for c in range(n_curves)}


@pytest.fixture(autouse=True)
def columnar_fast_on():
    """Every test runs with the startup fast path ON unless it flips it."""
    tr._COLUMNAR_FAST = True
    yield
    tr._COLUMNAR_FAST = True


# ---------------------------------------------------------------------------
# spec + rule SHA
# ---------------------------------------------------------------------------
def test_samplers_spec_frozen_fields() -> None:
    spec = vs.SAMPLER_SPEC_V541
    assert spec["sampler_id"] == "v541_curve_stratified_v1" \
        == vs.SAMPLER_ID_V541
    assert spec["strata"]["order"] == ["start", "pre_TS", "post_TS",
                                       "endpoint"]
    assert tuple(spec["strata"]["order"]) == vs.STRATA_ORDER_V541
    assert spec["batch_shape"] == {
        "batch_curves": 8, "frames_per_curve": 4, "batch_frames": 32,
        "direction_slots_per_curve": {"forward": 2, "reverse": 2}}
    assert vs.BATCH_CURVES_V541 == 8 and vs.FRAMES_PER_CURVE_V541 == 4
    assert spec["strata"]["class_priority"] == [
        "endpoint", "start", "TS_row", "pre_TS", "post_TS"]
    assert "BOTH directions" in spec["strata"]["ts_row"]
    assert spec["strata"]["intent_mirror"]["start"] == "endpoint"
    assert spec["strata"]["intent_mirror"]["pre_TS"] == "post_TS"
    assert spec["strata"]["intent_mirror"]["ts_row_exception"]
    assert vs.MIRROR_STRATA_V541 == {
        "start": "endpoint", "endpoint": "start",
        "pre_TS": "post_TS", "post_TS": "pre_TS"}
    excluded = spec["excluded_from_strata"]["fields"]
    for field in ("label_status", "teacher", "q"):
        assert field in excluded
    assert spec["rng"]["salt"] == vs.SALT_BATCH_V541 == 0
    assert "_update_rng" in spec["rng"]["protocol"]
    assert spec["missing_direction_pool"]["code"] == \
        "SAMPLER_DIRECTION_POOL_EMPTY"
    assert spec["missing_direction_pool"]["silent_direction_rebalance"] \
        is False
    assert spec["empty_stratum"]["shrink_batch"] is False
    assert spec["equivalence"]["legacy_v53mp_sampler"].startswith(
        "NOT equivalent")
    # machine-readable: strict JSON round-trip
    assert json.loads(json.dumps(spec)) == spec


def test_rule_sha_canonical_and_stable() -> None:
    expected = hashlib.sha256(json.dumps(
        vs.SAMPLER_SPEC_V541, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    assert vs.sampler_spec_sha256() == expected
    assert len(vs.sampler_spec_sha256()) == 64
    assert vs.sampler_spec_sha256() == vs.sampler_spec_sha256()


# ---------------------------------------------------------------------------
# classification rule
# ---------------------------------------------------------------------------
def _meta(cid: str, n: int, ts: int, a: Any = 0,
          b: Any = None) -> vs.CurveStrataMeta:
    return vs.curve_strata_meta(cid, {
        "n_nodes": n, "ts_index": ts,
        "endpoint_a_index": a,
        "endpoint_b_index": (n - 1 if b is None else b)})


def test_classify_canonical_layout() -> None:
    meta = _meta("cv", 6, 2)
    assert not meta.path_desc
    assert (meta.fwd_start_node, meta.fwd_end_node) == (0, 5)
    forward = {node: vs.classify_stratum_v541(node, "forward", meta)
               for node in range(6)}
    reverse = {node: vs.classify_stratum_v541(node, "reverse", meta)
               for node in range(6)}
    assert forward == {0: "start", 1: "pre_TS", 2: "pre_TS",
                       3: "post_TS", 4: "post_TS", 5: "endpoint"}
    assert reverse == {5: "start", 4: "pre_TS", 3: "pre_TS",
                       2: "pre_TS", 1: "post_TS", 0: "endpoint"}
    for table in (forward, reverse):
        for node, stratum in table.items():
            assert stratum in vs.STRATA_ORDER_V541, node


def test_classify_descending_path_layout() -> None:
    """endpoint_a_index > endpoint_b_index flips the canonical path.

    Uses the v1 CSV-style STRING row: ``"0"`` is truthy, so the verbatim
    ``curve_endpoints`` expression yields 0 → a=5 > b=0 → descending.
    The v2 idx INT row with ``b=0`` instead reproduces that expression's
    falsy-default quirk (0 → n_nodes-1 → ascending) — pinned below, since
    strata must follow the SAME node_order flip the labels used on the
    same loader.
    """
    meta = _meta("cv", 6, 2, a="5", b="0")
    assert meta.path_desc
    assert (meta.fwd_start_node, meta.fwd_end_node) == (5, 0)
    forward = {node: vs.classify_stratum_v541(node, "forward", meta)
               for node in range(6)}
    reverse = {node: vs.classify_stratum_v541(node, "reverse", meta)
               for node in range(6)}
    assert forward == {5: "start", 4: "pre_TS", 3: "pre_TS",
                       2: "pre_TS", 1: "post_TS", 0: "endpoint"}
    assert reverse == {0: "start", 1: "pre_TS", 2: "pre_TS",
                       3: "post_TS", 4: "post_TS", 5: "endpoint"}
    # loader-style rows: int b=0 hits the falsy-default quirk (ascending)
    quirk = _meta("cv", 6, 2, a=5, b=0)
    assert not quirk.path_desc and quirk.fwd_start_node == 0
    # int b=1 with a=5 genuinely descends on the columnar style as well
    int_desc = _meta("cv", 6, 2, a=5, b=1)
    assert int_desc.path_desc and int_desc.fwd_start_node == 5


def test_classify_priority_short_and_degenerate() -> None:
    # TS at the forward endpoint: endpoint wins over the TS row
    meta = _meta("cv", 3, 2)
    assert vs.classify_stratum_v541(2, "forward", meta) == "endpoint"
    # TS at the forward start: start wins over the TS row
    meta = _meta("cv", 3, 0)
    assert vs.classify_stratum_v541(0, "forward", meta) == "start"
    # 2-node curve, TS at node 1: fwd node1 endpoint / rev node1 start
    meta = _meta("cv", 2, 1)
    assert vs.classify_stratum_v541(1, "forward", meta) == "endpoint"
    assert vs.classify_stratum_v541(1, "reverse", meta) == "start"
    assert vs.classify_stratum_v541(0, "forward", meta) == "start"
    assert vs.classify_stratum_v541(0, "reverse", meta) == "endpoint"
    # single-node curve: start == TS == target → endpoint (priority 1)
    meta = _meta("cv", 1, 0)
    for direction in ("forward", "reverse"):
        assert vs.classify_stratum_v541(0, direction, meta) == "endpoint"
    # every node of every short curve lands in exactly ONE declared stratum
    for n in (1, 2, 3, 4):
        for ts in range(n):
            meta = _meta("cv", n, ts)
            for direction in ("forward", "reverse"):
                got = [vs.classify_stratum_v541(node, direction, meta)
                       for node in range(n)]
                assert len(set(got)) >= 1
                for stratum in got:
                    assert stratum in vs.STRATA_ORDER_V541


def test_reverse_intent_mirror_with_ts_exception() -> None:
    """Forward/reverse mirror on EVERY meta × node; TS row frozen."""
    for n in range(2, 8):
        for ts in range(n):
            # ascending (int defaults) and descending (CSV-string row)
            for a, b in ((0, n - 1), (str(n - 1), "0")):
                meta = _meta("cv", n, ts, a=a, b=b)
                for node in range(n):
                    fwd = vs.classify_stratum_v541(node, "forward", meta)
                    rev = vs.classify_stratum_v541(node, "reverse", meta)
                    if (node == meta.ts_index
                            and node not in (meta.fwd_start_node,
                                             meta.fwd_end_node)):
                        # frozen exception: a MIDDLE TS row is pre_TS for
                        # BOTH directions (the one non-mirrored case; a
                        # TS row AT start/target obeys endpoint > start
                        # priority and still mirrors start<->endpoint)
                        assert (fwd, rev) == ("pre_TS", "pre_TS"), \
                            (n, ts, a, b, node)
                    else:
                        assert rev == vs.MIRROR_STRATA_V541[fwd], \
                            (n, ts, a, b, node, fwd, rev)


def test_classify_typed_refusals() -> None:
    meta = _meta("cv", 6, 2)
    with pytest.raises(vs.V541StrataError) as exc:
        vs.classify_stratum_v541(9, "forward", meta)
    assert exc.value.code == "STRATA_NODE_RANGE"
    with pytest.raises(vs.V541StrataError) as exc:
        vs.classify_stratum_v541(1, "sideways", meta)
    assert exc.value.code == "STRATA_DIRECTION_INVALID"
    # unreachable-segment guard (meta that is structurally inconsistent)
    weird = vs.CurveStrataMeta(curve_id="cv", n_nodes=6, ts_index=2,
                               fwd_start_node=4, fwd_end_node=5,
                               path_desc=True)
    with pytest.raises(vs.V541StrataError) as exc:
        vs.classify_stratum_v541(0, "forward", weird)
    assert exc.value.code == "STRATA_UNCLASSIFIED"
    with pytest.raises(vs.V541StrataError) as exc:
        vs.curve_strata_meta("cv", {"n_nodes": 6, "ts_index": 9})
    assert exc.value.code == "STRATA_META_INVALID"


# ---------------------------------------------------------------------------
# index build: exactly-one, v1/v2 equivalence, fast-path discipline
# ---------------------------------------------------------------------------
def _all_positions(index: vs.StrataIndex) -> List[int]:
    out: List[int] = []
    for key in sorted(index.pools):
        out.extend(index.pools[key])
    return out


def test_build_each_position_exactly_one_stratum() -> None:
    specs = duck_specs_full()
    specs["cv_deg"] = {"n_nodes": 3, "ts_index": 0}
    v1, v2, meta = make_duck_pair(specs)
    positions = list(range(len(v1)))
    for samples, kwargs in ((v1, {"curve_meta": meta}),
                            (v2, {})):
        index = vs.StrataIndex.build(samples, positions, **kwargs)
        assert index.n_positions == len(positions)
        pooled = _all_positions(index)
        assert sorted(pooled) == positions           # every position once
        assert len(set(pooled)) == len(pooled)       # and ONLY once
        for p in positions:
            cid, node, direction = vs.position_fields_v541(samples, p)
            stratum = vs.classify_stratum_v541(
                node, direction, index.curve_meta(cid))
            assert (cid, direction, stratum) == \
                vs.strata_of_position_v541(samples, index, p)
            assert p in index.pools[(cid, direction, stratum)]
            assert p in index.dir_pools[(cid, direction)]
        assert index.curves == sorted(specs)
        assert len(index.dir_pools) == 2 * len(specs)
    # direction pools are exactly the sorted union of the strata pools
    index = vs.StrataIndex.build(v1, positions, curve_meta=meta)
    for cid in index.curves:
        for direction in vs.DIRECTIONS_V541:
            union = sorted(
                p for s in vs.STRATA_ORDER_V541
                for p in index.pools.get((cid, direction, s), []))
            assert index.dir_pools[(cid, direction)] == union


def test_pool_construction_v1_v2_equivalent() -> None:
    specs = duck_specs_full(n_curves=5)
    v1, v2, meta = make_duck_pair(specs)
    for positions in (list(range(len(v1))),
                      [0, 2, 5, 7, 11, 15, 20, 57],
                      []):
        ref = vs.StrataIndex.build(v1, positions, curve_meta=meta)
        got = vs.StrataIndex.build(v2, positions)
        assert got.curves == ref.curves
        assert {k: list(v) for k, v in got.pools.items()} == \
            {k: list(v) for k, v in ref.pools.items()}
        assert {k: list(v) for k, v in got.dir_pools.items()} == \
            {k: list(v) for k, v in ref.dir_pools.items()}
        assert got.pool_sha256 == ref.pool_sha256
        assert got.n_positions == ref.n_positions
    # pools are canonical: independent of the caller's indices ORDER
    shuffled = [57, 0, 15, 2, 11, 20, 7]
    sub_a = vs.StrataIndex.build(v1, shuffled, curve_meta=meta)
    sub_b = vs.StrataIndex.build(v1, sorted(shuffled), curve_meta=meta)
    assert sub_a.pools == sub_b.pools
    assert sub_a.pool_sha256 == sub_b.pool_sha256


def test_fast_path_never_materialises_records() -> None:
    specs = duck_specs_full(n_curves=4)
    v1, v2, meta = make_duck_pair(specs, forbid_v2_reads=True)
    positions = list(range(len(v2)))
    index = vs.StrataIndex.build(v2, positions)   # would raise if a record
    assert v2.record_reads == 0                   #   was constructed
    # probes use the columnar arm too
    for p in positions:
        vs.position_fields_v541(v2, p)
        vs.strata_of_position_v541(v2, index, p)
    assert v2.record_reads == 0
    # ... and the draws only consume pools
    batch = vs.draw_curve_batch_v541(
        index, rng=vs.batch_rng_v541(17, 1))
    assert len(batch.positions) == 32
    assert v2.record_reads == 0
    # explicit curve_meta never falls back to records either
    vs.StrataIndex.build(v2, positions, curve_meta=meta)
    assert v2.record_reads == 0


def test_flag_off_per_item_arm_equals_fast() -> None:
    """V53MP_DISABLE_COLUMNAR_FAST-style flag: reference arm agrees."""
    specs = duck_specs_full(n_curves=6)
    v1, v2, meta = make_duck_pair(specs)
    positions = list(range(len(v2)))
    fast = vs.StrataIndex.build(v2, positions)
    assert v2.record_reads == 0
    tr._COLUMNAR_FAST = False
    try:
        slow = vs.StrataIndex.build(v2, positions)     # per-item records
    finally:
        tr._COLUMNAR_FAST = True
    assert v2.record_reads == len(positions)
    assert slow.curves == fast.curves
    assert slow.pools == fast.pools
    assert slow.dir_pools == fast.dir_pools
    assert slow.pool_sha256 == fast.pool_sha256
    # and it equals the v1 arm (cross-loader, like the startup certificate)
    v1_ref = vs.StrataIndex.build(v1, positions, curve_meta=meta)
    assert v1_ref.pools == fast.pools


# ---------------------------------------------------------------------------
# metadata sources + typed refusals
# ---------------------------------------------------------------------------
def test_meta_resolution_chain_and_typed_refusals() -> None:
    specs = duck_specs_full(n_curves=3)
    v1, v2, meta = make_duck_pair(specs)
    positions = list(range(len(v1)))

    # (1) v1 list, no dataset/curve_meta → source missing
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v1, positions)
    assert exc.value.code == "STRATA_META_SOURCE_MISSING"

    # (2) dataset chain (structural _curves table)
    dataset = SimpleNamespace(_curves=meta)
    index = vs.StrataIndex.build(v1, positions, dataset=dataset)
    assert index.curves == sorted(specs)

    # (2b) dataset without a structural table → source missing
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v1, positions, dataset=SimpleNamespace())
    assert exc.value.code == "STRATA_META_SOURCE_MISSING"

    # (3) startup columnar chain (v2) works with no extra arguments
    index = vs.StrataIndex.build(v2, positions)
    assert index.curves == sorted(specs)

    # (3b) explicit curve_meta WINS over the startup table
    flipped = dict(meta)
    # CSV-string endpoints genuinely flip the path (see descending test)
    flipped["cv_00"] = {"n_nodes": 6, "ts_index": 2,
                        "endpoint_a_index": "5", "endpoint_b_index": "0"}
    node1_fwd = None
    for p, rec in enumerate(v1):
        if rec.curve_id == "cv_00" and rec.node_index == 1 \
                and rec.intent.direction == "forward":
            node1_fwd = p
            break
    assert node1_fwd is not None
    base = vs.strata_of_position_v541(
        v2, vs.StrataIndex.build(v2, positions), node1_fwd)
    other = vs.strata_of_position_v541(
        v2, vs.StrataIndex.build(v2, positions, curve_meta=flipped),
        node1_fwd)
    assert base[2] == "pre_TS" and other[2] == "post_TS"

    # (4) a curve absent from the structural table refuses
    partial = {k: v for k, v in meta.items() if k != "cv_02"}
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v1, positions, curve_meta=partial)
    assert exc.value.code == "STRATA_CURVE_META_MISSING"

    # (5) invalid structural row refuses
    bad = dict(meta)
    bad["cv_01"] = {"n_nodes": 6, "ts_index": 99}
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v1, positions, curve_meta=bad)
    assert exc.value.code == "STRATA_META_INVALID"

    # (6) position range refuses (both ends)
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v1, [-1, 0], curve_meta=meta)
    assert exc.value.code == "STRATA_POSITION_RANGE"
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v1, [0, len(v1)], curve_meta=meta)
    assert exc.value.code == "STRATA_POSITION_RANGE"

    # (7) node out of the curve's range refuses on the FAST path
    _v1bad, v2_bad, _m = make_duck_pair(duck_specs_full(n_curves=2))
    v2_bad._ds._columns["node_index"][3] = 99
    with pytest.raises(vs.V541StrataError) as exc:
        vs.StrataIndex.build(v2_bad, list(range(len(v2_bad))))
    assert exc.value.code == "STRATA_NODE_RANGE"

    # (8) empty indices → empty index, no meta source needed
    empty = vs.StrataIndex.build(v1, [])
    assert empty.curves == [] and empty.pools == {}
    assert empty.n_positions == 0 and empty.pool_sha256 == \
        vs._pool_sha256([], {})


# ---------------------------------------------------------------------------
# no label import / no label access
# ---------------------------------------------------------------------------
def test_no_label_import_or_access() -> None:
    source = Path(vs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # (a) no direct import of label/teacher machinery
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for module in imported:
        low = module.lower()
        assert "label" not in low, module
        assert "teacher" not in low, module
    for banned in ("key_atom_label_cache_v53mp", "key_atom_endpoint_labels_v1",
                   "key_atom_teacher", "key_atom_v52_train_v3"):
        assert not any(banned in m for m in imported), banned

    # (b) no attribute/name ACCESS of label-side fields (strings in the
    #     frozen spec may NAME the excluded fields — that is a declaration)
    forbidden = set(FORBIDDEN_LABEL_NAMES) - {"q"}
    forbidden |= {"labels_v52", "endpoint_labels", "by_sample"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, node.attr
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden, node.id

    # (c) runtime: poison records — build, probe and draw must not touch
    v1, _v2, meta = make_duck_pair(duck_specs_full(n_curves=4))
    assert v1[0].curve_id and v1[0].node_index == 0     # structural reads
    with pytest.raises(AssertionError):
        _ = v1[0].label_status                            # poison is live
    index = vs.StrataIndex.build(v1, list(range(len(v1))),
                                 curve_meta=meta)
    for p in range(len(v1)):
        vs.strata_of_position_v541(v1, index, p)
    batch = vs.draw_curve_batch_v541(index, rng=vs.batch_rng_v541(7, 3))
    assert len(batch.positions) == 32                    # no AssertionError


# ---------------------------------------------------------------------------
# rebuild discipline (double-buffer prefetch replays the index per update)
# ---------------------------------------------------------------------------
def test_index_rebuild_identical_and_draws_replay() -> None:
    specs = duck_specs_full(n_curves=9)
    v1, _v2, meta = make_duck_pair(specs)
    positions = list(range(len(v1)))
    index_a = vs.StrataIndex.build(v1, positions, curve_meta=meta)
    index_b = vs.StrataIndex.build(v1, positions, curve_meta=meta)
    assert index_a.pools == index_b.pools
    assert index_a.dir_pools == index_b.dir_pools
    assert index_a.pool_sha256 == index_b.pool_sha256
    n_updates = 20
    for update in range(1, n_updates + 1):
        prebuilt = vs.draw_curve_batch_v541(
            index_a, rng=vs.batch_rng_v541(17, update))
        rebuilt = vs.StrataIndex.build(v1, positions, curve_meta=meta)
        replay = vs.draw_curve_batch_v541(
            rebuilt, rng=vs.batch_rng_v541(17, update))
        assert replay.positions == prebuilt.positions
        assert replay.chosen_curves == prebuilt.chosen_curves
        assert replay.fallback_counts == prebuilt.fallback_counts


# ---------------------------------------------------------------------------
# real loader: v1 dataset vs columnar fast loader, declared layout
# ---------------------------------------------------------------------------
def test_real_loader_layout_and_v1_v2_equivalence(tmp_path: Path) -> None:
    from tests.pesn.test_r3_v53mp_dataset import (  # noqa: E402
        ALL_STATUSES, CURVE_IDS, N_FRAMES, TS_INDEX, _write_fixture,
    )
    from tests.pesn.test_r3_v53mp_geometry_pack import (  # noqa: E402
        _build_stack,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
        ColumnarKeyAtomDatasetV3,
    )

    tables = _write_fixture(tmp_path / "data")
    stack = _build_stack(tables)
    v1 = stack["v1"]
    v2 = ColumnarKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    positions = list(range(len(v1.samples)))
    assert len(positions) == len(CURVE_IDS) * N_FRAMES * 2
    assert (N_FRAMES, TS_INDEX) == (4, 2)   # fixture layout the pins below use

    # poison the real record gate: the fast arm must never construct one
    def _boom(row: int) -> Any:
        raise AssertionError("columnar fast arm materialised a record")
    v2._record_at = _boom

    idx_fast = vs.StrataIndex.build(v2.samples, positions)     # startup
    assert v2._record_at is _boom                             # still unused
    idx_v1 = vs.StrataIndex.build(v1.samples, positions, dataset=v1)

    assert idx_fast.curves == idx_v1.curves == sorted(CURVE_IDS)
    assert idx_fast.pools == idx_v1.pools
    assert idx_fast.dir_pools == idx_v1.dir_pools
    assert idx_fast.pool_sha256 == idx_v1.pool_sha256
    assert idx_fast.n_positions == len(positions)

    # declared layout on REAL records: n=4, ts=2, endpoints (0, 3)
    for cid in CURVE_IDS:
        expected: Dict[Tuple[int, str], str] = {
            (0, "forward"): "start", (1, "forward"): "pre_TS",
            (2, "forward"): "pre_TS", (3, "forward"): "endpoint",
            (3, "reverse"): "start", (2, "reverse"): "pre_TS",
            (1, "reverse"): "post_TS", (0, "reverse"): "endpoint",
        }
        for p in positions:
            rec = v1.samples[p]
            if str(rec.curve_id) != cid:
                continue
            got = vs.strata_of_position_v541(v2.samples, idx_fast, p)
            assert got == (cid, str(rec.intent.direction),
                           expected[(int(rec.node_index),
                                     str(rec.intent.direction))]), p
        # forward post_TS does not exist on this short curve (n=4, ts=2)
        assert ("post_TS" not in [
            s for (c, d, s) in idx_v1.pools
            if c == cid and d == "forward"])
        fwd_pre = idx_v1.pools[(cid, "forward", "pre_TS")]
        assert sorted(fwd_pre) == sorted(
            p for p in positions
            if str(v1.samples[p].curve_id) == cid
            and int(v1.samples[p].node_index) in (1, 2)
            and str(v1.samples[p].intent.direction) == "forward")
        assert len(idx_v1.dir_pools[(cid, "forward")]) == N_FRAMES
        assert len(idx_v1.dir_pools[(cid, "reverse")]) == N_FRAMES

    # legacy discipline cross-check: the old index still ignores strata
    legacy = tr._CurveDirIndex.build(v1.samples, positions)
    assert set(legacy.pools.keys()) == {
        (c, d) for c in CURVE_IDS for d in ("forward", "reverse")}
    assert legacy.curves == idx_v1.curves
