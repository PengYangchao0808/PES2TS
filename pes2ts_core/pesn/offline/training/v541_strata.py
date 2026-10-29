"""V5.4.1 WP4a — frozen curve-stratified sampler (path-A 方案四 分层兑现).

Implements the sampling contract declared by the frozen V5.4.1 config
(``config/campaigns/pes2ts_v541_embed_50k.yaml`` → ``training_contract.batch
= {curves: 8, frames: 4, strata: [start, pre_TS, post_TS, endpoint],
sampler: (seed, update, salt) 纯函数协议}``) as machine-readable
:data:`SAMPLER_SPEC_V541`, sampler_id **``v541_curve_stratified_v1``**.

Frozen rulings baked into this module (WP4a, path A; ADR-0032 §79 risk):

* **legacy NOT equivalent** — the historical V53MP
  ``_curve_batch_positions``/``_CurveDirIndex`` pool only ``(curve,
  direction)``; they do NOT implement the declared four strata.  The two are
  not interchangeable: where the WP5 audit shows they diverge, the §8.1
  "sampler/RNG equivalence failure → retrain E0×3" branch fires.  Those
  legacy functions stay UNCHANGED here — only their pure-function discipline
  (build once per run, seeded per-update draws, typed refusals) is mirrored,
  and the legacy ``_update_rng(seed, update, salt)`` is *imported and
  reused*, never re-implemented.
* **strata** are exactly ``(start, pre_TS, post_TS, endpoint)`` and are
  **intent-relative**: ``start`` = the intent's start node, ``endpoint`` =
  the intent's target node; the TS row maps to ``pre_TS`` under BOTH
  directions; short/degenerate curves resolve by the fixed priority
  ``endpoint > start > TS_row > pre_TS > post_TS``; every position belongs
  to exactly one stratum.  Forward/reverse intents mirror each other
  (``start↔endpoint``, ``pre_TS↔post_TS``) with the single frozen exception
  that the TS row stays ``pre_TS`` for both directions.
* **no label inputs** — stratification uses ONLY offline structural
  metadata (curve ``ts_index``/endpoints/``n_nodes``, sample
  ``node_index``, intent ``direction``).  ``label_status`` / teacher /
  ``q`` never participate, and no forward-view key is added.
* **batch shape** — per update: draw 8 curves with the reused
  ``_update_rng(seed, update, salt=0)``; per chosen curve the SAME rng
  picks 2 of the 4 strata (without replacement) for forward and the other
  2 for reverse (anti-confounding: the stratum→direction assignment is
  redrawn every curve/update, never fixed); then 1 frame per assigned
  stratum in fixed strata order, forward slots before reverse slots —
  always 4 frames per curve, direction balance always 2+2, batch always
  8×4 = 32 frames.
* **empty stratum** → seeded fallback draw from the SAME curve's full pool
  of the ASSIGNED direction, counted in ``fallback_counts``; the batch
  never shrinks.  A wholly missing direction pool is a typed refusal
  (``SAMPLER_DIRECTION_POOL_EMPTY``) — silently borrowing the other
  direction (the legacy ``or forward or reverse`` trick) is forbidden.
* **scale discipline** — the build is one pass over the position list;
  on the v2 loader it reads the startup columnar columns and NEVER
  materialises a single record or label row (6M-sample pools stay
  position lists only).  Pools are canonical ASCENDING position lists and
  the curve list is STRING-sorted, so results are independent of the
  caller's ``indices`` ordering and identical across loaders/processes.

Scope: sampler/index/hash machinery only — no trainer, no runner, no
training, no dev/confirm access.
"""

from __future__ import annotations

import collections.abc as _cabc
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Reuse the legacy V53MP per-update RNG and startup fast-path detector
# (imported, never copied — the (seed, update, salt) protocol is frozen):
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
    _startup_of,
    _update_rng,
)

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

SAMPLER_ID_V541: Final[str] = "v541_curve_stratified_v1"

#: Fixed stratum order — every iteration in this module follows it.
STRATA_ORDER_V541: Final[Tuple[str, ...]] = (
    "start", "pre_TS", "post_TS", "endpoint")

#: Frozen intent-mirror map (TS row exception documented in the spec).
MIRROR_STRATA_V541: Final[Dict[str, str]] = {
    "start": "endpoint",
    "endpoint": "start",
    "pre_TS": "post_TS",
    "post_TS": "pre_TS",
}

DIRECTIONS_V541: Final[Tuple[str, ...]] = ("forward", "reverse")
BATCH_CURVES_V541: Final[int] = 8
FRAMES_PER_CURVE_V541: Final[int] = 4
#: Salt of the batch (curve/stratum/frame) draws — legacy V53MP used 0.
SALT_BATCH_V541: Final[int] = 0

#: Identity byte label of the schedule hash chain domain.
SCHEDULE_CHAIN_LABEL_V541: Final[bytes] = (
    b"v541_curve_stratified_v1|batch_schedule_chain|v1")
#: Identity byte label of the pool hash domain.
POOL_HASH_LABEL_V541: Final[bytes] = b"pesn_v541_strata_pool_v1|"

_STRATUM_CODE: Final[Dict[str, int]] = {
    name: i for i, name in enumerate(STRATA_ORDER_V541)}
_STRATUM_NAMES: Final[Tuple[str, ...]] = STRATA_ORDER_V541
_DIRECTION_CODE: Final[Dict[str, int]] = {"forward": 0, "reverse": 1}


class V541StrataError(RuntimeError):
    """Typed refusal of the V5.4.1 stratified index/sampler."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V541StrataError(code, detail)


# ---------------------------------------------------------------------------
# machine-readable frozen spec (规则 SHA input)
# ---------------------------------------------------------------------------

#: THE frozen sampler contract.  Editing this dict is a contract change:
#: it invalidates ``sampler_spec_sha256()`` (the rule SHA recorded in
#: manifests) and therefore every run identity built on top of it.
SAMPLER_SPEC_V541: Final[Dict[str, Any]] = {
    "schema": "pesn_v541_sampler_spec_v1",
    "sampler_id": SAMPLER_ID_V541,
    "status": "frozen_wp4a_2026_10_27",
    "config_binding": {
        "config_path": (
            "config/campaigns/pes2ts_v541_embed_50k.yaml#"
            "training_contract.batch"),
        "curves": 8,
        "frames": 4,
        "strata": ["start", "pre_TS", "post_TS", "endpoint"],
        "sampler": "(seed, update, salt) 纯函数协议",
    },
    "batch_shape": {
        "batch_curves": 8,
        "frames_per_curve": 4,
        "batch_frames": 32,
        "direction_slots_per_curve": {"forward": 2, "reverse": 2},
    },
    "strata": {
        "order": ["start", "pre_TS", "post_TS", "endpoint"],
        "reference_frame": (
            "intent-relative: start = intent start node, endpoint = intent "
            "target node; frames ordered from the intent's start toward its "
            "target along the canonical curve path"),
        "path_order": (
            "ascending node_index iff endpoint_a_index <= "
            "endpoint_b_index else descending — mirrors the "
            "build_v52_labels node_order construction (incl. its "
            "endpoint_b_index default quirk, replicated verbatim)"),
        "ts_row": ("TS row -> pre_TS under BOTH directions (subordinate to "
                   "the endpoint/start priority when the TS sits on an "
                   "endpoint)"),
        "class_priority": ["endpoint", "start", "TS_row", "pre_TS",
                           "post_TS"],
        "priority_note": (
            "short/degenerate curves resolve by this fixed priority; each "
            "position belongs to exactly one stratum"),
        "intent_mirror": {
            "start": "endpoint", "endpoint": "start",
            "pre_TS": "post_TS", "post_TS": "pre_TS",
            "ts_row_exception": (
                "TS row stays pre_TS for both directions — the one frozen "
                "non-mirrored case"),
        },
    },
    "structural_metadata_only": [
        "curve ts_index", "curve endpoint_a_index", "curve endpoint_b_index",
        "curve n_nodes", "sample node_index", "intent direction",
    ],
    "excluded_from_strata": {
        "fields": ["label_status", "teacher", "q", "label tables",
                   "endpoint labels", "q/stop/weights"],
        "note": (
            "strata use ONLY the structural metadata listed above; no new "
            "forward-view key is introduced; caller-side pool filtering "
            "(label_status/fold/confirm) stays upstream of this index"),
    },
    "rng": {
        "protocol": (
            "key_atom_v53mp_train_v3._update_rng(seed, update, salt) — "
            "imported and reused, never re-implemented"),
        "salt": 0,
        "consumption_order": [
            "1. rng.choice(curves, size=8, replace=len(curves)<8)",
            "2. per chosen curve: rng.choice(strata, size=2, "
            "replace=False) -> the two forward strata",
            "3. the other two strata (fixed strata order) -> reverse",
            "4. forward slots: fixed strata order, rng.choice(pool, "
            "size=1, replace=False), one frame each",
            "5. reverse slots: fixed strata order, same one-frame draw",
            "6. empty stratum: rng.choice(same curve + assigned-direction "
            "full pool, size=1, replace=False), counted as fallback",
        ],
        "determinism": (
            "pure function of (seed, update[, salt]); identical across "
            "processes and loaders"),
        "anti_confounding": (
            "stratum->direction assignment is drawn per curve per update, "
            "so no fixed stratum x direction binding can emerge"),
    },
    "empty_stratum": {
        "action": (
            "seeded draw from the same curve's full frame pool of the "
            "ASSIGNED direction"),
        "counting": "fallback_counts per direction + total",
        "shrink_batch": False,
    },
    "missing_direction_pool": {
        "action": "typed refusal",
        "code": "SAMPLER_DIRECTION_POOL_EMPTY",
        "silent_direction_rebalance": False,
        "note": (
            "the legacy cross-direction `or forward or reverse` fallback "
            "is forbidden here; refusing keeps the 2+2 balance honest"),
    },
    "ordering": {
        "curves": "string-sorted unique curve list (legacy discipline)",
        "pools": "ascending dataset positions per pool, independent of "
                 "the caller's indices ordering",
        "batch": (
            "chosen-curve order; per curve forward slots then reverse "
            "slots; fixed strata order inside each direction"),
    },
    "hashes": {
        "rules_sha256": (
            "sampler_spec_sha256() — sha256 of the canonical JSON of this "
            "spec"),
        "pool_sha256": (
            "StrataIndex.pool_sha256 — sha256 over the curve list + every "
            "(curve, direction, stratum) pool (direction pools are the "
            "deterministic union of the four strata pools)"),
        "schedule": (
            "BatchScheduleHasher.chain_sha256 — sha256 chain over each "
            "update's batch positions; state_dict()/from_state() give "
            "checkpoint/resume continuity"),
    },
    "equivalence": {
        "legacy_v53mp_sampler": (
            "NOT equivalent: legacy pools are (curve, direction) only; "
            "path-A ruling — a declaration/implementation mismatch "
            "triggers the E0 x3 retrain branch (plan §8.1)"),
        "legacy_reference": (
            "key_atom_v53mp_train_v3._curve_batch_positions / "
            "_CurveDirIndex — discipline referenced, code unchanged"),
    },
}

_SPEC_SHA_CACHE: List[str] = []


def sampler_spec_sha256() -> str:
    """Rule SHA: sha256 over the canonical JSON of :data:`SAMPLER_SPEC_V541`."""
    if not _SPEC_SHA_CACHE:
        blob = json.dumps(
            SAMPLER_SPEC_V541, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")
        _SPEC_SHA_CACHE.append(hashlib.sha256(blob).hexdigest())
    return _SPEC_SHA_CACHE[0]


# ---------------------------------------------------------------------------
# per-curve structural meta + frozen classification rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveStrataMeta:
    """Offline structural metadata one curve needs for stratification.

    Only curve-level structure: trusted TS node, node count, and the two
    intent path ends (forward start/target; reverse mirrors them).  Nothing
    label-side ever enters this type.
    """

    curve_id: str
    n_nodes: int
    ts_index: int
    #: first node of the canonical path order (forward intent's start)
    fwd_start_node: int
    #: last node of the canonical path order (forward intent's target)
    fwd_end_node: int
    #: True when endpoint_a_index > endpoint_b_index (reversed path)
    path_desc: bool


def curve_strata_meta(curve_id: str, row: Mapping[str, Any]) -> CurveStrataMeta:
    """Normalise a loader curve row (v1 CSV strings or v2 idx ints).

    The ``endpoint_*`` expressions are replicated VERBATIM from
    ``KeyAtomDatasetV3.curve_endpoints`` (including its falsy-default
    quirk) and the ascending/descending flip mirrors
    ``build_v52_labels``' ``node_order`` construction, so strata always
    agree with the canonical path the labels were built on.
    """
    try:
        n_nodes = int(row["n_nodes"])
        ts_index = int(row["ts_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V541StrataError(
            "STRATA_META_INVALID",
            f"curve {curve_id}: structural row lacks n_nodes/ts_index"
        ) from exc
    _require(n_nodes >= 1, "STRATA_META_INVALID",
             f"curve {curve_id}: n_nodes {n_nodes} < 1")
    _require(0 <= ts_index < n_nodes, "STRATA_META_INVALID",
             f"curve {curve_id}: ts_index {ts_index} outside 0..{n_nodes - 1}")
    endpoint_a = int(row.get("endpoint_a_index", 0) or 0)
    endpoint_b = int(row.get("endpoint_b_index", "") or (n_nodes - 1))
    path_desc = endpoint_a > endpoint_b           # node_order flip rule
    if path_desc:
        fwd_start, fwd_end = n_nodes - 1, 0
    else:
        fwd_start, fwd_end = 0, n_nodes - 1
    return CurveStrataMeta(
        curve_id=str(curve_id), n_nodes=n_nodes, ts_index=ts_index,
        fwd_start_node=fwd_start, fwd_end_node=fwd_end, path_desc=path_desc)


def classify_stratum_v541(
    node_index: int,
    direction: str,
    meta: CurveStrataMeta,
) -> str:
    """ONE stratum per position — frozen priority, intent-relative.

    Priority (short/degenerate curves): ``endpoint > start > TS_row >
    pre_TS > post_TS``.  The TS row lands in ``pre_TS`` under both
    directions; segments are strict interiors of ``[start, TS]`` and
    ``[TS, target]`` in node-index space (the path is monotone in index).
    """
    node = int(node_index)
    _require(0 <= node < meta.n_nodes, "STRATA_NODE_RANGE",
             f"curve {meta.curve_id}: node {node} outside "
             f"0..{meta.n_nodes - 1}")
    if direction == "forward":
        start, target = meta.fwd_start_node, meta.fwd_end_node
    elif direction == "reverse":
        start, target = meta.fwd_end_node, meta.fwd_start_node
    else:
        raise V541StrataError(
            "STRATA_DIRECTION_INVALID",
            f"direction must be forward|reverse, got {direction!r}")
    ts = meta.ts_index
    # --- frozen priority chain -------------------------------------------
    if node == target:
        return "endpoint"
    if node == start:
        return "start"
    if node == ts:
        return "pre_TS"
    if min(start, ts) < node < max(start, ts):
        return "pre_TS"
    if min(ts, target) < node < max(ts, target):
        return "post_TS"
    raise V541StrataError(
        "STRATA_UNCLASSIFIED",
        f"curve {meta.curve_id}: node {node} ({direction}) matches no "
        "stratum segment")


# ---------------------------------------------------------------------------
# StrataIndex
# ---------------------------------------------------------------------------


class StrataIndex:
    """Prebuilt ``(curve, direction, stratum)`` pools over the train pool.

    Built ONCE per run (legacy ``_CurveDirIndex`` discipline): one pass
    over ``indices`` — vectorised on the v2 loader's startup columns
    (NO record or label materialisation, required at the 6M-sample tier),
    per-item on the v1 tuple (the reference arm).  Both arms produce
    bit-identical containers: STRING-sorted curve list, ASCENDING
    position list per pool (canonical, independent of the caller's
    ``indices`` order), direction pools as the ascending union of the
    four strata pools.
    """

    __slots__ = ("curves", "pools", "dir_pools", "meta",
                 "pool_sha256", "n_positions")

    def __init__(
        self,
        curves: List[str],
        pools: Dict[Tuple[str, str, str], List[int]],
        dir_pools: Dict[Tuple[str, str], List[int]],
        meta: Dict[str, CurveStrataMeta],
        pool_sha256: str,
        n_positions: int,
    ) -> None:
        self.curves = curves
        self.pools = pools
        self.dir_pools = dir_pools
        self.meta = meta
        self.pool_sha256 = pool_sha256
        self.n_positions = int(n_positions)

    # -- lookups -----------------------------------------------------------
    def curve_meta(self, curve_id: str) -> CurveStrataMeta:
        meta = self.meta.get(str(curve_id))
        if meta is None:
            raise V541StrataError(
                "STRATA_CURVE_META_MISSING",
                f"curve {curve_id!r} is not covered by this index")
        return meta

    # -- build -------------------------------------------------------------
    @classmethod
    def build(
        cls,
        samples: Sequence[Any],
        indices: Sequence[int],
        *,
        dataset: Any = None,
        curve_meta: Optional[_cabc.Mapping[str, Any]] = None,
    ) -> "StrataIndex":
        """One pass over ``indices`` → pools (+ direction pools + SHAs).

        ``curve_meta`` (explicit structural rows), ``dataset`` (its
        ``_curves`` table) and the columnar startup ``_ds._curves`` table
        are tried in that order for the per-curve structural metadata;
        with none available the build types-refuses.  ``indices`` is
        sorted first so the pools — and therefore every seeded draw —
        depend only on the position SET.
        """
        if isinstance(indices, np.ndarray):
            idx = np.sort(np.asarray(indices, dtype=np.int64))
        else:
            idx = np.sort(np.asarray(
                [int(i) for i in indices], dtype=np.int64))
        _require(idx.ndim == 1, "STRATA_POSITION_RANGE",
                 f"indices must be 1-D, got ndim={idx.ndim}")
        n_total = len(samples)
        if idx.size:
            _require(
                int(idx[0]) >= 0 and int(idx[-1]) < int(n_total),
                "STRATA_POSITION_RANGE",
                f"positions outside 0..{max(int(n_total) - 1, 0)}")
        st = _startup_of(samples)          # flag-governed fast-path probe
        if idx.size == 0:
            empty: Dict[Tuple[str, str, str], List[int]] = {}
            return cls([], {}, {}, {}, _pool_sha256([], empty), 0)

        rows = _resolve_meta_rows(samples, dataset=dataset,
                                  curve_meta=curve_meta)
        meta_cache: Dict[str, CurveStrataMeta] = {}

        def meta_for(curve_id: str) -> CurveStrataMeta:
            cached = meta_cache.get(curve_id)
            if cached is not None:
                return cached
            raw = rows.get(curve_id)
            _require(raw is not None, "STRATA_CURVE_META_MISSING",
                     f"curve {curve_id} missing from the structural table")
            if isinstance(raw, CurveStrataMeta):
                meta = raw
            elif isinstance(raw, _cabc.Mapping):
                meta = curve_strata_meta(curve_id, raw)
            else:
                raise V541StrataError(
                    "STRATA_META_INVALID",
                    f"curve {curve_id}: meta entry must be CurveStrataMeta "
                    "or a structural row mapping")
            meta_cache[curve_id] = meta
            return meta

        node_col = None
        if st is not None:
            cols = getattr(getattr(st, "_ds", None), "_columns", None)
            if isinstance(cols, _cabc.Mapping):
                node_col = cols.get("node_index")
        if (st is not None and node_col is not None
                and hasattr(st, "curve_idx") and hasattr(st, "curve_ids")
                and hasattr(st, "direction_int")):
            pools = _build_pools_fast(st, node_col, idx, meta_for)
        else:
            pools = _build_pools_per_item(samples, idx, meta_for)

        curves = sorted({cid for cid, _d, _s in pools})
        dir_pools = _merge_dir_pools(pools, curves)
        return cls(
            curves, pools, dir_pools, meta_cache,
            _pool_sha256(curves, pools), int(idx.size))


def _resolve_meta_rows(
    samples: Any,
    *,
    dataset: Any,
    curve_meta: Optional[_cabc.Mapping[str, Any]],
) -> _cabc.Mapping[str, Any]:
    """Structural table source: explicit arg > dataset > startup ``_ds``."""
    if curve_meta is not None:
        _require(isinstance(curve_meta, _cabc.Mapping), "STRATA_META_INVALID",
                 "curve_meta must be a mapping curve_id -> structural row")
        return curve_meta
    if dataset is not None:
        rows = getattr(dataset, "_curves", None)
        _require(isinstance(rows, _cabc.Mapping), "STRATA_META_SOURCE_MISSING",
                 "dataset exposes no _curves structural table — pass "
                 "curve_meta=")
        return rows
    # NOTE: deliberately flag-INdependent (the table is structure, not a
    # pool fast path) so the per-item reference arm needs no extra args.
    ds = getattr(samples, "_ds", None)
    rows = getattr(ds, "_curves", None)
    _require(isinstance(rows, _cabc.Mapping), "STRATA_META_SOURCE_MISSING",
             "no columnar startup curve table on samples — pass dataset= "
             "or curve_meta=")
    return rows


def _build_pools_fast(
    st: Any,
    node_col: Any,
    idx: np.ndarray,
    meta_for: Any,
) -> Dict[Tuple[str, str, str], List[int]]:
    """Vectorised pool construction over startup columns (no records)."""
    curve_ids: List[str] = list(st.curve_ids)
    ci = np.asarray(st.curve_idx)[idx]
    _require(not ci.size or int(ci.max()) < len(curve_ids),
             "STRATA_CURVE_IDX_RANGE",
             f"curve_idx column references id >= {len(curve_ids)}")
    node = np.asarray(node_col)[idx]
    dir_i = np.asarray(st.direction_int(idx), dtype=np.int64)
    bad_dir = ~np.isin(dir_i, (0, 1))
    if bool(bad_dir.any()):
        k = int(np.flatnonzero(bad_dir)[0])
        raise V541StrataError(
            "STRATA_DIRECTION_INVALID",
            f"position {int(idx[k])}: direction code {int(dir_i[k])} is "
            "not forward(0)/reverse(1)")
    code = np.empty(idx.size, dtype=np.int64)

    # one meta lookup + vectorised classification per curve group
    order_c = np.argsort(ci, kind="stable")
    bounds = np.flatnonzero(np.diff(ci[order_c])) + 1
    for group in np.split(order_c, bounds):
        head = int(group[0])
        cid = curve_ids[int(ci[head])]
        meta = meta_for(cid)
        nodes = node[group]
        dirs = dir_i[group]
        off = (nodes < 0) | (nodes >= meta.n_nodes)
        if bool(off.any()):
            k = int(np.flatnonzero(off)[0])
            raise V541StrataError(
                "STRATA_NODE_RANGE",
                f"curve {cid}: node {int(nodes[k])} outside "
                f"0..{meta.n_nodes - 1}")
        is_fwd = dirs == 0
        start = np.where(is_fwd, meta.fwd_start_node, meta.fwd_end_node)
        target = np.where(is_fwd, meta.fwd_end_node, meta.fwd_start_node)
        ts = meta.ts_index
        # frozen priority chain, vectorised (first True wins in np.select):
        # endpoint(3) > start(0) > TS_row->pre_TS(1) > pre(1) > post(2)
        sel = np.select(
            [nodes == target,
             nodes == start,
             nodes == ts,
             (nodes > np.minimum(start, ts)) & (nodes < np.maximum(start, ts)),
             (nodes > np.minimum(ts, target))
             & (nodes < np.maximum(ts, target))],
            [3, 0, 1, 1, 2],
            default=-1)
        bad = np.flatnonzero(sel < 0)
        if bad.size:
            k = int(bad[0])
            direction = "forward" if int(dirs[k]) == 0 else "reverse"
            raise V541StrataError(
                "STRATA_UNCLASSIFIED",
                f"curve {cid}: node {int(nodes[k])} ({direction}) matches "
                "no stratum segment")
        code[group] = sel

    # bucket runs by (curve, direction, stratum); idx is ASCENDING and the
    # sort is stable, so every pool comes out ascending.  NOTE: ``run``
    # holds ORIGINAL idx positions (legacy _CurveDirIndex discipline) —
    # the run's key comes from the ORIGINAL key array, not the sorted one.
    bkey = ci.astype(np.int64) * 8 + dir_i.astype(np.int64) * 4 + code
    run_order = np.argsort(bkey, kind="stable")
    keys_sorted = bkey[run_order]
    pools: Dict[Tuple[str, str, str], List[int]] = {}
    for run in np.split(run_order, np.flatnonzero(np.diff(keys_sorted)) + 1):
        k = int(bkey[int(run[0])])
        cid = curve_ids[k >> 3]
        direction = "forward" if ((k >> 2) & 1) == 0 else "reverse"
        pools[(cid, direction, _STRATUM_NAMES[k & 3])] = idx[run].tolist()
    return pools


def _build_pools_per_item(
    samples: Sequence[Any],
    idx: np.ndarray,
    meta_for: Any,
) -> Dict[Tuple[str, str, str], List[int]]:
    """Per-item reference path (v1 tuple / fast path disabled)."""
    pools: Dict[Tuple[str, str, str], List[int]] = {}
    for i in idx.tolist():
        rec = samples[int(i)]
        cid = str(rec.curve_id)
        node = int(rec.node_index)
        direction = str(rec.intent.direction)
        _require(direction in _DIRECTION_CODE, "STRATA_DIRECTION_INVALID",
                 f"position {int(i)}: direction {direction!r} is not "
                 "forward|reverse")
        stratum = classify_stratum_v541(node, direction, meta_for(cid))
        pools.setdefault((cid, direction, stratum), []).append(int(i))
    return pools


def _merge_dir_pools(
    pools: Dict[Tuple[str, str, str], List[int]],
    curves: Sequence[str],
) -> Dict[Tuple[str, str], List[int]]:
    """Full (curve, direction) pools = sorted union of the four strata."""
    out: Dict[Tuple[str, str], List[int]] = {}
    for cid in curves:
        for direction in DIRECTIONS_V541:
            parts = [pools[(cid, direction, s)]
                     for s in STRATA_ORDER_V541
                     if (cid, direction, s) in pools]
            if parts:
                merged: List[int] = []
                for part in parts:
                    merged.extend(part)
                merged.sort()
                out[(cid, direction)] = merged
    return out


def _pool_sha256(
    curves: Sequence[str],
    pools: Mapping[Tuple[str, str, str], Sequence[int]],
) -> str:
    """Pool SHA: curves + every strata pool (direction pools derive)."""
    h = hashlib.sha256()
    h.update(POOL_HASH_LABEL_V541)
    for cid in curves:
        raw = str(cid).encode("utf-8")
        h.update(len(raw).to_bytes(4, "little"))
        h.update(raw)
    for key in sorted(pools):
        cid, direction, stratum = key
        h.update(b"|")
        raw = str(cid).encode("utf-8")
        h.update(len(raw).to_bytes(4, "little"))
        h.update(raw)
        h.update(bytes((_DIRECTION_CODE[direction],
                        _STRATUM_CODE[stratum])))
        pos = np.asarray(pools[key], dtype="<i8")
        h.update(len(pos).to_bytes(8, "little"))
        h.update(pos.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# per-position probes (diagnostics / tests; fast arm, no record fallback)
# ---------------------------------------------------------------------------


def position_fields_v541(
    samples: Sequence[Any],
    position: int,
) -> Tuple[str, int, str]:
    """``(curve_id, node_index, direction)`` of ONE position.

    Uses the startup columnar columns when available (the flag-governed
    fast path), the record otherwise — probes only, never a bulk path.
    """
    p = int(position)
    st = _startup_of(samples)
    if st is not None:
        cols = getattr(getattr(st, "_ds", None), "_columns", None)
        node_col = cols.get("node_index") if isinstance(cols, _cabc.Mapping) \
            else None
        if (node_col is not None and hasattr(st, "curve_idx")
                and hasattr(st, "curve_ids")
                and hasattr(st, "direction_int")):
            cid = st.curve_ids[int(st.curve_idx[p])]
            dcode = int(np.asarray(
                st.direction_int(np.asarray([p], dtype=np.int64)))[0])
            return (str(cid), int(node_col[p]),
                    "forward" if dcode == 0 else "reverse")
    rec = samples[p]
    return str(rec.curve_id), int(rec.node_index), str(rec.intent.direction)


def strata_of_position_v541(
    samples: Sequence[Any],
    index: StrataIndex,
    position: int,
) -> Tuple[str, str, str]:
    """``(curve_id, direction, stratum)`` of ONE position (fresh rule)."""
    cid, node, direction = position_fields_v541(samples, position)
    return (cid, direction,
            classify_stratum_v541(node, direction, index.curve_meta(cid)))


# ---------------------------------------------------------------------------
# per-update draws (pure; RNG consumption order frozen in the spec)
# ---------------------------------------------------------------------------


def batch_rng_v541(
    seed: int,
    update: int,
    salt: int = SALT_BATCH_V541,
) -> np.random.RandomState:
    """The legacy ``_update_rng(seed, update, salt)`` for batch draws."""
    return _update_rng(int(seed), int(update), salt=int(salt))


@dataclass(frozen=True)
class StrataSlotV541:
    """ONE frame slot of one curve's 4-slot group."""

    curve_id: str
    direction: str          # the slot's ASSIGNED direction (2 + 2)
    stratum: str            # the slot's ASSIGNED stratum (all four cover)
    position: int           # dataset position actually drawn
    source: str             # "stratum" | "fallback"


@dataclass(frozen=True)
class StrataBatchV541:
    """One update's batch: chosen curves, slots and fallback accounting."""

    positions: Tuple[int, ...]
    slots: Tuple[StrataSlotV541, ...]
    chosen_curves: Tuple[str, ...]
    fallback_counts: Mapping[str, int]   # forward / reverse / total


def draw_curve_batch_v541(
    index: StrataIndex,
    *,
    rng: np.random.RandomState,
    batch_curves: int = BATCH_CURVES_V541,
) -> StrataBatchV541:
    """One 8×4 stratified batch — the detailed draw (positions+slots).

    RNG consumption order (frozen): choose the curves ONCE, then per
    chosen curve the stratum→direction split, then the four one-frame
    draws (forward slots first, fixed strata order; reverse slots after).
    Empty strata fall back inside the same curve/direction pool; a
    missing direction pool types-refuses (never rebalanced).
    """
    curves = index.curves
    _require(bool(curves), "SAMPLER_CURVES_EMPTY",
             "no train curves to sample")
    n_curves = int(batch_curves)
    _require(n_curves >= 1, "SAMPLER_BATCH_INVALID",
             f"batch_curves must be >= 1, got {n_curves}")
    chosen = [str(c) for c in np.atleast_1d(rng.choice(
        curves, size=n_curves, replace=len(curves) < n_curves))]
    slots: List[StrataSlotV541] = []
    fallback_counts: Dict[str, int] = {"forward": 0, "reverse": 0,
                                       "total": 0}
    for cid in chosen:
        pick = rng.choice(list(STRATA_ORDER_V541), size=2, replace=False)
        fwd_strata = {str(s) for s in np.atleast_1d(pick)}
        for direction in DIRECTIONS_V541:      # frozen: forward, then reverse
            if direction == "forward":
                assigned = [s for s in STRATA_ORDER_V541
                            if s in fwd_strata]
            else:
                assigned = [s for s in STRATA_ORDER_V541
                            if s not in fwd_strata]
            for stratum in assigned:           # fixed strata order, 1 frame
                pool = index.pools.get((cid, direction, stratum))
                if pool:
                    source = "stratum"
                    draw_from = pool
                else:
                    source = "fallback"
                    draw_from = index.dir_pools.get((cid, direction))
                    if not draw_from:
                        raise V541StrataError(
                            "SAMPLER_DIRECTION_POOL_EMPTY",
                            f"curve {cid!r} has no {direction} frames in "
                            "the train pool — typed refusal (no silent "
                            "direction rebalance)")
                    fallback_counts[direction] += 1
                    fallback_counts["total"] += 1
                position = int(np.atleast_1d(
                    rng.choice(draw_from, size=1, replace=False))[0])
                slots.append(StrataSlotV541(
                    curve_id=cid, direction=direction, stratum=stratum,
                    position=position, source=source))
    return StrataBatchV541(
        positions=tuple(s.position for s in slots),
        slots=tuple(slots),
        chosen_curves=tuple(chosen),
        fallback_counts=fallback_counts)


def curve_batch_positions_v541(
    index: StrataIndex,
    *,
    rng: np.random.RandomState,
    batch_curves: int = BATCH_CURVES_V541,
    counters: Optional[Dict[str, int]] = None,
) -> List[int]:
    """Dataset positions of one update (legacy-style thin wrapper).

    ``index`` is the prebuilt :class:`StrataIndex` (built ONCE per run);
    ``rng`` should come from :func:`batch_rng_v541`.  When ``counters``
    is given it is accumulated with this draw's ``fallback_counts``
    (runners record it; the batch shape never shrinks).
    """
    batch = draw_curve_batch_v541(
        index, rng=rng, batch_curves=batch_curves)
    if counters is not None:
        for key, value in batch.fallback_counts.items():
            counters[key] = int(counters.get(key, 0)) + int(value)
    return list(batch.positions)


# ---------------------------------------------------------------------------
# batch schedule hash chain (checkpoint / resume)
# ---------------------------------------------------------------------------


class BatchScheduleHasher:
    """sha256 chain over every update's batch positions.

    ``chain_u = sha256(chain_{u-1} | update | count | positions)`` with
    ``chain_0 = sha256(SCHEDULE_CHAIN_LABEL_V541)``.  Updates must be
    contiguous; ``state_dict()``/``from_state()`` snapshot the chain so a
    kill/resume run hashes EXACTLY like the uninterrupted one.
    """

    __slots__ = ("_chain", "_last_update", "_records")

    def __init__(self) -> None:
        self._chain = hashlib.sha256(SCHEDULE_CHAIN_LABEL_V541).digest()
        self._last_update: Optional[int] = None
        self._records = 0

    @property
    def chain_sha256(self) -> str:
        return self._chain.hex()

    @property
    def last_update(self) -> Optional[int]:
        return self._last_update

    @property
    def records(self) -> int:
        return self._records

    def record(self, update: int, positions: Sequence[int]) -> str:
        """Fold ``positions`` of ``update`` into the chain; returns hex."""
        u = int(update)
        _require(u >= 0, "SCHEDULE_UPDATE_INVALID",
                 f"update must be >= 0, got {u}")
        if self._last_update is not None:
            _require(u == self._last_update + 1, "SCHEDULE_UPDATE_NONCONTIGUOUS",
                     f"expected update {self._last_update + 1}, got {u}")
        pos = np.asarray(list(positions), dtype="<i8")
        h = hashlib.sha256()
        h.update(self._chain)
        h.update(b"|update|")
        h.update(struct.pack("<q", u))
        h.update(struct.pack("<Q", int(pos.size)))
        h.update(pos.tobytes())
        self._chain = h.digest()
        self._last_update = u
        self._records += 1
        return self.chain_sha256

    def state_dict(self) -> Dict[str, Any]:
        """JSON-safe checkpoint state (identity-checked on resume)."""
        return {
            "sampler_id": SAMPLER_ID_V541,
            "spec_sha256": sampler_spec_sha256(),
            "last_update": self._last_update,
            "chain_sha256": self.chain_sha256,
            "records": self._records,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "BatchScheduleHasher":
        """Resume from :meth:`state_dict`; identity mismatch refuses."""
        ok = isinstance(state, _cabc.Mapping) \
            and state.get("sampler_id") == SAMPLER_ID_V541 \
            and state.get("spec_sha256") == sampler_spec_sha256()
        if not ok:
            raise V541StrataError(
                "SCHEDULE_STATE_INVALID",
                "state identity does not match "
                f"{SAMPLER_ID_V541} spec {sampler_spec_sha256()}")
        chain_hex = state.get("chain_sha256")
        last = state.get("last_update")
        records = state.get("records")
        try:
            chain = bytes.fromhex(str(chain_hex))
            last_int = None if last is None else int(last)
            records_int = int(records)
        except (TypeError, ValueError) as exc:
            raise V541StrataError(
                "SCHEDULE_STATE_INVALID", "malformed schedule state"
            ) from exc
        if len(chain) != 32 or (last_int is not None and last_int < 0) \
                or records_int < 0:
            raise V541StrataError(
                "SCHEDULE_STATE_INVALID", "malformed schedule state fields")
        out = cls()
        out._chain = chain
        out._last_update = last_int
        out._records = records_int
        return out


__all__ = [
    "BATCH_CURVES_V541",
    "BatchScheduleHasher",
    "CurveStrataMeta",
    "DIRECTIONS_V541",
    "FRAMES_PER_CURVE_V541",
    "MIRROR_STRATA_V541",
    "SALT_BATCH_V541",
    "SAMPLER_ID_V541",
    "SAMPLER_SPEC_V541",
    "StrataBatchV541",
    "StrataIndex",
    "StrataSlotV541",
    "STRATA_ORDER_V541",
    "V541StrataError",
    "batch_rng_v541",
    "classify_stratum_v541",
    "curve_batch_positions_v541",
    "curve_strata_meta",
    "draw_curve_batch_v541",
    "position_fields_v541",
    "sampler_spec_sha256",
    "strata_of_position_v541",
]
