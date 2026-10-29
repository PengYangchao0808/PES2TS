"""Shared batch-pair ownership/mapping for the v3 evaluation path (WP1a).

Why this exists (V5.4.1 plan §4.1)
-----------------------------------
``collate_v3`` concatenates per-sample NODE rows along one batch-global node
axis and offsets ``pair_index``/``key_index`` by the per-graph ``n_nodes``
offsets, so ``pair_index`` columns are batch-global **node** rows.  Two eval
entry points historically attributed pairs with the **key-row** pointer
(``cumsum(n_keys)``) instead of the node-row pointer — silently dropping
legal pairs whenever ``n_nodes != n_keys`` or key atoms were not the leading
node rows of their graph.  Both entry points now call
:func:`assign_batch_pairs` from THIS module so the logic cannot drift:

* ``pes2ts_core.pesn.offline.training.key_atom_bc_v3.evaluate_checkpoint``
* ``scripts.campaigns.r3_v53mp_scale_paired_ci._collect``

Contract (plan §4.1 / V5.4.1 WP1a)
----------------------------------
* key-row ptr (``cumsum(n_keys)``) slices ONLY ``u``/``y``/``key_index``;
* node-row ptr (``cumsum(n_nodes)``) assigns each pair column to its owning
  graph window; both endpoints must fall inside ONE graph window —
  cross-graph or out-of-range endpoints raise
  :class:`PairOwnershipError` (fail-closed, never silently re-attributed);
* a pair whose endpoints are not BOTH key atoms of its graph is IGNORED  per contract and counted with an explicit reason — legal pairs are
  never silently dropped;
* emitted pairs keep batch-global node-row semantics (the ``pos`` /
  ``key_rows`` / ``pairs`` space ``evaluate_g53`` consumes) and
  ``pair_edit`` values pass through UNCHANGED (intent sign; MOVE eligibility
  and frame masks are decided UPSTREAM by ``label_status`` and are never
  consulted here);
* the audit exposes per-graph denominators (``n_active_pairs``,
  ``n_ignored_non_key`` with reason breakdown, ``n_invalid_pairs``) so
  before/after re-evals can be compared without trusting pair counts to be
  equal by construction.  ``n_invalid_pairs`` is always 0 in a RETURNED
  audit because any invalid pair raises before a result is produced — the
  typed error itself carries the offending count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "AUDIT_SCHEMA",
    "BatchPairOwnership",
    "GraphPairSlice",
    "IGNORE_REASONS",
    "PairOwnershipError",
    "assign_batch_pairs",
    "merge_pair_audits",
]

#: Schema id of the audit documents produced by :meth:`BatchPairOwnership.audit`.
AUDIT_SCHEMA = "pesn_pair_ownership_audit_v1"

#: Reasons a pair is ignored under the contract (at least one endpoint is
#: not a key atom of the owning graph).  Fixed vocabulary for auditability.
REASON_FIRST_END_NOT_KEY = "first_end_not_key"
REASON_SECOND_END_NOT_KEY = "second_end_not_key"
REASON_BOTH_ENDS_NOT_KEY = "both_ends_not_key"
IGNORE_REASONS: Tuple[str, ...] = (
    REASON_FIRST_END_NOT_KEY,
    REASON_SECOND_END_NOT_KEY,
    REASON_BOTH_ENDS_NOT_KEY,
)


class PairOwnershipError(RuntimeError):
    """Typed fail-closed refusal (cross-graph / out-of-range pair endpoints)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__("{}: {}".format(code, detail))
        self.code = code


@dataclass(frozen=True)
class GraphPairSlice:
    """Pair ownership for ONE graph of one collated batch.

    ``pairs`` are batch-global node rows (both endpoints key atoms of this
    graph) in ascending ``pair_index`` column order — exactly the space
    ``evaluate_g53``'s ``pos``/``key_rows`` indexing uses.  ``pair_edits``
    is column-aligned with ``pairs`` and passes ``pair_edit`` through
    unchanged (intent sign preserved).
    """

    graph: int
    key_lo: int          # key-row slice [lo, hi) for u/y/key_index ONLY
    key_hi: int
    node_lo: int         # node-row window [lo, hi) owning this graph's pairs
    node_hi: int
    pairs: Tuple[Tuple[int, int], ...]
    pair_edits: Tuple[float, ...]
    n_active_pairs: int
    n_ignored_non_key: int
    ignored_reasons: Mapping[str, int]
    n_invalid_pairs: int  # always 0: any invalid pair raises before return

    def audit(self) -> Dict[str, Any]:
        """JSON-serializable per-graph denominators (plan §4.2 fixtures)."""
        return {
            "graph": int(self.graph),
            "key_slice": [int(self.key_lo), int(self.key_hi)],
            "node_window": [int(self.node_lo), int(self.node_hi)],
            "n_active_pairs": int(self.n_active_pairs),
            "n_ignored_non_key": int(self.n_ignored_non_key),
            "ignored_non_key_reasons": {
                r: int(self.ignored_reasons.get(r, 0)) for r in IGNORE_REASONS
            },
            "n_invalid_pairs": int(self.n_invalid_pairs),
        }


@dataclass(frozen=True)
class BatchPairOwnership:
    """Pair ownership for a whole collated batch.

    ``key_ptr``/``node_ptr`` are the ``[G+1]`` pointers: ``key_ptr`` slices
    the packed key axis (``u``/``y``/``key_index`` ONLY), ``node_ptr`` is the
    graph window over batch-global node rows (pair ownership).
    """

    graphs: Tuple[GraphPairSlice, ...]
    key_ptr: np.ndarray
    node_ptr: np.ndarray
    n_pairs_total: int

    @property
    def n_active_pairs_total(self) -> int:
        return int(sum(g.n_active_pairs for g in self.graphs))

    @property
    def n_ignored_non_key_total(self) -> int:
        return int(sum(g.n_ignored_non_key for g in self.graphs))

    @property
    def n_invalid_pairs_total(self) -> int:
        return int(sum(g.n_invalid_pairs for g in self.graphs))

    def audit(self, *, per_graph: bool = True) -> Dict[str, Any]:
        """JSON-serializable audit of pair denominators.

        ``per_graph=True`` (default) embeds one entry per graph — used by
        tests and by tooling that calls this module directly.  Callers that
        persist the audit into run records pass ``per_graph=False`` to keep
        the document O(1) in the batch size.
        """
        reasons_total = {
            r: int(sum(g.ignored_reasons.get(r, 0) for g in self.graphs))
            for r in IGNORE_REASONS
        }
        doc: Dict[str, Any] = {
            "schema": AUDIT_SCHEMA,
            "n_graphs": len(self.graphs),
            "n_pairs_total": int(self.n_pairs_total),
            "n_active_pairs_total": self.n_active_pairs_total,
            "n_ignored_non_key_total": self.n_ignored_non_key_total,
            "n_invalid_pairs_total": self.n_invalid_pairs_total,
            "ignored_non_key_reasons_total": reasons_total,
        }
        if per_graph:
            doc["per_graph"] = [g.audit() for g in self.graphs]
        return doc


def _as_int1d(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise PairOwnershipError(
            "SHAPE_{}".format(name.upper()),
            "{} must be 1-D, got shape {}".format(name, arr.shape))
    return arr.astype(np.int64, copy=False)


def assign_batch_pairs(
    n_nodes: Any,
    n_keys: Any,
    key_index: Any,
    pair_index: Any,
    pair_edit: Any,
) -> BatchPairOwnership:
    """Attribute batch-global node-row ``pair_index`` columns to their graphs.

    Parameters
    ----------
    n_nodes : [G] node count per graph (defines the node-row pointer /
        pair ownership windows).
    n_keys : [G] key count per graph (defines the key-row pointer that
        slices ``u``/``y``/``key_index`` ONLY — never pair windows).
    key_index : [ΣK] batch-global node rows of the key atoms.
    pair_index : [2, P] batch-global node rows (collate convention).
    pair_edit : [P] signed intent edit per pair column (passed through
        unchanged; required so active pairs stay column-aligned).

    Raises
    ------
    PairOwnershipError
        shape/length violations, pair endpoints out of node range, or a
        pair whose endpoints span two graphs (fail-closed: never silently
        re-attributed, never silently dropped).
    """
    n_nodes_a = _as_int1d("n_nodes", n_nodes)
    n_keys_a = _as_int1d("n_keys", n_keys)
    key_a = _as_int1d("key_index", key_index)
    if n_nodes_a.shape != n_keys_a.shape:
        raise PairOwnershipError(
            "GRAPH_COUNT_MISMATCH",
            "n_nodes has {} graphs, n_keys has {}".format(
                n_nodes_a.size, n_keys_a.size))
    if n_nodes_a.size and (int(n_nodes_a.min()) < 0 or int(n_keys_a.min()) < 0):
        raise PairOwnershipError(
            "NEGATIVE_COUNT", "n_nodes/n_keys must be non-negative")
    pairs_a = np.asarray(pair_index, dtype=np.int64)
    if pairs_a.size == 0:
        pairs_a = pairs_a.reshape(2, 0)
    if pairs_a.ndim != 2 or pairs_a.shape[0] != 2:
        raise PairOwnershipError(
            "PAIR_INDEX_SHAPE",
            "pair_index must be [2, P], got shape {}".format(pairs_a.shape))
    edits_a = np.asarray(pair_edit, dtype=np.float64).reshape(-1)
    if edits_a.size != pairs_a.shape[1]:
        raise PairOwnershipError(
            "PAIR_EDIT_LENGTH",
            "pair_edit has {} entries, pair_index has {} columns".format(
                edits_a.size, pairs_a.shape[1]))

    key_ptr = np.concatenate(([0], np.cumsum(n_keys_a))).astype(np.int64)
    node_ptr = np.concatenate(([0], np.cumsum(n_nodes_a))).astype(np.int64)
    if key_a.size != int(key_ptr[-1]):
        raise PairOwnershipError(
            "KEY_INDEX_LENGTH",
            "key_index has {} rows, sum(n_keys) is {}".format(
                key_a.size, int(key_ptr[-1])))
    total_nodes = int(node_ptr[-1])
    # keys of graph g must sit inside graph g's node window (collate
    # invariant; fail-closed so a ptr-space mixup can never hide here)
    for g in range(int(n_nodes_a.size)):
        k_lo, k_hi = int(key_ptr[g]), int(key_ptr[g + 1])
        n_lo, n_hi = int(node_ptr[g]), int(node_ptr[g + 1])
        rows = key_a[k_lo:k_hi]
        if rows.size and (int(rows.min()) < n_lo or int(rows.max()) >= n_hi):
            raise PairOwnershipError(
                "KEY_ROW_OUT_OF_WINDOW",
                "graph {} key rows [{}, {}] escape its node window "
                "[{}, {})".format(
                    g, int(rows.min()), int(rows.max()), n_lo, n_hi))

    # per-graph key membership (batch-global node rows)
    key_sets = [
        {int(v) for v in key_a[int(key_ptr[g]):int(key_ptr[g + 1])]}
        for g in range(int(n_nodes_a.size))
    ]

    active: List[List[Tuple[int, int]]] = [[] for _ in key_sets]
    edits: List[List[float]] = [[] for _ in key_sets]
    reasons: List[Dict[str, int]] = [
        {r: 0 for r in IGNORE_REASONS} for _ in key_sets
    ]

    n_pairs = int(pairs_a.shape[1])
    for col in range(n_pairs):
        a = int(pairs_a[0, col])
        b = int(pairs_a[1, col])
        if not (0 <= a < total_nodes) or not (0 <= b < total_nodes):
            raise PairOwnershipError(
                "PAIR_ENDPOINT_OUT_OF_RANGE",
                "pair column {} endpoint ({}, {}) outside node range "
                "[0, {})".format(col, a, b, total_nodes))
        ga = int(np.searchsorted(node_ptr, a, side="right")) - 1
        gb = int(np.searchsorted(node_ptr, b, side="right")) - 1
        if ga != gb:
            raise PairOwnershipError(
                "PAIR_CROSS_GRAPH",
                "pair column {} endpoint ({}, {}) spans graph windows "
                "[{}, {}) and [{}, {})".format(
                    col, a, b,
                    int(node_ptr[ga]), int(node_ptr[ga + 1]),
                    int(node_ptr[gb]), int(node_ptr[gb + 1])))
        keys_g = key_sets[ga]
        a_key = a in keys_g
        b_key = b in keys_g
        if a_key and b_key:
            active[ga].append((a, b))
            edits[ga].append(float(edits_a[col]))
        elif not a_key and not b_key:
            reasons[ga][REASON_BOTH_ENDS_NOT_KEY] += 1
        elif not a_key:
            reasons[ga][REASON_FIRST_END_NOT_KEY] += 1
        else:
            reasons[ga][REASON_SECOND_END_NOT_KEY] += 1

    graphs = tuple(
        GraphPairSlice(
            graph=g,
            key_lo=int(key_ptr[g]),
            key_hi=int(key_ptr[g + 1]),
            node_lo=int(node_ptr[g]),
            node_hi=int(node_ptr[g + 1]),
            pairs=tuple(active[g]),
            pair_edits=tuple(edits[g]),
            n_active_pairs=len(active[g]),
            n_ignored_non_key=int(sum(reasons[g].values())),
            ignored_reasons=dict(reasons[g]),
            n_invalid_pairs=0,
        )
        for g in range(len(key_sets))
    )
    return BatchPairOwnership(
        graphs=graphs,
        key_ptr=key_ptr,
        node_ptr=node_ptr,
        n_pairs_total=n_pairs,
    )


def merge_pair_audits(audits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Sum per-batch audits into one O(1) document (totals only).

    Used by the eval entry points to persist ONE compact
    ``pair_ownership_audit`` record per evaluation instead of one entry per
    graph (dev evaluations cover tens of thousands of graphs).
    """
    reasons_total = {r: 0 for r in IGNORE_REASONS}
    doc: Dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "n_batches": len(audits),
        "n_graphs": 0,
        "n_pairs_total": 0,
        "n_active_pairs_total": 0,
        "n_ignored_non_key_total": 0,
        "n_invalid_pairs_total": 0,
        "ignored_non_key_reasons_total": reasons_total,
    }
    for audit in audits:
        if str(audit.get("schema", AUDIT_SCHEMA)) != AUDIT_SCHEMA:
            raise PairOwnershipError(
                "AUDIT_SCHEMA",
                "unexpected audit schema {!r}".format(audit.get("schema")))
        for key in ("n_graphs", "n_pairs_total", "n_active_pairs_total",
                    "n_ignored_non_key_total", "n_invalid_pairs_total"):
            doc[key] = int(doc[key]) + int(audit.get(key, 0))
        for r in IGNORE_REASONS:
            reasons_total[r] += int(
                audit.get("ignored_non_key_reasons_total", {}).get(r, 0))
    return doc
