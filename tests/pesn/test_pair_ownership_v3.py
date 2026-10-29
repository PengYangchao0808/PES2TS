"""WP1a: shared pair-ownership tool + both eval entry points (V5.4.1 §4.1–4.2).

Scope of these tests (bounded, no training / no dev-confirm access):

* the shared tool :mod:`pes2ts_core.pesn.offline.evaluation.pair_ownership_v3`
  — pair_index is batch-global NODE rows; the key-row pointer only slices
  u/y/key_index, the node-row pointer (n_nodes) owns pair→graph windows;
* the plan §4.2 fixture matrix, each with INDEPENDENTLY stated expected
  active-pair / sign denominators (never derived from the tool under test);
* a replica of the OLD key-ptr filter showing it silently drops legal pairs
  on these fixtures (the defect WP1a fixes);
* both entry points (``evaluate_checkpoint`` and the paired-CI ``_collect``)
  import the SAME tool object and keep the g53 sample structure, q/endpoint
  collection, intent sign pass-through and MOVE eligibility mask unchanged.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.evaluation import pair_ownership_v3 as shared  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.pair_ownership_v3 import (  # noqa: E402
    PairOwnershipError,
    assign_batch_pairs,
    merge_pair_audits,
)
from pes2ts_core.pesn.offline.training import key_atom_bc_v3 as kbc  # noqa: E402
from scripts.campaigns import r3_v53mp_scale_paired_ci as cli  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic built-sample cache (the dicts collate_v3 consumes)
# ---------------------------------------------------------------------------
# Geometry: every sample's atoms sit 1.0 A apart along +x (local rows), so an
# exact pair distance is |j - i| A and a key displacement of +0.02 A along x
# gives Δd = 0.02 A (well above the 0.005 A sign threshold).


def _built_sample(
    *,
    n_nodes: int,
    key_rows: list[int],
    pairs: list[tuple[int, int]],
    edits: list[float],
    y_teacher: np.ndarray,
    sample_id: str,
    curve_id: str,
    intent_id: str = "intent_fwd",
) -> dict:
    """One hand-built sample dict accepted by ``collate_v3`` (local rows)."""
    n = int(n_nodes)
    key_rows = [int(r) for r in key_rows]
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = np.arange(n, dtype=np.float32)
    is_key = np.zeros(n, dtype=np.float32)
    for r in key_rows:
        is_key[r] = 1.0
    edge = np.array([[0], [1]] if n >= 2 else [[0], [0]], dtype=np.int64)
    pair_arr = (np.asarray(pairs, dtype=np.int64).reshape(2, -1)
                if pairs else np.zeros((2, 0), dtype=np.int64))
    return {
        "z": np.ones(n, dtype=np.int64),
        "pos": pos,
        "graph_index": np.zeros(n, dtype=np.int64),
        "key_index": np.asarray(key_rows, dtype=np.int64),
        "is_key": is_key,
        "degK": np.ones(n, dtype=np.float32),
        "intent_role": np.zeros((n, 3), dtype=np.float32),
        "edge_index": edge,
        "edge_attr": np.zeros((edge.shape[1], 17), dtype=np.float32),
        "b_src": np.zeros(edge.shape[1], dtype=np.float32),
        "b_tgt": np.zeros(edge.shape[1], dtype=np.float32),
        "delta_b_intent": np.zeros(edge.shape[1], dtype=np.float32),
        "abs_delta": np.zeros(edge.shape[1], dtype=np.float32),
        "is_changed": np.zeros(edge.shape[1], dtype=np.float32),
        "cv_target_delta": np.zeros(4, dtype=np.float32),
        "signed_intent": np.zeros(1, dtype=np.float32),
        "endpoint_names": (0, n - 1),
        "pair_index": pair_arr,
        "pair_edit": np.asarray(edits, dtype=np.float32),
        "frame_weight": np.float32(1.0),
        "d0": np.float64(0.3),
        "sample_id": sample_id,
        "curve_id": curve_id,
        "intent_id": intent_id,
        "label_status": "ok",
        "y_teacher": np.asarray(y_teacher, dtype=np.float32),
    }


def _two_graph_cache() -> list[dict]:
    """Plan §4.2 cat.1: two graphs, n_nodes != n_keys, keys non-prefix.

    graph0: nodes 0..4 (n_nodes=5), keys local [1, 3]  -> n_keys=2
    graph1: nodes 5..8 (n_nodes=4), keys local [0, 2]  -> global [5, 7]
    Independent expectations:
      active pairs   = [(1, 3), (5, 7)]   (2 total; OLD filter keeps only 1)
      sign pairs     = 2 (|Δd(y)| = 0.02 / 0.03 A, both >= 0.005 A)
      pairs total    = 2, ignored_non_key = 0
    """
    return [
        _built_sample(
            n_nodes=5, key_rows=[1, 3], pairs=[(1, 3)], edits=[0.4],
            y_teacher=[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
            sample_id="sA", curve_id="RXN_A"),
        _built_sample(
            n_nodes=4, key_rows=[0, 2], pairs=[(0, 2)], edits=[-0.4],
            y_teacher=[[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]],
            sample_id="sB", curve_id="RXN_B"),
    ]


def _legacy_pairs(batch) -> list[tuple[int, int, int]]:
    """Replica of the PRE-WP1a filter (key-ptr window on node-row pairs).

    This is the exact algorithm removed from both entry points; kept here so
    every fixture can state how the old implementation FAILED.
    """
    n_keys = batch["n_keys"].cpu().numpy()
    key_index = batch["key_index"].cpu().numpy()
    pair_index = batch["pair_index"].cpu().numpy()
    ptr = np.concatenate([[0], np.cumsum(n_keys)]).astype(np.int64)
    kept: list[tuple[int, int, int]] = []
    for g in range(len(n_keys)):
        ks, ke = int(ptr[g]), int(ptr[g + 1])
        rows_g = key_index[ks:ke]
        slot = {int(r): k for k, r in enumerate(rows_g)}
        if pair_index.shape[1]:
            in_frame = np.nonzero(
                (pair_index[0] >= ptr[g]) & (pair_index[0] < ptr[g + 1]))[0]
            for p in in_frame:
                a, b = int(pair_index[0, p]), int(pair_index[1, p])
                if a in slot and b in slot:
                    kept.append((g, a, b))
    return kept


class _StubModel(torch.nn.Module):
    """u = zeros (abstention), q = 0.5, endpoint_logit = 0 — shapes only."""

    def forward(self, batch):
        n_keys_total = int(batch["key_index"].shape[0])
        n_graphs = len(batch["endpoint_names"])
        return {
            "u": torch.zeros(n_keys_total, 3, dtype=torch.float32),
            "q": torch.full((n_graphs,), 0.5, dtype=torch.float32),
            "endpoint_logit": torch.zeros(n_graphs, dtype=torch.float32),
        }


class _Labels:
    def __init__(self, rows: dict) -> None:
        self.by_sample = rows


class _Endpoint:
    def __init__(self, rows: dict) -> None:
        self.by_sample = rows


def _label_rows() -> dict:
    return {
        "sA": SimpleNamespace(q=0.5, q_mask=True, curve_id="RXN_A"),
        "sB": SimpleNamespace(q=-0.25, q_mask=True, curve_id="RXN_B"),
    }


def _endpoint_rows() -> dict:
    return {
        "sA": SimpleNamespace(endpoint_label=1, ts_row=False, start_row=False),
        "sB": SimpleNamespace(endpoint_label=-1, ts_row=False, start_row=False),
    }


# ---------------------------------------------------------------------------
# tool fixtures — plan §4.2 categories 1..5 (independent expectations)
# ---------------------------------------------------------------------------

def test_cat1_two_graphs_n_nodes_ne_n_keys():
    """cat.1: n_nodes=[5,4] != n_keys=[2,2]; both graphs' pairs attributed
    by the NODE window, not the key-row pointer."""
    own = assign_batch_pairs(
        n_nodes=[5, 4], n_keys=[2, 2], key_index=[1, 3, 5, 7],
        pair_index=np.array([[1, 5], [3, 7]]),
        pair_edit=[0.4, -0.4])
    # independent expectation: one legal pair per graph
    assert own.graphs[0].pairs == ((1, 3),)
    assert own.graphs[1].pairs == ((5, 7),)
    assert own.graphs[0].pair_edits == (0.4,)
    assert own.graphs[1].pair_edits == (-0.4,)   # reverse-intent sign kept
    # pointers: key ptr only for u/y/key_index, node ptr for pair windows
    assert own.key_ptr.tolist() == [0, 2, 4]
    assert own.node_ptr.tolist() == [0, 5, 9]
    assert own.n_active_pairs_total == 2
    assert own.n_ignored_non_key_total == 0

    # OLD filter on the same batch drops graph1's pair (key-ptr window
    # [2,4) never contains node rows 5..8) -> 1 instead of 2.
    cache = _two_graph_cache()
    batch = kbc._batch_from_cache(cache, [0, 1], "cpu")
    assert len(_legacy_pairs(batch)) == 1
    assert own.n_active_pairs_total == 2   # NEW keeps the legal pair


def test_cat2_keys_noncontiguous_and_non_prefix():
    """cat.2: keys at node rows [4, 0] (non-contiguous, non-prefix)."""
    own = assign_batch_pairs(
        n_nodes=[6], n_keys=[2], key_index=[4, 0],
        pair_index=np.array([[4], [0]]), pair_edit=[1.0])
    # independent expectation: the (4,0) pair is legal -> active
    assert own.graphs[0].pairs == ((4, 0),)
    assert own.n_active_pairs_total == 1

    # OLD key-ptr window [0,2) never contains node row 4 -> pair dropped.
    batch = kbc._batch_from_cache(
        [_built_sample(
            n_nodes=6, key_rows=[4, 0], pairs=[(4, 0)], edits=[1.0],
            y_teacher=np.zeros((2, 3), dtype=np.float32),
            sample_id="sC", curve_id="RXN_C")],
        [0], "cpu")
    assert _legacy_pairs(batch) == []
    assert own.n_active_pairs_total == 1


def test_cat3_second_graph_node_offset_exceeds_first_key_total():
    """cat.3: node offset of graph1 (6) > total keys of graph0 (2)."""
    own = assign_batch_pairs(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pair_index=np.array([[0, 6], [1, 8]]),
        pair_edit=[0.1, 0.2])
    # independent expectation: one legal pair per graph
    assert own.graphs[0].pairs == ((0, 1),)
    assert own.graphs[1].pairs == ((6, 8),)
    assert own.n_active_pairs_total == 2
    assert (own.graphs[1].key_lo, own.graphs[1].key_hi) == (2, 4)
    assert (own.graphs[1].node_lo, own.graphs[1].node_hi) == (6, 9)

    # OLD key-ptr window for graph1 is [2,4): node rows 6,8 never enter it
    # -> graph1's pair is silently dropped (the historical defect).
    cache = [
        _built_sample(
            n_nodes=6, key_rows=[0, 1], pairs=[(0, 1)], edits=[0.1],
            y_teacher=np.zeros((2, 3), dtype=np.float32),
            sample_id="sC", curve_id="RXN_C"),
        _built_sample(
            n_nodes=3, key_rows=[0, 2], pairs=[(0, 2)], edits=[0.2],
            y_teacher=np.zeros((2, 3), dtype=np.float32),
            sample_id="sD", curve_id="RXN_D"),
    ]
    batch = kbc._batch_from_cache(cache, [0, 1], "cpu")
    legacy = _legacy_pairs(batch)
    assert legacy == [(0, 0, 1)]            # graph1 pair (6,8) LOST
    assert own.n_active_pairs_total == 2    # NEW keeps both


def test_cat4_one_graph_empty_pairs_one_graph_many():
    """cat.4: graph0 contributes 0 pairs, graph1 contributes 3."""
    own = assign_batch_pairs(
        n_nodes=[3, 6], n_keys=[1, 3], key_index=[0, 3, 5, 7],
        pair_index=np.array([[3, 5, 3], [5, 7, 7]]),
        pair_edit=[0.1, 0.2, 0.3])
    # independent expectations
    assert own.graphs[0].pairs == () and own.graphs[0].n_active_pairs == 0
    assert own.graphs[1].pairs == ((3, 5), (5, 7), (3, 7))
    assert own.graphs[1].pair_edits == (0.1, 0.2, 0.3)  # column order kept
    assert own.n_pairs_total == 3
    assert own.n_active_pairs_total == 3
    audit = own.audit()
    assert audit["per_graph"][0]["n_active_pairs"] == 0
    assert audit["per_graph"][1]["n_active_pairs"] == 3
    assert audit["n_invalid_pairs_total"] == 0


def test_cat5_non_key_endpoints_counted_legal_pair_kept():
    """cat.5: one end outside K is IGNORED with an explicit reason — the
    legal pair in the same graph is never silently dropped."""
    # keys at rows 1,3 of a 6-node graph; pairs: legal (1,3), then
    # (0,1) first end not K, (1,2) second end not K, (2,4) both not K.
    own = assign_batch_pairs(
        n_nodes=[6], n_keys=[2], key_index=[1, 3],
        pair_index=np.array([[1, 0, 1, 2], [3, 1, 2, 4]]),
        pair_edit=[1.0, 1.0, 1.0, 1.0])
    g = own.graphs[0]
    # independent expectation: exactly the one legal pair survives
    assert g.pairs == ((1, 3),)
    assert g.n_active_pairs == 1
    assert g.n_ignored_non_key == 3
    assert g.ignored_reasons["first_end_not_key"] == 1
    assert g.ignored_reasons["second_end_not_key"] == 1
    assert g.ignored_reasons["both_ends_not_key"] == 1
    # denominators reconcile: every column is classified, none vanished
    assert own.n_pairs_total == 4
    assert (own.n_active_pairs_total + own.n_ignored_non_key_total
            == own.n_pairs_total)


# ---------------------------------------------------------------------------
# typed errors (fail-closed) + audit shape
# ---------------------------------------------------------------------------

def test_cross_graph_and_out_of_range_raise_typed_error():
    with pytest.raises(PairOwnershipError) as exc:
        assign_batch_pairs([3, 3], [1, 1], [0, 3],
                           np.array([[0], [3]]), [1.0])
    assert exc.value.code == "PAIR_CROSS_GRAPH"
    with pytest.raises(PairOwnershipError) as exc:
        assign_batch_pairs([3, 3], [1, 1], [0, 3],
                           np.array([[0], [7]]), [1.0])
    assert exc.value.code == "PAIR_ENDPOINT_OUT_OF_RANGE"
    with pytest.raises(PairOwnershipError) as exc:
        assign_batch_pairs([3], [1, 2], [0],
                           np.zeros((2, 0), dtype=int), [])
    assert exc.value.code == "GRAPH_COUNT_MISMATCH"
    with pytest.raises(PairOwnershipError) as exc:
        assign_batch_pairs([3], [2], [0, 9],   # key row escapes node window
                           np.zeros((2, 0), dtype=int), [])
    assert exc.value.code == "KEY_ROW_OUT_OF_WINDOW"


def test_audit_is_json_serializable_and_mergeable():
    own = assign_batch_pairs(
        n_nodes=[5, 4], n_keys=[2, 2], key_index=[1, 3, 5, 7],
        pair_index=np.array([[1, 5], [3, 7]]), pair_edit=[0.4, -0.4])
    audit = own.audit()
    import json
    json.dumps(audit)                      # must survive metrics.json write
    assert audit["schema"] == shared.AUDIT_SCHEMA
    assert audit["n_graphs"] == 2
    merged = merge_pair_audits([audit, audit])
    assert merged["n_active_pairs_total"] == 4
    assert merged["n_graphs"] == 4
    assert merged["n_invalid_pairs_total"] == 0
    # totals-only audits (per_graph=False) are the persisted form
    assert "per_graph" not in own.audit(per_graph=False)


# ---------------------------------------------------------------------------
# entry points: same tool, unchanged structures, unchanged eligibility
# ---------------------------------------------------------------------------

def test_entry_points_import_the_same_tool_object():
    assert kbc.assign_batch_pairs is shared.assign_batch_pairs
    assert cli.assign_batch_pairs is shared.assign_batch_pairs
    assert kbc.merge_pair_audits is shared.merge_pair_audits
    assert cli.merge_pair_audits is shared.merge_pair_audits
    # no local mirror of the buggy window filter may remain
    assert "pair_index[0] >= ptr" not in inspect.getsource(kbc.evaluate_checkpoint)
    assert "pair_index[0] >= ptr" not in inspect.getsource(cli._collect)


def test_evaluate_checkpoint_sign_denominator_recovers_dropped_pair():
    """End-to-end: n_sign_pairs == 2 (independent), OLD filter only found 1."""
    cache = _two_graph_cache()
    out = kbc.evaluate_checkpoint(
        _StubModel(), None, cache, [0, 1],
        device="cpu", bootstrap_draws=10)
    # independent expectation from the fixture docstring
    assert out["sre_and_sign"]["n_sign_pairs"] == 2
    assert out["sre_and_sign"]["n_active_pairs"] == 2
    assert out["sre_and_sign"]["n_dead_pairs"] == 2      # u == 0 abstains
    assert out["n_val_samples"] == 2
    # additive audit: totals only, consistent with the g53 denominators
    audit = out["pair_ownership_audit"]
    assert audit["n_graphs"] == 2
    assert audit["n_pairs_total"] == 2
    assert audit["n_active_pairs_total"] == 2
    assert audit["n_ignored_non_key_total"] == 0
    assert audit["n_invalid_pairs_total"] == 0
    # the OLD algorithm would have produced a sign denominator of 1
    batch = kbc._batch_from_cache(cache, [0, 1], "cpu")
    assert len(_legacy_pairs(batch)) == 1


def test_collect_pairs_q_endpoint_and_audit_unchanged():
    """_collect: pairs recovered, q/endpoint collection byte-identical in
    shape/meaning, g53 sample structure unchanged."""
    cache = _two_graph_cache()
    col = cli._collect(
        _StubModel(), cache, [0, 1], _Labels(_label_rows()),
        _Endpoint(_endpoint_rows()), "cpu")
    g0, g1 = col["g53_samples"]
    # independent expectations (global node rows, batch-global pos kept)
    assert g0["pairs"] == [(1, 3)]
    assert g1["pairs"] == [(5, 7)]
    # pair_edit is float32 upstream (same as the old code's float(pair_edit[p]))
    assert g0["pair_edits"] == pytest.approx([0.4], abs=1e-6)
    assert g1["pair_edits"] == pytest.approx([-0.4], abs=1e-6)
    assert set(g0.keys()) == {"pos", "key_rows", "pairs", "pair_edits",
                              "u", "y", "group_id"}
    assert g0["group_id"] == "RXN_A" and g1["group_id"] == "RXN_B"
    # key-row slicing of u/y unchanged (key_ptr only)
    assert len(g0["u"]) == 2 and len(g1["y"]) == 2
    # q collection unchanged (returned as arrays, same as before the fix)
    assert len(col["q_preds"]) == 2 and len(col["q_trues"]) == 2
    np.testing.assert_allclose(col["q_trues"], [0.5, -0.25], atol=1e-6)
    np.testing.assert_allclose(col["q_masks"], [True, True])
    assert list(col["q_curves"]) == ["RXN_A", "RXN_B"]
    # endpoint collection unchanged (label -1 -> masked out of the denominator)
    assert len(col["ep_logits"]) == 2
    assert list(col["ep_y"]) == [1, 0]
    assert list(col["ep_keep"]) == [True, False]
    assert list(col["ep_curves"]) == ["RXN_A", "RXN_B"]
    # audit agrees with the g53 pairs actually collected
    assert col["pair_ownership_audit"]["n_active_pairs_total"] == 2
    assert col["pair_ownership_audit"]["n_pairs_total"] == 2


def test_intent_sign_passes_through_counts_unchanged():
    """Forward (+0.4) vs reverse (-0.4) intent: same pairs/counts, only the
    signed edits flip — ownership must not depend on the intent sign."""
    cache_f = _two_graph_cache()
    cache_r = _two_graph_cache()
    cache_r[0]["pair_edit"] = np.asarray([-0.4], dtype=np.float32)
    cache_r[1]["pair_edit"] = np.asarray([0.4], dtype=np.float32)
    col_f = cli._collect(_StubModel(), cache_f, [0, 1],
                         _Labels(_label_rows()), _Endpoint(_endpoint_rows()),
                         "cpu")
    col_r = cli._collect(_StubModel(), cache_r, [0, 1],
                         _Labels(_label_rows()), _Endpoint(_endpoint_rows()),
                         "cpu")
    assert [s["pairs"] for s in col_f["g53_samples"]] == \
           [s["pairs"] for s in col_r["g53_samples"]]
    assert col_f["pair_ownership_audit"]["n_active_pairs_total"] == \
           col_r["pair_ownership_audit"]["n_active_pairs_total"] == 2
    # intent sign is preserved exactly (forward +, reverse -)
    assert col_f["g53_samples"][0]["pair_edits"] == pytest.approx(
        [0.4], abs=1e-6)
    assert col_r["g53_samples"][0]["pair_edits"] == pytest.approx(
        [-0.4], abs=1e-6)
    assert col_f["g53_samples"][1]["pair_edits"] == pytest.approx(
        [-0.4], abs=1e-6)
    assert col_r["g53_samples"][1]["pair_edits"] == pytest.approx(
        [0.4], abs=1e-6)


def test_move_eligibility_mask_unchanged():
    """terminal/masked/conflict rows stay OUT of the MOVE denominator —
    eligibility lives in _ok_mask (label_status), never in pair ownership."""
    status = {0: "ok", 1: "terminal", 2: "masked", 3: "conflict"}

    class _DS:
        def records(self, ps):
            return [SimpleNamespace(label_status=status[int(p)]) for p in ps]

    mask = cli._ok_mask(_DS(), [0, 1, 2, 3])
    assert mask.tolist() == [True, False, False, False]
    # the ownership tool never sees label_status/frame_weight at all:
    # its signature is structural only (no eligibility inputs)
    params = set(inspect.signature(assign_batch_pairs).parameters)
    assert params == {"n_nodes", "n_keys", "key_index", "pair_index",
                      "pair_edit"}


def test_g53_sample_structure_identical_to_contract():
    """The fix may not alter the evaluate_g53 input structure."""
    col = cli._collect(_StubModel(), _two_graph_cache(), [0, 1],
                       _Labels(_label_rows()), _Endpoint(_endpoint_rows()),
                       "cpu")
    for s in col["g53_samples"]:
        assert sorted(s.keys()) == ["group_id", "key_rows", "pair_edits",
                                    "pairs", "pos", "u", "y"]
        # batch-global semantics: pos is the WHOLE batch, rows are global
        assert s["pos"].shape[0] == 9
        assert all(0 <= r < 9 for r in s["key_rows"])
        assert all(0 <= a < 9 and 0 <= b < 9 for a, b in s["pairs"])
