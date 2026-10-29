"""WP1b: v541_baseline_repair unit tests (fixtures only — no real data).

What is pinned here (plan §4.3 implementation surface):

* the audit-only LEGACY key-ptr replica reproduces the pre-WP1a defect
  (drops legal pairs) and is a subset of the repaired node-ptr collection;
* the g53 sample structure handed to evaluate_g53 is byte-identical in
  shape to the production contract;
* pair-independent metric fields (RMSE/zero, fixed-coverage direction)
  are identical before/after by construction, while g53 fields move with
  the pair set;
* the process-metrics builder (q/ordering/endpoint) structurally cannot
  see pairs (signature guard) — the "pair fix must not change them" claim;
* gate helpers, sanity gates, H2 guard formula, canonical SHAs, ledger
  schema and the fail-closed failure record.

The REAL-data consistency smoke (audit vs BOTH production entry points)
runs in `v541_baseline_repair.py smoke`, not here.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.campaigns import v541_baseline_repair as br  # noqa: E402
from pes2ts_core.pesn.offline.evaluation import pair_ownership_v3 as po  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (  # noqa: E402
    evaluate_g53,
)


# ---------------------------------------------------------------------------
# fixtures: the WP1a old-fail sample + the prefix agreement sample
# ---------------------------------------------------------------------------


def _fixture(n_nodes, n_keys, key_index, pairs, edits, curves):
    return br._fixture_samples(n_nodes, n_keys, key_index, pairs, edits, curves)


def test_legacy_drops_legal_pair_repaired_keeps_it():
    """old-fail fixture: legacy key-ptr window loses graph1's pair (6,8)."""
    legacy, repaired, own = _fixture(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pairs=[(0, 1), (6, 8)], edits=[0.4, -0.4],
        curves=["FIX_A", "FIX_B"])
    assert [s["pairs"] for s in legacy] == [[(0, 1)], []]
    assert [s["pairs"] for s in repaired] == [[(0, 1)], [(6, 8)]]
    # audit denominators: both pairs input, legacy collected 1, repaired 2
    assert own.n_pairs_total == 2
    assert own.n_active_pairs_total == 2
    assert own.n_ignored_non_key_total == 0
    assert own.n_invalid_pairs_total == 0


def test_sign_denominator_recovers_the_dropped_pair():
    """Independent expectation: repaired sign denominator = legacy + 1."""
    legacy, repaired, _ = _fixture(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pairs=[(0, 1), (6, 8)], edits=[0.4, -0.4],
        curves=["FIX_A", "FIX_B"])
    n_leg = evaluate_g53(legacy)["aggregate"]["n_sign_pairs"]
    n_rep = evaluate_g53(repaired)["aggregate"]["n_sign_pairs"]
    assert n_leg == 1
    assert n_rep == n_leg + 1 == 2


def test_legacy_agrees_on_prefix_case():
    """Where the old window was correct, before == after (no fake deltas)."""
    legacy, repaired, _ = _fixture(
        n_nodes=[2, 2], n_keys=[2, 2], key_index=[0, 1, 2, 3],
        pairs=[(0, 1), (2, 3)], edits=[0.5, 0.5],
        curves=["FIX_C", "FIX_D"])
    assert [s["pairs"] for s in legacy] == [s["pairs"] for s in repaired]


def test_noncontiguous_nonprefix_keys_recovered():
    """cat.2: keys at node rows [4,0] — legacy window [0,2) drops (4,0)."""
    legacy, repaired, own = _fixture(
        n_nodes=[6], n_keys=[2], key_index=[4, 0],
        pairs=[(4, 0)], edits=[1.0], curves=["FIX_D"])
    assert [s["pairs"] for s in legacy] == [[]]
    assert [s["pairs"] for s in repaired] == [[(4, 0)]]
    assert own.n_active_pairs_total == 1


def test_g53_sample_structure_matches_production_contract():
    _, repaired, _ = _fixture(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pairs=[(0, 1), (6, 8)], edits=[0.4, -0.4],
        curves=["FIX_A", "FIX_B"])
    for s in repaired:
        assert set(s.keys()) == {"pos", "key_rows", "pairs", "pair_edits",
                                 "u", "y", "group_id"}
    # batch-global semantics: key_rows/pairs are global rows, pos covers all
    assert repaired[1]["key_rows"] == [6, 8]
    assert repaired[1]["pairs"] == [(6, 8)]
    assert repaired[0]["pos"].shape[0] == 9
    assert repaired[1]["group_id"] == "FIX_B"


def test_legacy_is_subset_of_repaired_on_random_fixtures():
    """Property: legacy ⊆ repaired for every fixture (never reversed)."""
    rng = np.random.RandomState(11)
    for trial in range(25):
        n_graphs = int(rng.randint(1, 4))
        n_nodes = [int(rng.randint(2, 8)) for _ in range(n_graphs)]
        n_keys = [int(rng.randint(1, n + 1)) for n in n_nodes]
        # scatter key rows inside each graph's node window (non-prefix ok)
        key_index = []
        pairs = []
        edits = []
        off = 0
        for g, (n, k) in enumerate(zip(n_nodes, n_keys)):
            rows = sorted(rng.choice(n, size=k, replace=False).tolist())
            key_index.extend([r + off for r in rows])
            # up to 2 pairs per graph over the graph's key rows
            for _ in range(int(rng.randint(0, 3))):
                if k < 2:
                    break
                i, j = rng.choice(k, size=2, replace=False).tolist()
                pairs.append((rows[i] + off, rows[j] + off))
                edits.append(float(rng.choice([-1.0, 1.0])))
            off += n
        if not pairs:
            continue
        legacy, repaired, own = _fixture(
            n_nodes, n_keys, key_index, pairs, edits,
            ["T%d" % g for g in range(n_graphs)])
        leg = [s["pairs"] for s in legacy]
        rep = [s["pairs"] for s in repaired]
        for lg, rp in zip(leg, rep):
            assert set(lg).issubset(set(rp)), (trial, lg, rp)
        # repaired collects EVERY legal (both-ends-K) pair: no silent drop
        assert own.n_active_pairs_total + own.n_ignored_non_key_total \
            == own.n_pairs_total
        assert own.n_active_pairs_total == sum(len(x) for x in rep)


def test_repaired_pair_audit_matches_counts():
    _, repaired, own = _fixture(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pairs=[(0, 1), (6, 8)], edits=[0.4, -0.4],
        curves=["FIX_A", "FIX_B"])
    audit = own.audit()
    assert audit["n_pairs_total"] == 2
    assert audit["n_active_pairs_total"] == 2
    assert audit["n_ignored_non_key_total"] == 0
    assert audit["n_invalid_pairs_total"] == 0
    assert len(audit["per_graph"]) == 2
    json.dumps(audit)  # must survive JSON persistence


def test_shared_tool_identity_between_script_and_production():
    """The audit script imports the SAME WP1a tool object as production."""
    from pes2ts_core.pesn.offline.training import key_atom_bc_v3 as kbc
    from scripts.campaigns import r3_v53mp_scale_paired_ci as cli
    assert br.assign_batch_pairs is po.assign_batch_pairs
    assert br.assign_batch_pairs is kbc.assign_batch_pairs
    assert br.assign_batch_pairs is cli.assign_batch_pairs


def test_legacy_logic_never_leaks_into_production():
    """legacy replica lives ONLY in the audit script/tests."""
    from pes2ts_core.pesn.offline.training import key_atom_bc_v3 as kbc
    from scripts.campaigns import r3_v53mp_scale_paired_ci as cli
    assert hasattr(br, "_legacy_g53_samples")
    assert not [n for n in dir(kbc) if n.startswith("_legacy")]
    assert not [n for n in dir(cli) if n.startswith("_legacy")]
    # and the production sources no longer contain the key-ptr window filter
    assert "pair_index[0] >= ptr" not in inspect.getsource(kbc.evaluate_checkpoint)
    assert "pair_index[0] >= ptr" not in inspect.getsource(cli._collect)


# ---------------------------------------------------------------------------
# metric-block guarantees
# ---------------------------------------------------------------------------


def test_process_metrics_structurally_cannot_see_pairs():
    """q/ordering/endpoint builder takes only raw predictions (+tau)."""
    sig = inspect.signature(br._process_metrics)
    assert set(sig.parameters) == {"preds", "tau_end"}
    assert sig.parameters["tau_end"].kind is inspect.Parameter.KEYWORD_ONLY


def test_move_metrics_pair_independent_fields_identical():
    """RMSE/zero + fixed-coverage direction are computed ONCE and reported
    identically in before/after; only g53 moves with the pair set."""
    legacy, repaired, own = _fixture(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pairs=[(0, 1), (6, 8)], edits=[0.4, -0.4],
        curves=["FIX_A", "FIX_B"])
    meta = {
        "n_pairs_input": own.n_pairs_total,
        "n_pairs_collected_before": sum(len(s["pairs"]) for s in legacy),
        "n_pairs_collected_after": sum(len(s["pairs"]) for s in repaired),
        "pair_ownership_audit": own.audit(per_graph=False),
    }
    move = br._move_metrics(legacy, repaired, meta)
    b, a = move["before_legacy"], move["after_repaired"]
    assert b["key_rmse_over_zero"] == a["key_rmse_over_zero"]
    assert b["fixed_coverage_direction"] == a["fixed_coverage_direction"]
    # g53 DOES move: repaired has one more sign pair
    assert (a["sre_and_sign"]["n_sign_pairs"]
            == b["sre_and_sign"]["n_sign_pairs"] + 1)
    assert "fixed_coverage_direction" in move["pair_independent_fields"]
    # denominator deltas exposed
    d = move["delta_after_minus_before"]
    assert d["n_sign_pairs"] == 1
    assert d["n_pairs_collected"] == 1


def test_move_delta_reports_pair_independent_zero():
    legacy, repaired, own = _fixture(
        n_nodes=[2, 2], n_keys=[2, 2], key_index=[0, 1, 2, 3],
        pairs=[(0, 1), (2, 3)], edits=[0.5, 0.5],
        curves=["FIX_C", "FIX_D"])
    meta = {"n_pairs_input": 2,
            "n_pairs_collected_before": 2,
            "n_pairs_collected_after": 2,
            "pair_ownership_audit": own.audit(per_graph=False)}
    move = br._move_metrics(legacy, repaired, meta)
    d = move["delta_after_minus_before"]
    assert d["n_sign_pairs"] == 0
    assert d["median_rmse_over_zero"] == 0.0


# ---------------------------------------------------------------------------
# gates / sanity / H2 / SHAs / ledger / failure record
# ---------------------------------------------------------------------------


def test_compare_gate_modes():
    g = br._compare("x", 1.0, 1.0 + 5e-7, 1e-6, "abs")
    assert g["pass"] is True
    g = br._compare("x", 1.0, 1.0 + 5e-5, 1e-6, "abs")
    assert g["pass"] is False
    g = br._compare("n", 7, 7, 0, "exact")
    assert g["pass"] is True
    g = br._compare("n", 7, 8, 0, "exact")
    assert g["pass"] is False
    # None pairing
    assert br._compare("z", None, None, 0, "exact")["pass"] is True
    assert br._compare("z", None, 1.0, 0, "exact")["pass"] is False


def _move_dict(n_input, n_before, n_after, n_sign_b, n_sign_a, ignored=0):
    return {
        "pair_denominators": {
            "n_pairs_input": n_input,
            "n_pairs_collected_before": n_before,
            "n_pairs_collected_after": n_after,
            "pair_ownership_audit": {
                "n_ignored_non_key_total": ignored,
                "n_invalid_pairs_total": 0,
            },
        },
        "before_legacy": {"sre_and_sign": {"n_sign_pairs": n_sign_b,
                                           "n_active_pairs": n_sign_b}},
        "after_repaired": {"sre_and_sign": {"n_sign_pairs": n_sign_a,
                                            "n_active_pairs": n_sign_a}},
    }


def test_move_sanity_gates_pass_on_consistent_move():
    move = _move_dict(n_input=5, n_before=3, n_after=4,
                      n_sign_b=3, n_sign_a=4, ignored=1)
    gates = br._move_sanity(move, n_positions=10, n_positions_ok=8,
                            n_curves_move=3)
    assert br._pass_all(gates)


def test_move_sanity_gates_fail_on_silent_drop_or_regression():
    # silent drop: input - ignored != collected_after
    move = _move_dict(n_input=5, n_before=4, n_after=3,
                      n_sign_b=3, n_sign_a=3, ignored=1)
    gates = {g["name"]: g["pass"] for g in br._move_sanity(
        move, n_positions=10, n_positions_ok=8, n_curves_move=3)}
    assert gates["s1_repaired_no_silent_pair_drop"] is False
    assert gates["s4_legacy_subset_of_repaired"] is False
    # sign regression after >= before violated
    move2 = _move_dict(n_input=5, n_before=2, n_after=4,
                       n_sign_b=5, n_sign_a=4, ignored=1)
    gates2 = {g["name"]: g["pass"] for g in br._move_sanity(
        move2, n_positions=10, n_positions_ok=8, n_curves_move=3)}
    assert gates2["s5_sign_pairs_monotone"] is False
    assert gates2["s6_active_pairs_monotone"] is False
    # zero denominators fail
    gates3 = {g["name"]: g["pass"] for g in br._move_sanity(
        _move_dict(0, 0, 0, 0, 0), n_positions=0, n_positions_ok=0,
        n_curves_move=0)}
    assert gates3["s7_denominators_nonzero"] is False


def test_h2_guard_formula_matches_existing_guards():
    def _move(median_rmse, rmse0):
        return {"key_rmse_over_zero": {"median_rmse": median_rmse,
                                       "median_rmse_over_zero": rmse0},
                "sre_and_sign": {"sign_accuracy_cluster_median": 0.9,
                                 "dead_zone_fraction": 0.1}}
    g = br._h2_move_guard(_move(0.10, 0.90), _move(0.08, 0.80),
                          split="dev", variant="after_repaired", seed=17)
    assert g["rmse_degradation_pct"] == pytest.approx(25.0)
    assert g["within_rmse_guard_2pct"] is False
    assert g["rmse_over_zero_degradation_pct"] == pytest.approx(12.5)
    assert g["within_rmse0_guard_2pct"] is False
    assert g["threshold_pct"] == 2.0
    # within guard
    g2 = br._h2_move_guard(_move(0.0805, 0.80), _move(0.08, 0.80),
                           split="dev", variant="after_repaired", seed=17)
    assert g2["within_rmse_guard_2pct"] is True


def test_canonical_shas_are_order_invariant_and_content_sensitive():
    a = br._canonical_positions_sha([3, 1, 2])
    b = br._canonical_positions_sha([1, 2, 3])
    assert a == b
    assert a != br._canonical_positions_sha([1, 2, 4])
    c = br._canonical_curves_sha(["b", "a"])
    assert c == br._canonical_curves_sha(["a", "b"])
    assert c != br._canonical_curves_sha(["a", "c"])


def test_ledger_entry_schema_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "AUDIT_DIR", tmp_path)
    monkeypatch.setattr(br, "LEDGER_PATH", tmp_path / "dev_access_ledger.jsonl")
    cost = {"wall_s": 1.0, "cpu_s": 1.0, "peak_rss_mib": 100,
            "new_QC_calls": 0}
    entry = br._ledger_entry(
        split="dev", model="C_s17", output_path="x.json",
        output_sha256="ab", n_positions=135926, n_curves=589,
        curve_set_sha256="c1", positions_sha256="p1",
        code_sha256="code", cost=cost)
    assert entry["purpose"] == "baseline_repair"
    assert entry["sealed"] is True
    assert entry["unseal_before_wp7"] is False
    assert entry["new_QC_calls"] == 0
    br._append_ledger(entry)
    lines = (tmp_path / "dev_access_ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1
    back = json.loads(lines[0])
    assert back["output_sha256"] == "ab"
    assert back["positions_sha256"] == "p1"


def test_failure_record_is_fail_closed_and_keeps_r3_guard(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(br, "AUDIT_DIR", tmp_path)
    monkeypatch.setattr(br, "LEDGER_PATH", tmp_path / "dev_access_ledger.jsonl")
    path = br._record_failure("dev", br.RepairError("ASSET_MISSING", "x"))
    doc = json.loads(path.read_text())
    assert doc["status"] == "failed"
    assert doc["fail_closed"] is True
    assert doc["new_QC_calls"] == 0
    assert "NOT skipped" in doc["r3_guard"]
    assert doc["error_code"] == "ASSET_MISSING"
    # dev failures also land in the ledger
    lines = (tmp_path / "dev_access_ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "failed"


def test_freeze_drift_detection(monkeypatch):
    recorded = {"code": {"identity": {"a.py": "sha1"},
                         "provenance": {"git_head": "h",
                                        "git_diff_sha256": "d",
                                        "git_porcelain_sha256": "p"}},
                "config": {"sha256": "x"},
                "split_manifest": {"sha256": "y"},
                "data_manifest": {"sha256": "z"},
                "label_spec_sha256": "L",
                "c_runs": {"17": {"checkpoint": {"sha256": "c"}}},
                "r3_reference": {"checkpoint": {"sha256": "r"}}}
    current = json.loads(json.dumps(recorded))
    monkeypatch.setattr(br, "_freeze_assets", lambda: current)
    br._assert_freeze_unchanged(recorded)   # identical → no raise
    # identity drift (eval-path file changed) → typed refusal
    drifted = json.loads(json.dumps(recorded))
    drifted["code"]["identity"]["a.py"] = "sha2"
    with pytest.raises(br.RepairError) as exc:
        br._assert_freeze_unchanged(drifted)
    assert exc.value.code == "CODE_DRIFTED_SINCE_INNER_VAL"
    assert "a.py" in str(exc.value)
    # provenance drift (git porcelain/diff — e.g. unrelated untracked
    # files from a concurrent session) must NOT fail the freeze
    prov_drift = json.loads(json.dumps(recorded))
    prov_drift["code"]["provenance"]["git_porcelain_sha256"] = "other"
    br._assert_freeze_unchanged(prov_drift)
    drifted2 = json.loads(json.dumps(recorded))
    drifted2["config"]["sha256"] = "other"
    with pytest.raises(br.RepairError) as exc:
        br._assert_freeze_unchanged(drifted2)
    assert exc.value.code == "FROZEN_ASSET_DRIFTED"


def test_code_identity_sha_is_identity_only():
    """Ledger code SHA derives from identity, not git provenance."""
    a = {"code": {"identity": {"a.py": "1"}, "provenance": {"p": "x"}}}
    b = {"code": {"identity": {"a.py": "1"}, "provenance": {"p": "y"}}}
    c = {"code": {"identity": {"a.py": "2"}, "provenance": {"p": "x"}}}
    assert br._code_identity_sha(a) == br._code_identity_sha(b)
    assert br._code_identity_sha(a) != br._code_identity_sha(c)


def test_jsonable_handles_numpy_and_paths(tmp_path):
    doc = {"f": np.float64(1.5), "i": np.int64(3), "b": np.bool_(True),
           "a": np.asarray([1, 2]), "p": tmp_path}
    text = json.dumps(br._jsonable(doc))
    back = json.loads(text)
    assert back["f"] == 1.5 and back["i"] == 3 and back["b"] is True
    assert back["a"] == [1, 2] and back["p"] == str(tmp_path)


def test_code_freeze_covers_eval_surface():
    """Every eval-path file the repair depends on is in the freeze list."""
    for rel in ("pes2ts_core/pesn/offline/evaluation/pair_ownership_v3.py",
                "pes2ts_core/pesn/offline/training/key_atom_bc_v3.py",
                "scripts/campaigns/r3_v53mp_scale_paired_ci.py",
                "scripts/campaigns/v541_baseline_repair.py"):
        assert rel in br.CODE_FILES
        assert (PROJECT_ROOT / rel).is_file()


# ---------------------------------------------------------------------------
# derived ordering tolerance (float-noise bound, never from history)
# ---------------------------------------------------------------------------


def _pred(q_pred: float, q_true: float, *, curve: str = "C1",
          direction: str = "forward") -> br._PosPred:
    return br._PosPred(
        position=0, sample_id="s", curve_id=curve, n_nodes=4, n_keys=2,
        pos=np.zeros((4, 3), dtype=np.float32),
        key_rows=np.asarray([0, 1], dtype=np.int64),
        pairs=np.zeros((2, 0), dtype=np.int64),
        edits=np.zeros(0, dtype=np.float32),
        u=np.zeros((2, 3), dtype=np.float32),
        y=np.zeros((2, 3), dtype=np.float32),
        direction=direction, q_pred=q_pred, q_true=q_true, q_mask=True)


def test_ordering_tie_tols_wide_gaps_gives_floor_tolerance():
    """Well-separated q values → no float-noise-flippable pair → floor tol."""
    preds = [_pred(0.0, 0.1), _pred(1.0, 0.2),
             _pred(2.0, 0.3), _pred(3.0, 0.4)]
    out = br._ordering_tie_tols(preds, q_noise=1e-9)
    assert out["n_pairs_full"] == 6            # 4 choose 2, distinct truths
    assert out["n_flippable_full"] == 0
    assert out["tol_ordering_full"] == pytest.approx(1e-9)
    # cross_ts: truths all same sign → no cross pairs
    assert out["n_pairs_cross_ts"] == 0
    assert out["tol_ordering_cross_ts"] is None or \
        out["tol_ordering_cross_ts"] == pytest.approx(1e-9)


def test_ordering_tie_tols_counts_near_ties_within_noise_band():
    """q values inside the measured noise band → all pairs flippable →
    tol = flippable/n_pairs (the provable float-noise bound)."""
    base = 0.5
    preds = [_pred(base + k * 1e-9, 0.1 + k * 0.1) for k in range(4)]
    out = br._ordering_tie_tols(preds, q_noise=1e-9)
    assert out["n_pairs_full"] == 6
    assert out["n_flippable_full"] == 6
    assert out["tol_ordering_full"] == pytest.approx(1.0)


def test_ordering_tie_tols_denominators_match_ordering_pair_accuracy():
    """The counted denominators must equal the gated metric's own n_pairs
    (same pair enumeration as ordering_pair_accuracy)."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        ordering_pair_accuracy,
    )
    preds = [_pred(0.3, 0.1), _pred(0.35, -0.2), _pred(-0.1, 0.4),
             _pred(0.5, -0.3), _pred(0.05, 0.2)]
    out = br._ordering_tie_tols(preds, q_noise=1e-8)
    qp = [p.q_pred for p in preds]
    qt = [p.q_true for p in preds]
    groups = [(p.curve_id, p.direction) for p in preds]
    full = ordering_pair_accuracy(qp, qt, groups, gap=0.0, cross_ts_only=False)
    cross = ordering_pair_accuracy(qp, qt, groups, gap=0.0, cross_ts_only=True)
    assert out["n_pairs_full"] == full["n_pairs"]
    assert out["n_pairs_cross_ts"] == cross["n_pairs"]
    # and the derived tol bounds the possible accuracy drift: flippable/n
    assert 0.0 <= out["tol_ordering_full"] <= 1.0


def test_ordering_tolerance_is_fail_closed_when_missing():
    """No probe → no guessed flat tolerance: typed refusal."""
    move = {"after_repaired": {"key_rmse_over_zero": {}, "sre_and_sign": {}},
            "before_legacy": {"sre_and_sign": {}}}
    process = {"q_head": {}, "ordering": {}, "endpoint": {}}
    history = {"history_move_metrics": {}}
    with pytest.raises(br.RepairError) as exc:
        br._dev_fidelity_gates(move, process, history, {}, None)
    assert exc.value.code == "ORDERING_TOLERANCE_MISSING"
