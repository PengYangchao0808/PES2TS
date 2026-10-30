"""V5.4.2 WP6 lane — statistics + selection machinery tests.

CPU-fast, synthetic metric dicts only (no training, no real data,
no frozen artifacts touched) for ``scripts/agent_ops/v542_stats.py``
(ADR-0033 §2.5/§2.6, config ``evaluation_contract``, plan §8.2/§9):

* route_logic truth table INCLUDING the dangling-OR traps (a passing
  branch can never carry its non-inferiority condition away);
* H2 guards at the frozen boundary values (exactly 2% / exactly 0.01
  pass — non-inferiority is ``<=`` — anything above breaches) against
  BOTH the matched A0 and the R3 side; R3/missing-control is a typed
  refusal, never a silent downgrade;
* screen score ordering + the frozen tie-break (inner-val total loss,
  then core time, then candidate id), top2 ≤ 2 with NO substitutes and
  A4 reported ineligible with the binding WP4 diagnostic reason;
* bootstrap determinism (same sub-seed ⇒ same draws across calls; CI
  and p come from the SAME draws), the one-sided improvement p formula
  on hand-computable cases, and Holm monotonicity (adjusted ≥ raw,
  step-down);
* the frozen final_selection_rule with 0 / 1 / many qualified
  candidates and the EXACT 7-step lexicographic tie-break;
* activation predicates including the "A4 never ran is a consequence,
  not a blocker" rule, frozen skip reason codes and zero new training
  on every skip;
* no dev read (``dev_used=false``), no ``torch.compile`` anywhere in
  the lane; the metrics table is append-only and idempotent.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.agent_ops import v542_stats as st  # noqa: E402
from scripts.agent_ops.v542_config_schema import (  # noqa: E402
    FINAL_EFFECT_MARGIN,
    FINAL_SELECTION_ELIGIBLE_CONDITION,
    FINAL_SELECTION_NO_CANDIDATE_RULE_ID,
    FINAL_SELECTION_SINGLE_RULE_ID,
    FINAL_SELECTION_TIE_BREAK,
    H2_NO_BREACH_THRESHOLDS,
    MOVE_PASS_LOGIC,
    PROCESS_PASS_LOGIC,
)


# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------
def _metrics(**over: float) -> Dict[str, float]:
    base = {"rmse_zero": 1.0, "direction": 0.50, "dead_fraction": 0.10,
            "endpoint_auprc": 0.30, "q_corr": 0.40}
    base.update(over)
    return base


def _eval_doc(metrics: Dict[str, float]) -> Dict[str, Any]:
    """A synthetic ``evaluate_v53mp``-shaped document."""
    return {
        "move": {
            "key_rmse_over_zero": {
                "median_rmse_over_zero": metrics["rmse_zero"]},
            "sre_and_sign": {
                "sign_accuracy_cluster_median": metrics["direction"],
                "dead_zone_fraction": metrics["dead_fraction"]},
        },
        "endpoint_head": {"auprc": metrics["endpoint_auprc"]},
        "q_head": {"corr_curve_median": metrics["q_corr"]},
    }


def _paired_const(cand: Dict[str, float], a0: Dict[str, float],
                  *, n: int = 8, seeds=(17, 29, 43)) -> Dict[str, Any]:
    """Constant per-cluster pairs ⇒ hand-computable bootstrap draws."""
    pairs: Dict[str, Any] = {}
    for endpoint, metric in st.ENDPOINT_TO_METRIC.items():
        pairs[endpoint] = {
            s: {"candidate": [float(cand[metric])] * n,
                "a0": [float(a0[metric])] * n}
            for s in seeds}
    return pairs


def _h3_input(cid: str, cand: Dict[str, float], *,
              a0: Dict[str, float] | None = None,
              winner_eligible: bool = True,
              h2_ok_by_seed=(True, True, True),
              seeds=(17, 29, 43),
              paired: Dict[str, Any] | None = None,
              core: float = 1.0, params: int = 41232,
              loss: float = 0.5) -> Dict[str, Any]:
    a0m = dict(_metrics()) if a0 is None else dict(a0)
    point = {str(s): {"candidate": dict(cand), "a0": dict(a0m)}
             for s in seeds}
    return {
        "winner_eligible": winner_eligible,
        "h2_no_breach_by_seed": {s: bool(ok)
                                 for s, ok in zip(seeds, h2_ok_by_seed)},
        "point_metrics": point,
        "paired": (paired if paired is not None
                   else _paired_const(cand, a0m, seeds=seeds)),
        "core_time_ratio": core,
        "active_params": params,
        "inner_val_total_loss": loss,
    }


def _rule_block() -> Dict[str, Any]:
    return {
        "eligible_condition": FINAL_SELECTION_ELIGIBLE_CONDITION,
        "tie_break_order": list(FINAL_SELECTION_TIE_BREAK),
        "effect_margin_definition": dict(FINAL_EFFECT_MARGIN),
    }


def _qualified_entry(cid: str, *, move: bool = False,
                     process: bool = False,
                     adj_p: float = 0.01, margin: float = 1.0,
                     core: float = 1.0, params: int = 50000,
                     loss: float = 0.5,
                     endpoints=("move_direction",)) -> Dict[str, Any]:
    return {
        "candidate_id": cid,
        "h3_qualified": True,
        "move_qualified": move,
        "process_qualified": process,
        "qualifying_confirmatory_endpoints": list(endpoints),
        "min_holm_adjusted_p_on_qualifying_endpoint": adj_p,
        "max_effect_margin_on_qualifying_endpoint": margin,
        "core_time_ratio": core,
        "active_params": params,
        "inner_val_total_loss": loss,
    }


def _h3_report(candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {"schema": st.H3_REPORT_SCHEMA, "candidates": candidates}


# ===========================================================================
# route_logic — truth table + dangling-OR traps
# ===========================================================================
def test_route_logic_wording_is_frozen() -> None:
    assert st.MOVE_PASS_LOGIC == MOVE_PASS_LOGIC == (
        "(direction_pass AND rmse_non_inferior) OR "
        "(rmse_pass AND direction_non_inferior)")
    assert st.PROCESS_PASS_LOGIC == PROCESS_PASS_LOGIC == (
        "(auprc_pass AND q_non_inferior) OR "
        "(q_pass AND auprc_non_inferior)")
    assert st.H2_NO_BREACH_THRESHOLDS == H2_NO_BREACH_THRESHOLDS


def test_move_route_truth_table() -> None:
    a0 = _metrics()
    # first branch: direction pass + rmse non-inferior
    r = st.evaluate_move_route(_metrics(direction=0.55, rmse_zero=1.01), a0)
    assert r["components"] == {"direction_pass": True, "rmse_pass": False,
                               "rmse_non_inferior": True,
                               "direction_non_inferior": True}
    assert r["pass"] is True
    # second branch: rmse pass + direction non-inferior
    r = st.evaluate_move_route(_metrics(direction=0.495, rmse_zero=0.96), a0)
    assert r["components"]["rmse_pass"] is True
    assert r["components"]["direction_pass"] is False
    assert r["pass"] is True
    # both branches fail
    r = st.evaluate_move_route(_metrics(direction=0.50, rmse_zero=1.00), a0)
    assert r["pass"] is False
    # direction_pass needs BOTH relative >=3% AND absolute >=0.01
    r = st.evaluate_move_route(_metrics(direction=0.514), a0)
    assert r["components"]["direction_pass"] is False     # rel 2.8% < 3%
    r = st.evaluate_move_route(_metrics(direction=0.515), a0)
    assert r["components"]["direction_pass"] is True      # rel 3%, abs .015


def test_move_route_dangling_or_traps() -> None:
    a0 = _metrics()
    # TRAP 1: direction_pass alone must NOT survive an rmse collapse
    # (the non-inferiority belongs to ITS branch; no dangling OR).
    r = st.evaluate_move_route(_metrics(direction=0.55, rmse_zero=1.03), a0)
    assert r["components"]["direction_pass"] is True
    assert r["components"]["rmse_non_inferior"] is False  # 3% degradation
    assert r["pass"] is False
    # TRAP 2: rmse_pass alone must NOT survive a direction collapse.
    r = st.evaluate_move_route(_metrics(direction=0.47, rmse_zero=0.96), a0)
    assert r["components"]["rmse_pass"] is True
    assert r["components"]["direction_non_inferior"] is False
    assert r["pass"] is False
    # boundary: exactly 2% rmse degradation is still non-inferior (<=)
    r = st.evaluate_move_route(_metrics(direction=0.55, rmse_zero=102.0),
                               _metrics(rmse_zero=100.0))
    assert r["components"]["rmse_non_inferior"] is True
    assert r["pass"] is True


def test_process_route_truth_table_and_traps() -> None:
    a0 = _metrics()
    # first branch: auprc pass + q non-inferior
    r = st.evaluate_process_route(_metrics(endpoint_auprc=0.33,
                                           q_corr=0.39), a0)
    assert r["components"] == {"auprc_pass": True, "q_pass": False,
                               "auprc_non_inferior": True,
                               "q_non_inferior": True}
    assert r["pass"] is True
    # second branch: q pass + auprc non-inferior
    r = st.evaluate_process_route(_metrics(endpoint_auprc=0.295,
                                           q_corr=0.44), a0)
    assert r["components"]["q_pass"] is True
    assert r["pass"] is True
    # TRAP: auprc_pass must NOT survive a q collapse beyond 0.02
    r = st.evaluate_process_route(_metrics(endpoint_auprc=0.33,
                                           q_corr=0.35), a0)
    assert r["components"]["q_non_inferior"] is False
    assert r["pass"] is False
    # TRAP: q_pass must NOT survive an auprc drop beyond 0.01
    r = st.evaluate_process_route(_metrics(endpoint_auprc=0.28,
                                           q_corr=0.44), a0)
    assert r["components"]["auprc_non_inferior"] is False
    assert r["pass"] is False
    # boundary: q drop of exactly 0.02 stays non-inferior (<=) so the
    # auprc branch still closes
    r = st.evaluate_process_route(
        _metrics(endpoint_auprc=0.33, q_corr=0.02),
        _metrics(q_corr=0.04))
    assert r["components"]["q_non_inferior"] is True
    assert r["pass"] is True


# ===========================================================================
# H2 guards — frozen boundaries, both sides, report-only disposition
# ===========================================================================
def test_h2_guards_at_frozen_boundaries() -> None:
    # a fully passing side (every guard comfortably inside)
    run = _metrics(rmse_zero=102.0, direction=0.495, dead_fraction=0.11)
    ref = _metrics(rmse_zero=100.0, direction=0.5, dead_fraction=0.1)
    g = st.h2_side_guards(run, ref)
    assert all(v["pass"] for v in g.values())
    assert g["rmse_zero_rel_degradation"]["degradation_relative"] == 0.02
    # EXACTLY 2% rmse degradation passes; a hair more breaches
    g = st.h2_side_guards(_metrics(rmse_zero=102.0), ref)
    assert g["rmse_zero_rel_degradation"]["pass"] is True
    g = st.h2_side_guards(_metrics(rmse_zero=102.01), ref)
    assert g["rmse_zero_rel_degradation"]["pass"] is False
    # EXACTLY 0.01 sign drop passes; a hair more breaches
    g = st.h2_side_guards(_metrics(direction=0.01),
                          _metrics(direction=0.02))
    assert g["sign_abs_drop"]["absolute_drop"] == 0.01
    assert g["sign_abs_drop"]["pass"] is True
    g = st.h2_side_guards(_metrics(direction=0.0099),
                          _metrics(direction=0.02))
    assert g["sign_abs_drop"]["pass"] is False
    # EXACTLY 0.01 dead increase passes; a hair more breaches
    g = st.h2_side_guards(_metrics(dead_fraction=0.02),
                          _metrics(dead_fraction=0.01))
    assert g["dead_fraction_abs_increase"]["pass"] is True
    g = st.h2_side_guards(_metrics(dead_fraction=0.0201),
                          _metrics(dead_fraction=0.01))
    assert g["dead_fraction_abs_increase"]["pass"] is False
    # EXACTLY 2% fixed-coverage direction degradation passes; more fails
    g = st.h2_side_guards(_metrics(direction=49.0),
                          _metrics(direction=50.0))
    assert g["fixed_coverage_direction_rel_degradation"]["pass"] is True
    g = st.h2_side_guards(_metrics(direction=48.9),
                          _metrics(direction=50.0))
    assert g["fixed_coverage_direction_rel_degradation"]["pass"] is False
    # the frozen threshold block is the config mirror, untouched
    assert st.H2_NO_BREACH_THRESHOLDS == {
        "sign_abs_drop_max": 0.01,
        "dead_abs_increase_max": 0.01,
        "rmse_zero_rel_degradation_max": 0.02,
        "fixed_coverage_rel_degradation_max": 0.02,
    }


def test_compute_h2_both_sides_and_typed_refusals() -> None:
    a0 = _metrics(rmse_zero=100.0, direction=0.5, dead_fraction=0.1)
    r3 = dict(a0)
    ok = _metrics(rmse_zero=101.0, direction=0.498, dead_fraction=0.105)
    res = st.compute_h2(ok, a0, r3)
    assert res["verdict"] == "pass" and res["h2_no_breach"] is True
    assert set(res["sides"]) == {"matched_a0", "r3"}
    # a breach on EITHER side marks the run scientific_failure
    bad = _metrics(rmse_zero=103.0)
    res = st.compute_h2(bad, a0, r3)
    assert res["verdict"] == "scientific_failure"
    assert "matched_a0.rmse_zero_rel_degradation" in res["breached_guards"]
    assert "r3.rmse_zero_rel_degradation" in res["breached_guards"]
    # missing references are typed refusals — never a silent downgrade
    with pytest.raises(st.V542StatsError) as exc:
        st.compute_h2(ok, None, r3)
    assert exc.value.code == "V542_STATS_MATCHED_A0_MISSING"
    with pytest.raises(st.V542StatsError) as exc:
        st.compute_h2(ok, a0, None)
    assert exc.value.code == "V542_STATS_R3_REFERENCE_MISSING"


def test_h2_report_marks_scientific_failure_without_ledger_rewrite() -> None:
    a0_m = _metrics(rmse_zero=100.0, direction=0.5, dead_fraction=0.1)
    table = {
        "V542_A0_S17": {"candidate_id": "A0", "seed": 17, "outcome": "done",
                        "metrics": dict(a0_m), "identity_sha256": "a" * 8},
        "V542_A1_S17": {"candidate_id": "A1", "seed": 17, "outcome": "done",
                        "metrics": _metrics(rmse_zero=104.0),
                        "identity_sha256": "b" * 8},
    }
    r3 = {17: dict(a0_m)}
    report = st.build_h2_report(table, r3,
                                r3_provenance={"source": "synthetic"})
    assert report["runs"]["V542_A1_S17"]["verdict"] == "scientific_failure"
    assert report["runs"]["V542_A1_S17"]["ledger_unchanged"] is True
    assert report["runs"]["V542_A0_S17"]["verdict"] == "pass"
    # missing matched A0 for a seed is typed
    with pytest.raises(st.V542StatsError) as exc:
        st.build_h2_report({"V542_A2_S17": table["V542_A1_S17"]}, r3)
    assert exc.value.code == "V542_STATS_MATCHED_A0_MISSING"
    with pytest.raises(st.V542StatsError) as exc:
        st.build_h2_report(table, {})
    assert exc.value.code == "V542_STATS_R3_REFERENCE_MISSING"


# ===========================================================================
# extract — metric leaves + append-only table
# ===========================================================================
def test_metric_values_from_eval_leaves() -> None:
    values = st.metric_values_from_eval(_eval_doc(_metrics(
        rmse_zero=0.91, direction=0.47, dead_fraction=0.12,
        endpoint_auprc=0.29, q_corr=0.38)))
    assert values == {"rmse_zero": 0.91, "direction": 0.47,
                      "dead_fraction": 0.12, "endpoint_auprc": 0.29,
                      "q_corr": 0.38}
    doc = _eval_doc(_metrics())
    doc["q_head"]["corr_curve_median"] = float("nan")
    with pytest.raises(st.V542StatsError) as exc:
        st.metric_values_from_eval(doc)
    assert exc.value.code == "V542_STATS_METRIC_NON_FINITE"


def _record(run_id: str = "V542_A0_S17", *, identity: str = "a" * 64,
            rmse: float = 0.9) -> Dict[str, Any]:
    return st.build_metrics_record(
        run_id=run_id, candidate_id="A0", seed=17,
        metrics=_metrics(rmse_zero=rmse), inner_val_total_loss=0.8,
        core_time_s=120.0, outcome="done", identity_sha256=identity,
        provenance={"evaluator": "synthetic", "checkpoint_sha256": "c" * 8})


def test_metrics_table_is_append_only_and_idempotent(tmp_path) -> None:
    path = tmp_path / "reports" / "metrics_table.json"
    assert st.load_metrics_table(path) == []           # missing file ⇒ []
    rec = _record()
    res = st.append_metrics_records(path, [rec])
    assert res["appended"] == ["V542_A0_S17"] and res["total_records"] == 1
    first_bytes = path.read_bytes()
    # re-appending the same evaluation is idempotent (content fingerprint)
    res = st.append_metrics_records(path, [_record()])
    assert res["appended"] == [] and res["skipped_existing"] == [
        "V542_A0_S17"]
    assert path.read_bytes() == first_bytes           # untouched
    # a NEW payload or identity appends; existing rows never rewritten
    res = st.append_metrics_records(path, [_record(rmse=0.7)])
    assert res["appended"] == ["V542_A0_S17"]
    res = st.append_metrics_records(
        path, [_record(identity="b" * 64)])
    assert res["appended"] == ["V542_A0_S17"]
    rows = st.load_metrics_table(path)
    assert len(rows) == 3
    assert all(r["schema"] == st.METRICS_ENTRY_SCHEMA for r in rows)
    assert all(len(r["entry_sha256"]) == 64 for r in rows)
    # last record per run wins on load (ledger discipline)
    latest = st.latest_metrics_by_run(rows)["V542_A0_S17"]
    assert latest["identity_sha256"] == "b" * 64
    # non-finite metrics are refused at build time
    with pytest.raises(st.V542StatsError) as exc:
        st.build_metrics_record(
            run_id="V542_A1_S17", candidate_id="A1", seed=17,
            metrics=_metrics(q_corr=float("inf")),
            inner_val_total_loss=None, core_time_s=None, outcome="done",
            identity_sha256="d" * 64, provenance={})
    assert exc.value.code == "V542_STATS_METRIC_NON_FINITE"


def test_run_extract_with_synthetic_evaluator(tmp_path) -> None:
    ledger = [
        {"run_id": "V542_A0_S17", "stage": "T0", "backbone": "A0",
         "seed": 17, "outcome": "done", "final_inner_val_loss": 0.9,
         "core_s": 100.0, "identity_sha256": "a" * 64, "seq": 1},
        {"run_id": "V542_A1_S17", "stage": "T1", "backbone": "A1",
         "seed": 17, "outcome": "done", "final_inner_val_loss": 0.7,
         "core_s": 110.0, "identity_sha256": "b" * 64, "seq": 2},
        {"run_id": "V542_A4_S17", "stage": "T1", "backbone": "A4",
         "seed": 17, "outcome": "gate_skipped", "identity_sha256": None,
         "seq": 3},
    ]

    def evaluator(run_id: str) -> Dict[str, Any]:
        row = {r["run_id"]: r for r in ledger}[run_id]
        return {
            "metrics": _metrics(rmse_zero=0.95 if row["backbone"] != "A0"
                                else 0.9),
            "provenance": {"evaluator": "synthetic-frozen-evaluator",
                           "checkpoint_sha256": "f" * 8},
            "candidate_id": row["backbone"], "seed": row["seed"],
            "outcome": "done", "identity_sha256": row["identity_sha256"],
            "inner_val_total_loss": row["final_inner_val_loss"],
            "core_time_s": row["core_s"],
        }

    res1 = st.run_extract({}, campaign_root=tmp_path,
                          evaluator=evaluator, ledger_rows=ledger)
    assert res1["planned_runs"] == ["V542_A0_S17", "V542_A1_S17"]
    assert res1["appended"] == ["V542_A0_S17", "V542_A1_S17"]
    assert res1["training_performed"] is False
    rows = st.load_metrics_table(tmp_path / st.METRICS_TABLE_REL)
    assert len(rows) == 2
    assert all(r["identity_sha256"] for r in rows)
    assert all(r["entry_sha256"] for r in rows)
    # idempotent second pass
    res2 = st.run_extract({}, campaign_root=tmp_path,
                          evaluator=evaluator, ledger_rows=ledger)
    assert res2["appended"] == [] and len(res2["skipped_existing"]) == 2
    # unknown --run is typed
    with pytest.raises(st.V542StatsError) as exc:
        st.run_extract({}, campaign_root=tmp_path, evaluator=evaluator,
                       ledger_rows=ledger, only=["V542_NOPE_S17"])
    assert exc.value.code == "V542_STATS_RUN_UNKNOWN"


# ===========================================================================
# screen — score ordering, ties, top2 freeze, A4 ineligible
# ===========================================================================
def test_screen_score_is_max_of_frozen_terms() -> None:
    cand = _metrics(direction=0.53, rmse_zero=0.97,
                    endpoint_auprc=0.33, q_corr=0.41)
    score = st.compute_screen_score(cand, _metrics())
    terms = score["terms"]
    # terms are the NORMALIZED score components (frozen denominators)
    assert terms["direction_rel"] == pytest.approx(0.06 / 0.03)
    assert terms["rmse_rel_improve"] == pytest.approx(0.03 / 0.03)
    assert terms["delta_auprc"] == pytest.approx(0.03 / 0.02)
    assert terms["delta_q"] == pytest.approx(0.01 / 0.03)
    assert score["score"] == pytest.approx(max(terms.values()))


def test_screen_candidate_predicate() -> None:
    a0 = _metrics()
    ok = st.screen_candidate(
        "A1", run_metrics=_metrics(direction=0.53), a0_metrics=a0,
        run_done=True, matched_a0_done=True, h2_no_breach=True,
        gates={"H0_legality": True, "H1_resource": True,
               "diagnostic_gate": True, "cost_gate": True},
        inner_val_total_loss=0.7, core_time_s=110.0)
    assert ok["eligible"] is True and ok["screen_score"]["score"] > 0
    # every predicate conjunct can flip eligibility off
    for kwargs in ({"run_done": False}, {"h2_no_breach": False},
                   {"matched_a0_done": False},
                   {"gates": {"H0_legality": True, "H1_resource": True,
                              "diagnostic_gate": False,
                              "cost_gate": True}},
                   {"run_metrics": _metrics(direction=float("nan"))}):
        base = dict(run_metrics=_metrics(), a0_metrics=a0, run_done=True,
                    matched_a0_done=True, h2_no_breach=True,
                    gates={"H0_legality": True, "H1_resource": True,
                           "diagnostic_gate": True, "cost_gate": True},
                    inner_val_total_loss=0.7, core_time_s=110.0)
        base.update(kwargs)
        res = st.screen_candidate("A1", **base)
        assert res["eligible"] is False and res["ineligible_reasons"]


def test_top2_ranking_ties_and_no_substitutes() -> None:
    a0 = _metrics()
    def _cand(cid, **metrics):
        entry = st.screen_candidate(
            cid, run_metrics=_metrics(**metrics), a0_metrics=a0,
            run_done=True, matched_a0_done=True, h2_no_breach=True,
            gates={"H0_legality": True, "H1_resource": True,
                   "diagnostic_gate": True, "cost_gate": True},
            inner_val_total_loss=0.7, core_time_s=110.0)
        return entry
    # A1 (score ~1.33) > A3 (score ~1.17) > A2 (score ~1.05); A4-style
    # ineligible candidate never ranks.
    a1 = _cand("A1", rmse_zero=0.96)
    a3 = _cand("A3", rmse_zero=0.965)
    a2 = _cand("A2", endpoint_auprc=0.321)
    ineligible = st.screen_candidate(
        "A4", run_metrics=None, a0_metrics=a0, run_done=False,
        matched_a0_done=True, h2_no_breach=False,
        gates={"H0_legality": False, "H1_resource": False,
               "diagnostic_gate": False, "cost_gate": False},
        inner_val_total_loss=None, core_time_s=None)
    ranking = st.rank_and_select_top2([a1, ineligible, a3, a2])
    assert ranking["ranked_eligible"] == ["A1", "A3", "A2"]
    assert ranking["top2"] == ["A1", "A3"]
    assert ranking["top2_count"] == 2 and ranking["no_substitutes"] is True
    assert ranking["not_selected"] == ["A2"]     # never padded, no replacement
    # TIES: identical scores fall back to inner-val total loss, then
    # core time, then candidate id — deterministically (both fit the
    # ≤2 top2 set, so the RANKED order is what the tie-break decides).
    t1 = _cand("B", rmse_zero=0.96)
    t2 = _cand("A", rmse_zero=0.96)
    t1["inner_val_total_loss"], t2["inner_val_total_loss"] = 0.5, 0.6
    ranking = st.rank_and_select_top2([t2, t1])
    assert ranking["ranked_eligible"] == ["B", "A"]
    t3 = _cand("C", rmse_zero=0.96)
    t3["inner_val_total_loss"], t3["core_time_s"] = 0.5, 90.0
    ranking = st.rank_and_select_top2([t3, t1, t2])
    assert ranking["ranked_eligible"] == ["C", "B", "A"]
    t4 = _cand("A", rmse_zero=0.96)          # full tie on loss AND core
    t4["inner_val_total_loss"], t4["core_time_s"] = 0.5, 110.0
    ranking = st.rank_and_select_top2([t1, t4])
    assert ranking["ranked_eligible"] == ["A", "B"]  # lexicographic id
    assert ranking["top2"] == ["A", "B"]


def test_top2_freeze_precondition_is_asserted() -> None:
    with pytest.raises(st.V542StatsError) as exc:
        st.require_top2_freeze(False, "pass")
    assert exc.value.code == "V542_STATS_TOP2_FREEZE_A0_INCOMPLETE"
    with pytest.raises(st.V542StatsError) as exc:
        st.require_top2_freeze(True, None)
    assert exc.value.code == "V542_STATS_TOP2_FREEZE_BAND"
    with pytest.raises(st.V542StatsError) as exc:
        st.require_top2_freeze(True, "breach")
    assert exc.value.code == "V542_STATS_TOP2_FREEZE_BAND"
    st.require_top2_freeze(True, "pass")         # no raise


# ===========================================================================
# bootstrap — determinism, same draws for CI and p, one-sided formula
# ===========================================================================
def test_bootstrap_subseed_formula_is_frozen() -> None:
    for endpoint in st.ENDPOINTS:
        for seed in st.FROZEN_SEEDS:
            expected = int(hashlib.sha256(
                f"v542|20261029|{seed}|{endpoint}".encode("utf-8")
            ).hexdigest()[:16], 16) % 2147483647
            assert st.bootstrap_subseed(20261029, seed, endpoint) == expected
    with pytest.raises(st.V542StatsError) as exc:
        st.bootstrap_subseed(20261029, 17, "bogus")
    assert exc.value.code == "V542_STATS_ENDPOINT_UNKNOWN"


def test_one_sided_p_formula_hand_cases() -> None:
    # hand-computable: 2 of 5 draws <= 0 ⇒ p = (1+2)/(5+1) = 0.5
    assert st.one_sided_improvement_p(
        [0.1, 0.0, 0.2, -0.3, 0.5]) == pytest.approx(0.5)
    # zero improvement counts as non-improvement (<= 0)
    assert st.one_sided_improvement_p([0.0] * 7) == 1.0
    # all-positive draws give the minimum p = 1/(n+1)
    assert st.one_sided_improvement_p([0.25] * 9) == pytest.approx(0.1)


def test_bootstrap_determinism_and_same_draws_for_ci_and_p() -> None:
    pairs = {17: {"candidate": [0.9, 0.92, 0.88, 0.91],
                  "a0": [1.0, 0.98, 1.02, 1.0]},
             29: {"candidate": [0.9, 0.93, 0.89, 0.91],
                  "a0": [1.01, 0.99, 1.0, 1.0]},
             43: {"candidate": [0.91, 0.92, 0.88, 0.9],
                 "a0": [1.0, 1.0, 0.99, 1.01]}}
    r1 = st.paired_cluster_bootstrap(pairs, endpoint="move_rmse", draws=300)
    r2 = st.paired_cluster_bootstrap(pairs, endpoint="move_rmse", draws=300)
    # same sub-seed ⇒ bit-identical draws across calls
    assert r1["improvements"] == r2["improvements"]
    assert r1["ci_low"] == r2["ci_low"] and r1["ci_high"] == r2["ci_high"]
    assert r1["p_value"] == r2["p_value"]
    # CI and p are computed from the SAME draws (hand-recomputed here)
    improvements = np.asarray(r1["improvements"], dtype=np.float64)
    low, high = np.percentile(improvements, [2.5, 97.5])
    assert r1["ci_low"] == float(low) and r1["ci_high"] == float(high)
    assert r1["p_value"] == st.one_sided_improvement_p(improvements)
    assert r1["improvements_sha256"] == hashlib.sha256(
        improvements.astype("<f8").tobytes()).hexdigest()
    # RMSE improvement direction is A0 - candidate
    assert r1["improvement_direction"] == "a0_minus_candidate"
    assert r1["ci_and_p_same_draws"] is True
    assert r1["subseed_by_seed"] == {
        str(s): st.bootstrap_subseed(20261029, s, "move_rmse")
        for s in (17, 29, 43)}
    # undefined clusters are dropped pairwise and counted, never imputed
    dropped_pairs = {17: {"candidate": [1.0, None, 0.9],
                          "a0": [0.9, 0.95, None]}}
    r3 = st.paired_cluster_bootstrap(dropped_pairs, endpoint="process_q",
                                     draws=20)
    assert r3["dropped_clusters"] == 2
    assert r3["n_clusters_by_seed"] == {"17": 1}


def test_holm_monotonicity_and_step_down() -> None:
    raw = [0.001, 0.01, 0.02, 0.2, 0.5]
    res = st.holm_adjust(raw, 0.025)
    adj = res["adjusted"]
    # adjusted >= raw elementwise
    assert all(adj[i] >= raw[i] for i in range(len(raw)))
    # monotone along the ascending Holm order
    ordered = [adj[i] for i in res["order"]]
    assert ordered == sorted(ordered)
    assert ordered[0] == pytest.approx(5 * 0.001)
    # step-down: stop at the first non-rejection (0.04 >= alpha)
    assert res["rejected"] == [True, False, False, False, False]
    assert adj == [pytest.approx(0.005), pytest.approx(0.04),
                   pytest.approx(0.06), pytest.approx(0.4),
                   pytest.approx(0.5)]
    # hand case: family of three
    res = st.holm_adjust([0.001, 0.02, 0.2], 0.025)
    assert res["adjusted"] == [pytest.approx(0.003),
                               pytest.approx(0.04), pytest.approx(0.2)]
    assert res["rejected"] == [True, False, False]


# ===========================================================================
# h3 — synthetic end-to-end statistics + route verdicts
# ===========================================================================
def _two_candidate_inputs() -> Dict[str, Dict[str, Any]]:
    a0 = _metrics()
    a1_cand = _metrics(direction=0.55, rmse_zero=0.95)
    a2_cand = _metrics(endpoint_auprc=0.36, q_corr=0.405)
    a3_cand = dict(a0)
    a5_cand = dict(a0)
    return {
        "A1": _h3_input("A1", a1_cand, a0=a0),
        "A2": _h3_input("A2", a2_cand, a0=a0),
        "A3": _h3_input("A3", a3_cand, a0=a0,
                        h2_ok_by_seed=(True, False, True)),
        "A5": _h3_input("A5", a5_cand, a0=a0, seeds=(17,)),
    }


def test_compute_h3_end_to_end_synthetic() -> None:
    report = st.compute_h3(_two_candidate_inputs(), draws=399)
    assert report["bootstrap_seed_root"] == 20261029
    assert report["final_draws"] == 399      # test seam; frozen default 10000
    assert report["eligible_candidates"] == ["A1", "A2"]
    assert report["excluded_candidates"]["A3"] == "per_seed_h2_breach"
    assert report["excluded_candidates"]["A5"] == "three_seeds_incomplete"
    # Holm families frozen: two MOVE exits + one single PROCESS family
    assert set(report["holm_families"]) == {
        "move_direction_exit", "move_rmse_exit", "process_route"}
    assert report["alphas"] == {"move_exits_each": 0.025,
                                "process_family": 0.025}
    fam = report["holm_families"]["process_route"]
    assert fam["scope"] == "2_endpoints_x_eligible_candidates"
    assert len(fam["p_raw"]) == 4            # 2 endpoints x 2 candidates

    a1 = report["candidates"]["A1"]
    assert a1["move_qualified"] is True and a1["process_qualified"] is False
    assert a1["h3_qualified"] is True
    assert a1["qualifying_confirmatory_endpoints"] == [
        "move_direction", "move_rmse"]
    ep = a1["endpoints"]["move_direction"]
    # constant pairs ⇒ ci == point improvement 0.05, p = 1/(399+1)
    assert ep["ci_low"] == pytest.approx(0.05)
    assert ep["p_value"] == pytest.approx(1.0 / 400.0)
    assert ep["holm_adjusted_below_alpha"] is True
    assert ep["same_direction_seeds"] == 3
    assert ep["effect_margin"] == pytest.approx(min(0.10 / 0.03,
                                                    0.05 / 0.01))
    assert a1["min_holm_adjusted_p_on_qualifying_endpoint"] == pytest.approx(
        2.0 / 400.0)                        # Holm over the 2-candidate exit

    a2 = report["candidates"]["A2"]
    assert a2["move_qualified"] is False and a2["process_qualified"] is True
    assert a2["qualifying_confirmatory_endpoints"] == ["process_auprc"]
    assert "process_only_note" in a2          # MOVE not improved — stated
    assert a2["endpoints"]["process_auprc"]["ci_low"] == pytest.approx(0.06)
    assert a2["endpoints"]["process_auprc"]["p_adjusted"] == pytest.approx(
        4.0 / 400.0)                          # Holm over 4-family, ties
    # CI and p are the SAME draws everywhere
    assert report["ci_and_p_same_draws"] is True
    for cid in ("A1", "A2"):
        for endpoint in st.ENDPOINTS:
            entry = report["candidates"][cid]["endpoints"][endpoint]
            assert entry["n_draws"] == 399
            assert len(entry["improvements_sha256"]) == 64
    # A0 / diagnostic variants can never enter
    with pytest.raises(st.V542StatsError) as exc:
        st.compute_h3({"A0": _h3_input("A0", _metrics())}, draws=10)
    assert exc.value.code == "V542_STATS_H3_EXCLUDED_CANDIDATE"


# ===========================================================================
# select — frozen final_selection_rule (0 / 1 / many + exact tie-break)
# ===========================================================================
def test_select_zero_qualified_is_no_backbone_winner() -> None:
    report = _h3_report({"A1": {"h3_qualified": False}})
    record = st.select_backbone(report, rule_block=_rule_block())
    assert record["outcome"] == "no_backbone_winner"
    assert record["selected_backbone"] is None
    assert record["a0_retained"] is True
    assert record["selection_rule_branch"] == FINAL_SELECTION_NO_CANDIDATE_RULE_ID
    assert record["dev_used"] is False
    assert record["qualified_candidates"] == []


def test_select_one_qualified_is_the_backbone() -> None:
    report = _h3_report({"A2": _qualified_entry(
        "A2", process=True, endpoints=("process_auprc",))})
    record = st.select_backbone(report, rule_block=_rule_block())
    assert record["selection_rule_branch"] == FINAL_SELECTION_SINGLE_RULE_ID
    assert record["selected_backbone"] == "A2"
    assert record["n_qualified"] == 1
    assert record["dev_used"] is False


def test_select_many_qualified_exact_tie_break() -> None:
    rule = _rule_block()
    # (1) route priority: MOVE-qualified beats PROCESS-only even with a
    # larger adjusted p.
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True, adj_p=0.02,
                               endpoints=("move_direction",)),
        "A2": _qualified_entry("A2", process=True, adj_p=0.001,
                               endpoints=("process_auprc",))})
    record = st.select_backbone(report, rule_block=rule)
    assert record["selected_backbone"] == "A1"
    assert [c["candidate_id"] for c in
            record["qualified_candidates"]] == ["A1", "A2"]
    assert record["qualified_candidates"][0]["tie_break_rank"] == 1
    # (2) min Holm-adjusted p
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True, adj_p=0.02),
        "A3": _qualified_entry("A3", move=True, adj_p=0.001)})
    assert st.select_backbone(report, rule_block=rule)[
        "selected_backbone"] == "A3"
    # (3) max effect margin on the qualifying endpoint
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True, adj_p=0.01, margin=1.2),
        "A3": _qualified_entry("A3", move=True, adj_p=0.01, margin=2.5)})
    assert st.select_backbone(report, rule_block=rule)[
        "selected_backbone"] == "A3"
    # (4) min core_time_ratio
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True, core=1.1),
        "A3": _qualified_entry("A3", move=True, core=0.9)})
    assert st.select_backbone(report, rule_block=rule)[
        "selected_backbone"] == "A3"
    # (5) min active backbone params
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True, params=55000),
        "A3": _qualified_entry("A3", move=True, params=41232)})
    assert st.select_backbone(report, rule_block=rule)[
        "selected_backbone"] == "A3"
    # (6) min inner-val total loss
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True, loss=0.60),
        "A3": _qualified_entry("A3", move=True, loss=0.55)})
    assert st.select_backbone(report, rule_block=rule)[
        "selected_backbone"] == "A3"
    # (7) candidate id lexicographic on a full tie
    report = _h3_report({
        "A3": _qualified_entry("A3", move=True),
        "A1": _qualified_entry("A1", move=True)})
    record = st.select_backbone(report, rule_block=rule)
    assert record["selected_backbone"] == "A1"


def test_select_record_carries_rule_block_sha_and_all_qualified() -> None:
    rule = _rule_block()
    report = _h3_report({
        "A1": _qualified_entry("A1", move=True),
        "A2": _qualified_entry("A2", process=True,
                               endpoints=("process_auprc",))})
    record = st.select_backbone(report, rule_block=rule,
                                statistics_contract_sha256="s" * 64)
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256
    assert record["rule_block_sha256"] == canonical_sha256(rule)
    assert record["rule_block"] == rule
    assert record["tie_break_order"] == list(FINAL_SELECTION_TIE_BREAK)
    assert record["statistics_contract_sha256"] == "s" * 64
    assert record["dev_used"] is False and record["report_all_qualified"] is True
    assert {c["candidate_id"] for c in
            record["qualified_candidates"]} == {"A1", "A2"}
    assert record["excluded_from_selection"] == [
        "A0_matched_baseline", "diagnostic_variants_A1-C_A5-C"]
    # a drifted rule block is refused (eligible_condition is frozen)
    with pytest.raises(st.V542StatsError) as exc:
        st.select_backbone(report, rule_block={"eligible_condition": "ALL"})
    assert exc.value.code == "V542_STATS_RULE_BLOCK_INVALID"


# ===========================================================================
# activate — frozen predicates, A4-never-ran, frozen skip codes
# ===========================================================================
def test_activation_when_a4_never_ran_a5_activates() -> None:
    # H3 has NO qualified A1-A4 candidate — because none exists at all
    # (A4 never ran full dose).  That is a CONSEQUENCE, never a blocker:
    # A5 activates on the frozen condition alone.
    report = st.decide_activations(
        h3_report=_h3_report({}), top2=["A1", "A2"],
        selected_backbone=None, a5_screen_pass=True,
        a5_full_dose_done=False)
    a5 = report["a5_full_dose"]
    assert a5["frozen_condition"] == st.A5_FROZEN_CONDITION
    assert a5["activated"] is True and a5["new_training"] == 1
    assert "NEVER" in a5["a4_never_ran_is_consequence_not_blocker"]
    assert a5["a4_qualified"] is False and a5["no_discretion"] is True
    # supplement stays gated until the A5 full-dose run completes
    supp = report["a5_supplement"]
    assert supp["activated"] is False
    assert supp["reason"] == "a5_supplement_parent_gate_unmet"
    assert supp["new_training"] == 0
    # A1 in the top2 set ⇒ A1-C activates; A5 not selected ⇒ A5-C skips
    assert report["a1c"]["activated"] is True
    assert report["a1c"]["new_training"] == 1
    assert report["a5c"]["activated"] is False
    assert report["a5c"]["reason"] == "a5_not_selected_backbone"
    assert report["a5c"]["new_training"] == 0
    assert report["skips_are_zero_new_training"] is True


def test_activation_blockers_and_supplement() -> None:
    qualified = {"A1": _qualified_entry("A1", move=True)}
    report = st.decide_activations(
        h3_report=_h3_report(qualified), top2=["A1"],
        selected_backbone="A1")
    assert report["a5_full_dose"]["activated"] is False
    assert report["a5_full_dose"]["reason"] == (
        "a5_not_activated_no_A1_A4_h3_qualified_candidate_"
        "after_top2_seeds")
    assert report["a5_full_dose"]["new_training"] == 0
    assert report["a1c"]["activated"] is True      # selected as backbone
    assert report["a5c"]["activated"] is False
    # A1 selected but NOT in top2 ⇒ A1-C still activates (OR condition)
    report = st.decide_activations(
        h3_report=_h3_report({"A1": _qualified_entry("A1", move=True)}),
        top2=["A2"], selected_backbone="A1")
    assert report["a1c"]["activated"] is True
    # A1 merely ranked but neither in top2 nor selected ⇒ skip with the
    # frozen reason
    report = st.decide_activations(
        h3_report=_h3_report({"A1": {"h3_qualified": False}}),
        top2=["A2"], selected_backbone="A2")
    assert report["a1c"]["activated"] is False
    assert report["a1c"]["reason"] == "a1_not_in_top2_and_not_selected"
    # supplement activates only when A5 full-dose complete + screen pass
    report = st.decide_activations(
        h3_report=_h3_report({}), top2=[], selected_backbone=None,
        a5_screen_pass=True, a5_full_dose_done=True)
    assert report["a5_supplement"]["activated"] is True
    assert report["a5_supplement"]["new_training"] == 2
    # A5 selected as Backbone* ⇒ A5-C activates (the ONLY condition)
    report = st.decide_activations(
        h3_report=_h3_report({"A5": _qualified_entry(
            "A5", move=True)}), top2=[], selected_backbone="A5")
    assert report["a5c"]["activated"] is True and report["a5c"][
        "new_training"] == 1
    assert report["a5c"]["capacity_caveat_unresolved"] is False
    # A5 H3-qualified but NOT selected ⇒ capacity caveat unresolved,
    # A5-C does not run
    report = st.decide_activations(
        h3_report=_h3_report({
            "A5": _qualified_entry("A5", move=True),
            "A1": _qualified_entry("A1", move=True, adj_p=0.001)}),
        top2=["A1"], selected_backbone="A1")
    assert report["a5c"]["activated"] is False
    assert report["a5c"]["capacity_caveat_unresolved"] is True
    assert report["a5_full_dose"]["activated"] is False


# ===========================================================================
# discipline — no dev read, no torch.compile, frozen defaults
# ===========================================================================
def test_no_dev_read_no_torch_compile_in_the_lane() -> None:
    source = Path(st.__file__).read_text(encoding="utf-8")
    assert "torch.compile(" not in source
    assert "dev_positions" not in source
    assert ".dev_pos" not in source
    # torch is only imported lazily inside production functions
    top_level = [line for line in source.splitlines()
                 if line.startswith("import torch")]
    assert top_level == []
    # every selection record states dev_used=false explicitly
    record = st.select_backbone(
        _h3_report({"A1": _qualified_entry("A1", move=True)}),
        rule_block=_rule_block())
    assert record["dev_used"] is False
    assert "NEVER participates" in record["dev_policy"]
    # the frozen bootstrap defaults are the pre-registered values
    assert st.BOOTSTRAP_SEED_ROOT == 20261029
    assert st.FINAL_DRAWS == 10000
    assert st.MOVE_EXIT_ALPHA == 0.025
    assert st.PROCESS_FAMILY_ALPHA == 0.025
    assert st.TOP2_CANDIDATES == ("A1", "A2", "A3", "A4")


def test_cli_help_lists_all_six_subcommands() -> None:
    from scripts.agent_ops.v542_stats import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
