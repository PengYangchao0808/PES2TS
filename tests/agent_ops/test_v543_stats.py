"""CPU-fast synthetic tests for the V5.4.3 statistics lane.

The contract under test is the plan's, not the implementation's:

* ``scripts/agent_ops/v543_stats.py`` — §9.3 H0/H1/H2/H3, §8.2 screen score,
  §9.4 statistical contract, §9.5 baseline identity, 附录 B thresholds.
* ``pes2ts_core/pesn/offline/evaluation/v543_eval.py`` — the frozen
  ``pesn_v543_eval_report_v1`` document and the §9.4
  ``paired_curve_bootstrap`` this lane consumes.

Everything here is synthesised: four curves per seed, three seeds, small draw
counts (the frozen 10,000 is a default, never a test requirement).  The real
data certificate is the WSL run of ``extract``/``h2``/``h3`` against the frozen
M0/P0 baseline document.

Discipline being asserted, not just exercised:

* a "≤" gate passes a value exactly at the limit and fails it 1e-9 beyond;
* higher-is-better and lower-is-better metrics never share a sign convention;
* the screen score is hand-computable and never selects more than two
  candidates per track, and its tie-break order is the §8.2 one;
* the §9.3 brackets close their own parentheses (no dangling OR);
* the one-sided bootstrap p-value is ``(1 + #{Δ ≤ 0}) / (draws + 1)``;
* Holm is the textbook step-down (hand-computed);
* a frame-level metric may never enter the bootstrap (§9.4 :1025);
* an identity/SHA drift is a typed refusal (§9.5 :1037);
* M0/P0 compared with itself is Δ = 0, p = 1, CI ∋ 0, no champion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

from scripts.agent_ops import v543_stats as stats

# ---------------------------------------------------------------------------
# synthetic report construction
# ---------------------------------------------------------------------------
#: Baseline pooled values of the nine §9.3 H2 leaves + the curve-level
#: diagnostics (one seed, zero-residual M0/P0).
BASE_METRICS: Dict[str, float] = {
    "move.rmse_zero": 0.90,
    "move.sign_accuracy": 0.80,
    "move.dead_fraction": 0.10,
    "move.fixed_coverage.direction_at_3": 0.80,
    "process.endpoint.auprc": 0.40,
    "process.endpoint.recall_at_fpr_0p05": 0.16,
    "process.endpoint.ece": 0.13,
    "process.q.mae_curve_median": 0.60,
    "process.ordering.pairwise_accuracy_full": 0.80,
}
DIAGNOSTIC_METRICS: Dict[str, float] = {
    "process.endpoint.auroc": 0.65,
    "process.endpoint.brier": 0.21,
    "process.q.corr_curve_median": 0.40,
    "move.fixed_coverage.topk_overlap_at_3": 0.70,
    "move.atom_dead_fraction": 0.05,
    "move.pair_delta_d_mae": 0.04,
    "process.ordering.pairwise_accuracy_cross_ts": 0.86,
}
SPEC_SHA = "spec-0001"
CODE_SHA = "code-0001"
BRIDGE_SHA = "bridge-0001"


def _ckpt_sha(seed: int) -> str:
    return f"ckpt-{seed:04d}"


def _pooled_block(metrics: Mapping[str, float]) -> Dict[str, Any]:
    """Nest the flat metric ids into the evaluator's pooled layout."""
    return {
        "move": {
            "rmse_zero": metrics["move.rmse_zero"],
            "sign_accuracy_cluster_median": metrics["move.sign_accuracy"],
            "dead_fraction": metrics["move.dead_fraction"],
            "fixed_coverage_direction_at_3":
                metrics["move.fixed_coverage.direction_at_3"],
            "fixed_coverage_topk_overlap_at_3":
                metrics["move.fixed_coverage.topk_overlap_at_3"],
            "atom_dead_fraction": metrics["move.atom_dead_fraction"],
            "pair_delta_d_mae": metrics["move.pair_delta_d_mae"],
        },
        "process": {
            "q": {
                "mae_curve_median": metrics["process.q.mae_curve_median"],
                "corr_curve_median": metrics["process.q.corr_curve_median"],
            },
            "ordering": {
                "pairwise_accuracy_full":
                    metrics["process.ordering.pairwise_accuracy_full"],
                "pairwise_accuracy_cross_ts":
                    metrics["process.ordering.pairwise_accuracy_cross_ts"],
            },
            "endpoint": {
                "auprc": metrics["process.endpoint.auprc"],
                "recall_at_fpr_0p05":
                    metrics["process.endpoint.recall_at_fpr_0p05"],
                "ece": metrics["process.endpoint.ece"],
                "auroc": metrics["process.endpoint.auroc"],
                "brier": metrics["process.endpoint.brier"],
            },
        },
    }


def _curve_rows(curves: Mapping[str, Mapping[str, float]]) -> list:
    rows = []
    for curve_id, values in curves.items():
        row: Dict[str, Any] = {"curve_id": curve_id, "n_frames": 4,
                               "move": {}, "q": {}, "ordering": {},
                               "endpoint": {}}
        for metric_id, (block, key) in stats.CURVE_LEAVES.items():
            if metric_id in values:
                row[block][key] = float(values[metric_id])
        rows.append(row)
    return rows


def curves_for(value: float = 0.5, *, n: int = 4) -> Dict[str, Dict[str, float]]:
    """``n`` curves where every bootstrap metric carries ``value``."""
    out: Dict[str, Dict[str, float]] = {}
    for index in range(n):
        curve = f"RXN_{index:04d}"
        values: Dict[str, float] = {}
        for metric_id in stats.CURVE_LEVEL_HIGHER_IS_BETTER:
            values[metric_id] = value
        values["process.q.n_points"] = 3.0
        values["process.q.corr_curve_median"] = 0.4
        values["move.atom_dead_fraction"] = 0.05
        values["process.ordering.pairwise_accuracy_cross_ts"] = 0.86
        out[curve] = values
    return out


def curves_like(metrics: Mapping[str, float], *, n: int = 4,
                prefix: str = "") -> Dict[str, Dict[str, float]]:
    """``n`` curves whose bootstrap metrics all equal ``metrics``.

    ``prefix`` lets a test give each seed a DISJOINT curve set (the real
    campaign's per-seed inner-val samples overlap only partially).
    """
    out: Dict[str, Dict[str, float]] = {}
    for index in range(n):
        curve = f"{prefix}RXN_{index:04d}"
        values: Dict[str, float] = {}
        for metric_id in stats.CURVE_LEVEL_HIGHER_IS_BETTER:
            values[metric_id] = float(metrics.get(metric_id, 0.5))
        values["process.q.n_points"] = 3.0
        values["process.q.corr_curve_median"] = 0.4
        values["move.atom_dead_fraction"] = 0.05
        values["process.ordering.pairwise_accuracy_cross_ts"] = 0.86
        out[curve] = values
    return out


def eval_report(*, metrics: Mapping[str, float] | None = None,
                curves: Mapping[str, Mapping[str, float]] | None = None,
                spec_sha: str = SPEC_SHA,
                frac_curves_ge3: float = 1.0) -> Dict[str, Any]:
    """A minimal but schema-faithful ``pesn_v543_eval_report_v1``."""
    merged = dict(BASE_METRICS)
    merged.update(metrics or {})
    all_metrics = dict(merged)
    all_metrics.update(DIAGNOSTIC_METRICS)
    all_metrics.update({k: v for k, v in (metrics or {}).items()
                        if k not in merged})
    curve_map = dict(curves if curves is not None else curves_for())
    return {
        "schema": stats.EVAL_REPORT_SCHEMA,
        "evaluator_spec_sha256": spec_sha,
        "plan_section": "synthetic",
        "per_curve": _curve_rows(curve_map),
        "pooled": _pooled_block(all_metrics),
        "validity": {
            "q": {
                "n_eval_curves": len(curve_map),
                "median_points_per_curve": 3.0,
                "frac_curves_ge3": frac_curves_ge3,
                "frac_curves_ge5": 0.0,
                "corr_min_points": 3,
                "corr_diagnostic_only": frac_curves_ge3 < 0.70,
            }
        },
    }


def bootstrap_inputs(*, seeds, curves_by_seed=None,
                     higher: Mapping[str, bool] | None = None,
                     seed_order=None,
                     modes: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """The producer's ``bootstrap_inputs`` block (refreshed evaluator shape).

    Mirrors the real document: ``seed_order`` (the campaign seed order),
    ``draws``/``seed_root``, a per-metric ``higher_is_better`` flag, the
    per-seed curve tables, and the two declared modes (multi-seed confirmatory,
    shared-intersection audit).
    """
    higher = dict(higher or stats.CURVE_LEVEL_HIGHER_IS_BETTER)
    metrics: Dict[str, Any] = {}
    for metric_id in stats.CURVE_LEVEL_HIGHER_IS_BETTER:
        per_seed: Dict[str, Any] = {}
        for seed in seeds:
            table = (curves_by_seed or {}).get(seed) or curves_for()
            per_seed[str(seed)] = {
                curve: float(values[metric_id]) for curve, values in table.items()
                if metric_id in values}
        counts = {seed: len(values) for seed, values in per_seed.items()}
        shared_all = len(set.intersection(
            *(set(values) for values in per_seed.values()))) if per_seed else 0
        metrics[metric_id] = {
            "higher_is_better": bool(higher[metric_id]),
            "per_seed_curve_values": per_seed,
            "n_curves_per_seed": counts,
            "n_curves_shared_across_all_seeds": shared_all,
        }
    return {
        "metrics": metrics,
        "unit": "curve (reaction cluster)",
        "draws": 10000,
        "seed_root": 20261102,
        "seed_order": [int(s) for s in (seed_order or seeds)],
        "modes": dict(modes or {
            "per_seed_independent": {
                "api": "paired_curve_bootstrap_multiseed",
                "status": "CONFIRMATORY (use this for H3)",
            },
            "shared_intersection": {
                "api": "paired_curve_bootstrap",
                "status": "audit / compatibility only",
            },
        }),
        "consumption": (
            "paired_curve_bootstrap_multiseed(candidate_tables, "
            "baseline_tables, ...) with the per-seed tables in seed_order"),
    }


def baseline_doc(*, seeds=(17, 29, 43), metrics=None, curves_by_seed=None,
                 spec_sha: str = SPEC_SHA,
                 seed_order=None) -> Dict[str, Any]:
    """A ``pesn_v543_m0p0_baseline_v1`` document with a full SHA chain."""
    seed_list = list(seeds)
    effective = {
        seed: ((curves_by_seed or {}).get(seed)
               or curves_like(metrics or BASE_METRICS))
        for seed in seed_list}
    per_seed = {}
    for seed in seed_list:
        per_seed[str(seed)] = eval_report(
            metrics=metrics, curves=effective[seed], spec_sha=spec_sha)
    return {
        "schema": stats.BASELINE_SCHEMA,
        "campaign_id": "SYNTH",
        "seeds": seed_list,
        "evaluator": {
            "module": "pes2ts_core/pesn/offline/evaluation/v543_eval.py",
            "code_sha256": CODE_SHA,
            "spec_sha256": spec_sha,
            "report_schema": stats.EVAL_REPORT_SCHEMA,
        },
        "bridge_spec_sha256": BRIDGE_SHA,
        "checkpoints": {str(s): {"path": f"/ckpt/s{ s }.pt",
                                 "sha256": _ckpt_sha(s)} for s in seed_list},
        "per_seed": per_seed,
        "per_seed_document_sha256": {str(s): f"doc-{s}" for s in seed_list},
        "bootstrap_inputs": bootstrap_inputs(
            seeds=seed_list, curves_by_seed=effective, seed_order=seed_order),
    }


def bundle(*, run_id: str, candidate_id: str, track: str, seed: int,
           report: Mapping[str, Any], stage: str = "T3",
           run_info: Mapping[str, Any] | None = None,
           spec_sha: str = SPEC_SHA, code_sha: str = CODE_SHA,
           bridge_sha: str = BRIDGE_SHA,
           ckpt_sha: str | None = None) -> Dict[str, Any]:
    """A ``pesn_v543_candidate_eval_bundle_v1`` document."""
    return {
        "schema": stats.CANDIDATE_BUNDLE_SCHEMA,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "track": track,
        "seed": int(seed),
        "stage": stage,
        "outcome": "done",
        "evaluator": {"code_sha256": code_sha, "spec_sha256": spec_sha,
                      "report_schema": stats.EVAL_REPORT_SCHEMA},
        "bridge_spec_sha256": bridge_sha,
        "checkpoint": {"path": f"/ckpt/s{seed}.pt",
                       "sha256": ckpt_sha or _ckpt_sha(seed)},
        "run": dict(run_info or {}),
        "eval_report": dict(report),
    }


#: The frozen screen/h3 CANDIDATE metrics: better on every H2/H3 leaf.
BETTER_METRICS: Dict[str, float] = {
    "move.rmse_zero": 0.85,                     # −5.6 % relative
    "move.sign_accuracy": 0.82,
    "move.dead_fraction": 0.09,
    "move.fixed_coverage.direction_at_3": 0.88,  # +10 % relative, +0.08 abs
    "process.endpoint.auprc": 0.46,             # +0.06 absolute
    "process.endpoint.recall_at_fpr_0p05": 0.18,
    "process.endpoint.ece": 0.11,
    "process.q.mae_curve_median": 0.54,         # −10 % relative
    "process.ordering.pairwise_accuracy_full": 0.83,
}
#: Worse beyond every H2 limit.
WORSE_METRICS: Dict[str, float] = {
    "move.rmse_zero": 0.95,                     # +5.6 % relative
    "move.sign_accuracy": 0.77,
    "move.dead_fraction": 0.13,
    "move.fixed_coverage.direction_at_3": 0.75,
    "process.endpoint.auprc": 0.37,
    "process.endpoint.recall_at_fpr_0p05": 0.13,
    "process.endpoint.ece": 0.17,
    "process.q.mae_curve_median": 0.68,
    "process.ordering.pairwise_accuracy_full": 0.75,
}


def cfg_stub() -> Dict[str, Any]:
    return {"thresholds": {}, "evaluation": {}}


def run_extract_campaign(tmp_path: Path, *, bundles=(),
                         baseline: Mapping[str, Any] | None = None):
    root = Path(tmp_path)
    doc = baseline if baseline is not None else baseline_doc()
    (root / "audit").mkdir(parents=True, exist_ok=True)
    (root / "audit/baseline.json").write_text(json.dumps(doc),
                                             encoding="utf-8")
    paths = []
    for index, item in enumerate(bundles):
        path = root / f"bundle_{index}.json"
        path.write_text(json.dumps(item), encoding="utf-8")
        paths.append(str(path))
    stats.run_extract(cfg_stub(), campaign_root=root,
                      baseline_path=root / "audit/baseline.json",
                      candidates=paths)
    return root


# ---------------------------------------------------------------------------
# 1. H2 — the nine guards, boundaries, directions, zero-baseline rule
# ---------------------------------------------------------------------------
_SPEC = {spec["id"]: spec for spec in stats.H2_SPECS}


def test_h2_has_nine_guards_split_move_four_process_five():
    blocks = [spec["block"] for spec in stats.H2_SPECS]
    assert blocks.count("move") == 4
    assert blocks.count("process") == 5
    assert {spec["baseline_id"] for spec in stats.H2_SPECS
            if spec["block"] == "move"} == {"M0"}
    assert {spec["baseline_id"] for spec in stats.H2_SPECS
            if spec["block"] == "process"} == {"P0"}


def test_h2_relative_boundary_exact_limit_passes_one_nano_beyond_fails():
    spec = _SPEC["move_rmse_zero_rel_degradation"]
    base = 0.90
    at_limit = base * (1.0 + 0.02)
    beyond = at_limit * (1.0 + 1e-9)
    passed = stats.h2_check(spec, candidate=at_limit, baseline=base)
    failed = stats.h2_check(spec, candidate=beyond, baseline=base)
    assert passed["pass"] is True and passed["status"] == "pass"
    assert failed["pass"] is False and failed["status"] == "fail"
    assert passed["limit"] == pytest.approx(0.02)


def test_h2_absolute_drop_boundary():
    spec = _SPEC["move_sign_accuracy_abs_drop"]
    base = 0.80
    assert stats.h2_check(spec, candidate=base - 0.01,
                          baseline=base)["pass"] is True
    assert stats.h2_check(spec, candidate=base - 0.01 - 1e-9,
                          baseline=base)["pass"] is False


def test_h2_absolute_increase_boundary():
    spec = _SPEC["process_ece_abs_increase"]
    base = 0.13
    assert stats.h2_check(spec, candidate=base + 0.02,
                          baseline=base)["pass"] is True
    assert stats.h2_check(spec, candidate=base + 0.02 + 1e-9,
                          baseline=base)["pass"] is False


@pytest.mark.parametrize("spec", stats.H2_SPECS, ids=lambda s: s["id"])
def test_h2_direction_semantics_better_passes_worse_fails(spec):
    metric = spec["metric"]
    spec_by_metric = _SPEC[spec["id"]]
    assert stats.h2_check(spec_by_metric,
                          candidate=BETTER_METRICS[metric],
                          baseline=BASE_METRICS[metric])["pass"] is True
    assert stats.h2_check(spec_by_metric,
                          candidate=WORSE_METRICS[metric],
                          baseline=BASE_METRICS[metric])["pass"] is False


def test_h2_zero_baseline_degrades_to_the_plan_absolute_limit():
    spec = _SPEC["process_q_mae_rel_degradation"]
    zero_base = stats.h2_check(spec, candidate=0.005, baseline=0.0)
    assert zero_base["mode"] == "absolute_fallback_zero_baseline_ratio"
    assert zero_base["pass"] is True
    assert zero_base["limit"] == pytest.approx(
        stats.ZERO_BASELINE_ABS_MAX_RATIO_CLASS)
    assert stats.h2_check(spec, candidate=0.0105,
                          baseline=0.0)["pass"] is False


def test_h2_missing_metric_is_a_typed_refusal():
    spec = _SPEC["move_rmse_zero_rel_degradation"]
    with pytest.raises(stats.V543StatsError) as exc:
        stats.h2_check(spec, candidate=None, baseline=0.9)
    assert exc.value.code == "V543_STATS_METRIC_MISSING"


def test_h2_self_comparison_passes_every_guard_on_every_seed(tmp_path):
    root = run_extract_campaign(tmp_path)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.build_h2_report(table=table, self_comparison=True)
    summary = report["runs"]["M0/P0"]
    assert summary["verdict"] == "pass"
    assert summary["seeds_present"] == [17, 29, 43]
    assert summary["breached_seeds"] == []
    for seed in ("17", "29", "43"):
        entry = summary["per_seed"][seed]
        assert entry["n_checks"] == 9
        for check in entry["checks"].values():
            assert check["observed"] == pytest.approx(0.0)
            assert check["pass"] is True


def test_h2_candidate_worse_than_baseline_is_scientific_failure(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1",
                      track="move", seed=s,
                      report=eval_report(metrics=WORSE_METRICS))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.build_h2_report(table=table)
    summary = report["runs"]["M1"]
    assert summary["verdict"] == "scientific_failure"
    assert sorted(summary["breached_seeds"]) == ["S17", "S29", "S43"]


# ---------------------------------------------------------------------------
# 2. screen — §8.2 score, tie-break, ≤2 top-ups
# ---------------------------------------------------------------------------
def test_screen_score_hand_computed():
    score = stats.screen_terms(candidate=BETTER_METRICS,
                               baseline=BASE_METRICS, track="move")
    # rmse: (0.90 − 0.85)/0.90 = 0.05555… / 0.03 = 1.8518…
    # direction: (0.88 − 0.80)/0.80 = 0.10 / 0.03 = 3.3333…
    assert score["scored_terms"]["move_rmse_rel_improve"] == pytest.approx(
        (0.05 / 0.90) / 0.03)
    assert score["scored_terms"]["move_fixed_coverage_direction_rel"] == \
        pytest.approx(0.10 / 0.03)
    assert score["score"] == pytest.approx(0.10 / 0.03)

    process = stats.screen_terms(candidate=BETTER_METRICS,
                                 baseline=BASE_METRICS, track="process")
    # auprc: 0.06/0.02 = 3.0; q: 0.10/0.05 = 2.0; ordering: 0.03/0.02 = 1.5
    assert process["scored_terms"]["process_auprc_delta"] == pytest.approx(3.0)
    assert process["scored_terms"]["process_q_mae_rel_improve"] == \
        pytest.approx(2.0)
    assert process["scored_terms"]["process_ordering_abs_improve"] == \
        pytest.approx(1.5)
    assert process["score"] == pytest.approx(3.0)


def test_screen_formula_uses_only_the_tracks_own_terms():
    move = stats.screen_terms(candidate=BETTER_METRICS,
                              baseline=BASE_METRICS, track="move")
    assert set(move["scored_terms"]) == {"move_rmse_rel_improve",
                                         "move_fixed_coverage_direction_rel"}
    process = stats.screen_terms(candidate=BETTER_METRICS,
                                 baseline=BASE_METRICS, track="process")
    assert "move_rmse_rel_improve" not in process["scored_terms"]


def test_screen_zero_denominator_refuses_instead_of_inventing_a_term():
    bad_base = dict(BASE_METRICS)
    bad_base["move.rmse_zero"] = 0.0
    with pytest.raises(stats.V543StatsError) as exc:
        stats.screen_terms(candidate=BETTER_METRICS, baseline=bad_base,
                           track="move")
    assert exc.value.code == "V543_STATS_REFERENCE_ZERO"


def _screened(candidate_id, score, *, loss=None, core=None, params=None):
    return {
        "candidate_id": candidate_id,
        "eligible": True,
        "screen_score": {"score": score},
        "inner_val_total_loss": loss,
        "core_time_s": core,
        "n_trainable_params": params,
    }


def test_screen_tie_break_order_is_the_plan_order():
    # identical score → inner-val loss → core time → fewer params → id
    rows = [
        _screened("M3", 1.0, loss=2.0, core=10.0, params=36000),
        _screened("M1", 1.0, loss=1.0, core=99.0, params=36000),
        _screened("M2", 1.0, loss=1.0, core=9.0, params=36000),
        _screened("M0x", 1.0, loss=1.0, core=9.0, params=35000),
        _screened("M4x", 1.0, loss=1.0, core=9.0, params=35000),
    ]
    ranked = stats._rank_top2(rows)
    assert ranked["top2"] == ["M0x", "M4x"]      # fewer params, then id
    assert ranked["not_selected"] == ["M2", "M1", "M3"]
    assert ranked["ties_broken"]


def test_screen_never_selects_more_than_two_and_emits_appendix_c_ids():
    rows = [_screened(f"M{i}", 5.0 - i) for i in (1, 2, 3)]
    ranked = stats._rank_top2(rows)
    assert ranked["top2"] == ["M1", "M2"]
    assert ranked["not_selected"] == ["M3"]
    assert [f"V543_{cid}_S43" for cid in ranked["top2"]] == [
        "V543_M1_S43", "V543_M2_S43"]


def test_screen_report_structure_and_single_candidate_top2(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s, report=eval_report(metrics=BETTER_METRICS),
                      run_info={"inner_val_total_loss": 1.0, "core_time_s": 9.0,
                                "n_trainable_params": 36000})
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.build_screen_report(table=table)
    move = report["tracks"]["move"]
    assert move["top2"] == ["M1"]
    assert move["t5_run_ids"] == ["V543_M1_S43"]
    assert move["top2_max_candidates"] == 2
    assert report["scientific_claim"] is False
    assert report["t5_activation"]["emitted"] == 1
    process = report["tracks"]["process"]
    assert process["top2"] == []          # a MOVE candidate is not PROCESS
    assert process["t5_run_ids"] == []


def test_screen_excludes_a_candidate_with_an_h2_breach(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s, report=eval_report(metrics=WORSE_METRICS))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.build_screen_report(table=table)
    screened = report["tracks"]["move"]["screened"]
    assert screened and screened[0]["eligible"] is False
    assert any(reason.startswith("H2_breach")
               for reason in screened[0]["ineligible_reasons"])
    assert report["tracks"]["move"]["top2"] == []


# ---------------------------------------------------------------------------
# 3. H3 — bracket tiers, p-value, Holm, refusals, self-comparison
# ---------------------------------------------------------------------------
def _effects(rmse_rel, direction_rel, direction_abs, *,
             auprc_delta=0.0, q_rel=0.0):
    return {
        "move_rmse": {
            "metric": "move.rmse_zero", "candidate": 0.9 * (1 - rmse_rel),
            "baseline": 0.9, "absolute_improvement": 0.9 * rmse_rel,
            "relative_improvement": rmse_rel, "higher_is_better": False},
        "move_direction": {
            "metric": "move.fixed_coverage.direction_at_3",
            "candidate": 0.8 + direction_abs, "baseline": 0.8,
            "absolute_improvement": direction_abs,
            "relative_improvement": direction_rel, "higher_is_better": True},
        "process_auprc": {
            "metric": "process.endpoint.auprc", "candidate": 0.4 + auprc_delta,
            "baseline": 0.4, "absolute_improvement": auprc_delta,
            "relative_improvement": auprc_delta / 0.4,
            "higher_is_better": True},
        "process_q": {
            "metric": "process.q.mae_curve_median",
            "candidate": 0.6 * (1 - q_rel), "baseline": 0.6,
            "absolute_improvement": 0.6 * q_rel, "relative_improvement": q_rel,
            "higher_is_better": False},
    }


def test_move_bracket_pass_thresholds():
    # direction_pass needs ≥3 % relative AND ≥0.01 absolute
    route = stats.move_route(_effects(0.0, 0.03, 0.024))
    assert route["components"]["direction_pass"] is True
    route = stats.move_route(_effects(0.0, 0.03, 0.009))     # abs short
    assert route["components"]["direction_pass"] is False
    route = stats.move_route(_effects(0.0, 0.029, 0.024))    # rel short
    assert route["components"]["direction_pass"] is False


def test_move_bracket_non_inferiority_limits():
    route = stats.move_route(_effects(-0.02, -0.02, 0.0))
    assert route["components"]["rmse_non_inferior"] is True
    assert route["components"]["direction_non_inferior"] is True
    route = stats.move_route(_effects(-0.021, -0.021, 0.0))
    assert route["components"]["rmse_non_inferior"] is False
    assert route["components"]["direction_non_inferior"] is False


def test_move_bracket_qualification_has_no_dangling_or():
    branch1 = stats.move_route(_effects(0.0, 0.05, 0.05))    # direction passes
    ok, detail = stats._bracket_qualified(branch1["components"],
                                          confirmed_a=True, confirmed_b=False)
    assert ok is True and detail["branch_direction_pass"] is True

    branch2 = stats.move_route(_effects(0.05, 0.0, 0.0))     # rmse passes
    ok, detail = stats._bracket_qualified(branch2["components"],
                                          confirmed_a=False, confirmed_b=True)
    assert ok is True and detail["branch_rmse_pass"] is True

    neither = stats.move_route(_effects(0.0, 0.0, 0.0))
    ok, detail = stats._bracket_qualified(neither["components"],
                                          confirmed_a=False, confirmed_b=False)
    assert ok is False and detail["branch_direction_pass"] is False
    assert detail["branch_rmse_pass"] is False

    # unconfirmed pass component cannot qualify even if the point effect is big
    big = stats.move_route(_effects(0.10, 0.10, 0.10))
    ok, _ = stats._bracket_qualified(big["components"], confirmed_a=True,
                                     confirmed_b=False)
    assert ok is True
    ok, _ = stats._bracket_qualified(big["components"], confirmed_a=False,
                                     confirmed_b=True)
    assert ok is True
    ok, _ = stats._bracket_qualified(big["components"], confirmed_a=False,
                                     confirmed_b=False)
    assert ok is False


def test_move_bracket_inferior_on_the_other_side_kills_the_branch():
    # direction passes but rmse regressed >2 % → the bracket must fail
    route = stats.move_route(_effects(-0.05, 0.05, 0.05))
    ok, _ = stats._bracket_qualified(route["components"], True, False)
    assert ok is False


PROCESS_BASE = dict(BASE_METRICS)


def test_process_bracket_both_branches():
    metrics = {"candidate": dict(BETTER_METRICS), "baseline": dict(PROCESS_BASE)}
    route = stats.process_route(_effects(0.0, 0.0, 0.0, auprc_delta=0.06),
                                metrics)
    assert route["components"]["auprc_pass"] is True
    assert route["components"]["q_non_inferior"] is True
    assert route["components"]["recall_non_inferior"] is True
    assert route["components"]["calibration_non_inferior"] is True
    assert route["components"]["ordering_non_inferior"] is True
    ok, detail = stats._bracket_qualified(route["components"], True, False)
    assert ok is True and detail["branch_auprc_pass"] is True

    # q_pass branch: q MAE −10 % AND ordering non-inferior
    route = stats.process_route(_effects(0.0, 0.0, 0.0, q_rel=0.10), metrics)
    assert route["components"]["q_pass"] is True
    ok, detail = stats._bracket_qualified(route["components"], False, True)
    assert ok is True and detail["branch_q_pass"] is True


def test_process_bracket_recall_failure_blocks_the_auprc_branch():
    metrics = {"candidate": dict(BETTER_METRICS), "baseline": dict(PROCESS_BASE)}
    metrics["candidate"]["process.endpoint.recall_at_fpr_0p05"] = 0.10  # −0.06
    route = stats.process_route(_effects(0.0, 0.0, 0.0, auprc_delta=0.06),
                                metrics)
    assert route["components"]["recall_non_inferior"] is False
    ok, _ = stats._bracket_qualified(route["components"], True, False)
    assert ok is False


def test_process_bracket_q_pass_requires_ordering_non_inferior():
    metrics = {"candidate": dict(BETTER_METRICS), "baseline": dict(PROCESS_BASE)}
    metrics["candidate"]["process.ordering.pairwise_accuracy_full"] = 0.75  # −0.05
    route = stats.process_route(_effects(0.0, 0.0, 0.0, q_rel=0.10), metrics)
    assert route["components"]["q_pass"] is False
    assert route["components"]["ordering_non_inferior"] is False


def test_effect_margins_are_the_plan_formulas():
    effects = _effects(0.06, 0.06, 0.02, auprc_delta=0.04, q_rel=0.10)
    assert stats.effect_margin("move_rmse", effects) == pytest.approx(2.0)
    assert stats.effect_margin("move_direction", effects) == pytest.approx(2.0)
    assert stats.effect_margin("process_auprc", effects) == pytest.approx(2.0)
    assert stats.effect_margin("process_q", effects) == pytest.approx(2.0)
    # direction is the min of the two readings
    effects = _effects(0.06, 0.06, 0.005)
    assert stats.effect_margin("move_direction", effects) == pytest.approx(0.5)


def test_bootstrap_p_all_positive_is_one_over_draws_plus_one():
    curves = {curve: 0.5 for curve in curves_for(0.5)}
    better = {curve: 0.4 for curve in curves}
    out = stats.paired_bootstrap(candidate_tables=[better],
                                 baseline_tables=[curves],
                                 metric_id="move.rmse_zero", draws=10)
    assert out["n_draws_nonpositive"] == 0
    assert out["p_one_sided"] == pytest.approx(1.0 / 11.0)
    assert out["ci95"][0] > 0.0


def test_bootstrap_p_all_zero_is_one():
    curves = {curve: 0.5 for curve in curves_for(0.5)}
    out = stats.paired_bootstrap(candidate_tables=[curves],
                                 baseline_tables=[curves],
                                 metric_id="move.rmse_zero", draws=10)
    assert out["n_draws_nonpositive"] == 10
    assert out["p_one_sided"] == pytest.approx(1.0)
    assert out["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_bootstrap_delegates_to_the_frozen_implementation():
    curves = {curve: 0.5 for curve in curves_for(0.5)}
    out = stats.paired_bootstrap(candidate_tables=[curves],
                                 baseline_tables=[curves],
                                 metric_id="move.rmse_zero", draws=5)
    assert "paired_curve_bootstrap" in out["delegated_to"]
    assert out["draws"] == 5
    assert out["seed_root"] == stats.BOOTSTRAP_SEED_ROOT


def test_frame_level_metric_bootstrap_is_refused():
    curves = {curve: 0.5 for curve in curves_for(0.5)}
    with pytest.raises(stats.V543StatsError) as exc:
        stats.paired_bootstrap(candidate_tables=[curves],
                               baseline_tables=[curves],
                               metric_id="process.endpoint.recall_at_fpr_0p05",
                               draws=5)
    assert exc.value.code == "V543_STATS_FRAME_LEVEL_BOOTSTRAP"
    with pytest.raises(stats.V543StatsError):
        stats.require_curve_level("process.endpoint.recall_at_fpr_0p05")


def test_holm_hand_computed():
    result = stats.holm_adjust([0.001, 0.04, 0.03], 0.025)
    assert result["adjusted"] == [pytest.approx(0.003),
                                  pytest.approx(0.06),
                                  pytest.approx(0.06)]
    assert result["rejected"] == [True, False, False]
    assert result["family_size"] == 3


def test_holm_stops_at_the_first_non_rejection():
    result = stats.holm_adjust([0.03, 0.001], 0.025)
    # sorted: 0.001 (rank 0) → 2×0.001 = 0.002; 0.03 (rank 1) → 1×0.03 = 0.03
    assert result["adjusted"] == [pytest.approx(0.03), pytest.approx(0.002)]
    assert result["rejected"] == [False, True]


def test_holm_rejects_an_empty_family():
    with pytest.raises(stats.V543StatsError):
        stats.holm_adjust([], 0.025)


# ---------------------------------------------------------------------------
# 4. identity chain (§9.5) and instrument drift
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("spec_sha", "spec-9999"),
    ("code_sha", "code-9999"),
    ("bridge_sha", "bridge-9999"),
    ("ckpt_sha", "ckpt-9999"),
])
def test_identity_chain_mismatch_is_a_typed_refusal(tmp_path, field, value):
    kwargs: Dict[str, Any] = {field: value}
    bad = bundle(run_id="V543_M1_S17", candidate_id="M1", track="move",
                 seed=17, report=eval_report(), **kwargs)
    with pytest.raises(stats.V543StatsError) as exc:
        stats.run_extract(cfg_stub(), campaign_root=tmp_path,
                          baseline_path=_write_baseline(tmp_path),
                          candidates=[_write_bundle(tmp_path, bad)])
    assert exc.value.code == "V543_STATS_IDENTITY_MISMATCH"


def _write_baseline(root: Path, doc: Mapping[str, Any] | None = None) -> Path:
    (Path(root) / "audit").mkdir(parents=True, exist_ok=True)
    path = Path(root) / "audit/baseline.json"
    path.write_text(json.dumps(doc if doc is not None else baseline_doc()),
                    encoding="utf-8")
    return path


def _write_bundle(root: Path, doc: Mapping[str, Any]) -> str:
    path = Path(root) / f"bundle_{doc['run_id']}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_bootstrap_input_drift_is_refused(tmp_path):
    doc = baseline_doc()
    metrics = doc["bootstrap_inputs"]["metrics"]["move.rmse_zero"]
    metrics["per_seed_curve_values"]["17"]["RXN_0000"] = 0.123
    with pytest.raises(stats.V543StatsError) as exc:
        stats.run_extract(cfg_stub(), campaign_root=tmp_path,
                          baseline_path=_write_baseline(tmp_path, doc))
    assert exc.value.code == "V543_STATS_BOOTSTRAP_INPUT_DRIFT"


def test_extract_is_append_only_and_idempotent(tmp_path):
    root = run_extract_campaign(tmp_path)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    first = len(table["entries"])
    stats.run_extract(cfg_stub(), campaign_root=root,
                      baseline_path=root / "audit/baseline.json")
    again = stats.load_table(root / stats.METRICS_TABLE_REL)
    assert len(again["entries"]) == first
    assert again["append_log"][-1]["added"] == []
    assert len(again["append_log"][-1]["skipped_idempotent"]) == first


# ---------------------------------------------------------------------------
# 5. H0 / H1
# ---------------------------------------------------------------------------
def test_h0_counts_and_pending_default():
    report = stats.h0_checklist()
    assert report["verdict"] == "incomplete"
    assert report["counts"]["n_plan_bullets"] == 10
    assert report["counts"]["n_atomic_checks"] == 14
    assert report["counts"]["n_pending"] == 14
    assert report["counts"]["brief_expected_count"] == 12


def test_h0_fail_is_invalid_run_and_pass_is_pass():
    evidence = {"schema": stats.H0_EVIDENCE_SCHEMA,
                "items": {item: {"status": "pass", "evidence": "t"}
                          for item in stats.H0_ITEM_IDS}}
    assert stats.h0_checklist(evidence)["verdict"] == "pass"
    evidence["items"]["h0.encoder_forward_exactly_once"] = {"status": "fail",
                                                            "evidence": "2x"}
    report = stats.h0_checklist(evidence)
    assert report["verdict"] == "invalid_run"
    assert report["counts"]["n_fail"] == 1


def test_h0_unknown_item_is_refused():
    evidence = {"schema": stats.H0_EVIDENCE_SCHEMA,
                "items": {"h0.invented": {"status": "pass"}}}
    with pytest.raises(stats.V543StatsError) as exc:
        stats.h0_checklist(evidence)
    assert exc.value.code == "V543_STATS_H0_ITEM_UNKNOWN"


def test_h1_without_input_is_pending_never_pass():
    report = stats.evaluate_h1(None)
    assert report["data_status"] == "no_input_measured"
    assert report["per_candidate"] == {}


def test_h1_complete_input_passes_every_check():
    report = stats.evaluate_h1({
        "schema": stats.H1_INPUT_SCHEMA,
        "per_candidate": {"M1": {
            "run_id": "V543_M1_S17", "track": "move", "decoder_id": "M1",
            "single_lane": True, "swap_events": 0, "oom_events": 0,
            "host_peak_gib": 8.0,
            "vram_peak_gib": 9.0, "vram_limit_gib": 16.0,
            "core_time_candidate_s": 1800.0,
            "core_time_denominator_s": 1000.0,
            "core_time_denominator_source": "same_env_m0p0_v543",
            "p95_latency_candidate_s": 0.020, "p95_latency_a0_s": 0.012,
            "n_trainable_params": 36000,
        }}})
    entry = report["per_candidate"]["M1"]
    assert entry["verdict"] == "pass"
    assert entry["failed"] == []
    assert entry["checks"]["core_time_ratio_le_2"]["ratio"] == pytest.approx(1.8)
    assert entry["checks"]["p95_latency_ratio_le_2"]["pass"] is True
    assert report["data_status"] == "complete"


def test_h1_cost_ratio_breach_is_gate_skipped_not_silently_dropped():
    report = stats.evaluate_h1({
        "schema": stats.H1_INPUT_SCHEMA,
        "per_candidate": {"M1": {
            "single_lane": True, "swap_events": 0, "oom_events": 0,
            "host_peak_gib": 8.0, "vram_peak_gib": 9.0, "vram_limit_gib": 16.0,
            "core_time_candidate_s": 2500.0,
            "core_time_denominator_s": 1000.0,
            "core_time_denominator_source": "same_env_m0p0_v543",
            "p95_latency_candidate_s": 0.020, "p95_latency_a0_s": 0.012,
            "n_trainable_params": 36000,
        }}})
    entry = report["per_candidate"]["M1"]
    assert entry["verdict"] == "resource_failure"
    assert entry["disposition"] == "gate_skipped_cost_ratio_exceeds_2"


def test_h1_v542_archive_denominator_is_refused():
    with pytest.raises(stats.V543StatsError) as exc:
        stats.evaluate_h1({
            "schema": stats.H1_INPUT_SCHEMA,
            "per_candidate": {"M1": {
                "core_time_candidate_s": 1.0,
                "core_time_denominator_s": 1.0,
                "core_time_denominator_source": "v542_archive",
            }}})
    assert exc.value.code == "V543_STATS_COST_DENOMINATOR_SOURCE"


def test_h1_partial_input_is_pending_not_pass():
    report = stats.evaluate_h1({
        "schema": stats.H1_INPUT_SCHEMA,
        "per_candidate": {"M1": {"single_lane": True, "swap_events": 0,
                                 "oom_events": 0}}})
    entry = report["per_candidate"]["M1"]
    assert entry["verdict"] == "pending_wp7_profiling"
    assert "host_peak_within_12gib" in entry["pending"]
    assert report["data_status"] == "pending_wp7_profiling"


# ---------------------------------------------------------------------------
# 6. H3 end-to-end: self-comparison, qualification, selection
# ---------------------------------------------------------------------------
def test_h3_self_comparison_is_zero_effect_and_not_qualified(tmp_path):
    root = run_extract_campaign(tmp_path)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=50, self_comparison=True)
    candidate = report["candidates"]["M0/P0"]
    for endpoint, entry in candidate["endpoints"].items():
        assert entry["p_one_sided"] == pytest.approx(1.0)
        assert entry["p_adjusted_holm"] == pytest.approx(1.0)
        assert entry["ci95"][0] <= 0.0 <= entry["ci95"][1]
        assert entry["ci_lower_gt_zero"] is False
        assert entry["same_direction_seeds"] == 0
        assert entry["effect_margin"] == pytest.approx(0.0)
        assert entry["confirmed"] is False
    assert candidate["move_qualified"] is False
    assert candidate["process_qualified"] is False
    assert candidate["h3_qualified"] is False
    assert report["scientific_claim"] is False


def test_h3_better_candidate_qualifies_both_tracks_and_selects(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s,
                      report=eval_report(metrics=BETTER_METRICS,
                                         curves=curves_like(BETTER_METRICS)),
                      run_info={"inner_val_total_loss": 1.0, "core_time_s": 9.0,
                                "n_trainable_params": 36000})
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=200)
    candidate = report["candidates"]["M1"]
    assert candidate["endpoints"]["move_direction"]["same_direction_seeds"] == 3
    assert candidate["endpoints"]["move_direction"]["ci_lower_gt_zero"] is True
    assert candidate["endpoints"]["move_direction"]["p_adjusted_holm"] < 0.025
    assert candidate["move_qualified"] is True
    assert candidate["process_qualified"] is True
    assert candidate["h3_qualified"] is True
    assert candidate["min_holm_adjusted_p_on_qualifying_endpoint"] is not None
    assert candidate["max_effect_margin_on_qualifying_endpoint"] > 1.0
    record = stats.select_champions(report)
    assert record["screen_score_used"] is False
    assert record["tracks"]["move"]["champion"] == "M1"
    assert record["tracks"]["process"]["champion"] == "M1"


def test_h3_worse_candidate_is_excluded_by_h2_before_any_test(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s, report=eval_report(metrics=WORSE_METRICS))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=50)
    assert report["eligible_candidates"] == []
    assert "per_seed_h2_breach" in report["excluded_candidates"]["M1"]
    record = stats.select_champions(report)
    assert record["tracks"]["move"]["champion"] is None
    assert record["tracks"]["move"]["retained_baseline"] == "M0"
    assert record["tracks"]["process"]["retained_baseline"] == "P0"


def test_h3_requires_three_seeds(tmp_path):
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s, report=eval_report(metrics=BETTER_METRICS))
               for s in (17, 29)]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=20)
    assert "three_seeds_incomplete" in report["excluded_candidates"]["M1"]


def test_h3_single_seed_same_direction_is_not_enough(tmp_path):
    seeds = (17, 29, 43)
    # only seed17 improves; 29 and 43 regress but stay inside the H2 limits
    inside: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        if seed == 17:
            inside[seed] = dict(BETTER_METRICS)
        else:
            inside[seed] = {"move.rmse_zero": 0.915,          # +1.7 %
                            "move.sign_accuracy": 0.795,
                            "move.dead_fraction": 0.105,
                            "move.fixed_coverage.direction_at_3": 0.795,
                            "process.endpoint.auprc": 0.395,
                            "process.endpoint.recall_at_fpr_0p05": 0.15,
                            "process.endpoint.ece": 0.14,
                            "process.q.mae_curve_median": 0.62,
                            "process.ordering.pairwise_accuracy_full": 0.79}
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s, report=eval_report(metrics=inside[s]))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=50)
    candidate = report["candidates"]["M1"]
    assert candidate["move_qualified"] is False
    assert candidate["process_qualified"] is False
    assert candidate["h3_qualified"] is False


def test_q_corr_is_never_a_conclusion_and_the_flag_is_rederived(tmp_path):
    report = eval_report(frac_curves_ge3=0.5)
    entry = stats.metrics_entry(
        run_id="V543_M1_S17", candidate_id="M1", track="move", seed=17,
        stage="T3", role="candidate", report=report,
        identity={"evaluator": {"code_sha256": CODE_SHA,
                               "spec_sha256": SPEC_SHA},
                  "bridge_spec_sha256": BRIDGE_SHA,
                  "checkpoint": {"sha256": _ckpt_sha(17)}},
        bootstrap_inputs=None, source={"path": "synthetic"})
    validity = entry["q_validity"]
    assert validity["q_corr_role"] == "diagnostic_only"
    assert validity["q_corr_diagnostic_only_derived"] is True
    assert validity["q_corr_is_conclusion"] is False


# ---------------------------------------------------------------------------
# 6b. the refreshed producer contract: multi-seed CONFIRMATORY bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_contract_reads_the_producer_block():
    doc = baseline_doc(seed_order=[43, 29, 17])
    contract = stats.bootstrap_contract(doc, seeds=[17, 29, 43])
    assert contract["seed_order"] == [43, 29, 17]
    assert contract["draws"] == 10000
    assert contract["seed_root"] == 20261102
    assert contract["metrics_higher_is_better"]["move.rmse_zero"] is False
    assert contract["metrics_higher_is_better"][
        "move.fixed_coverage.direction_at_3"] is True
    assert contract["metrics_higher_is_better"][
        "process.endpoint.auprc"] is True
    assert contract["confirmatory_api"] == "paired_curve_bootstrap_multiseed"
    assert contract["audit_api"] == "paired_curve_bootstrap"
    assert "bootstrap_inputs" in contract["source"]
    assert contract["n_curves_shared_across_all_seeds"]["move.rmse_zero"] == 4


def test_bootstrap_contract_falls_back_to_the_frozen_defaults():
    contract = stats.bootstrap_contract({}, seeds=[17, 29, 43])
    assert contract["seed_order"] == [17, 29, 43]
    assert contract["draws"] == stats.BOOTSTRAP_DRAWS
    assert contract["seed_root"] == stats.BOOTSTRAP_SEED_ROOT
    assert "frozen plan defaults" in contract["source"]


def test_multiseed_bootstrap_p_is_the_draw_statistic():
    curves = {curve: 0.5 for curve in curves_for(0.5)}
    better = {curve: 0.4 for curve in curves}
    out = stats.paired_bootstrap_multiseed(
        candidate_tables=[better, better, better],
        baseline_tables=[curves, curves, curves],
        metric_id="move.rmse_zero", draws=10)
    assert out["mode"] == "per_seed_independent"
    assert out["n_curves_per_seed"] == [4, 4, 4]
    assert out["n_curves_total"] == 12
    assert out["n_draws_nonpositive"] == 0
    assert out["p_one_sided"] == pytest.approx(1.0 / 11.0)
    assert out["ci95"][0] > 0.0
    assert out["draw_values_retained"] is False
    assert out["draw_values_sha256"] and len(out["draw_values_sha256"]) == 64
    assert out["mode_role"].startswith("CONFIRMATORY")
    assert "paired_curve_bootstrap_multiseed" in out["delegated_to"]

    same = stats.paired_bootstrap_multiseed(
        candidate_tables=[curves, curves, curves],
        baseline_tables=[curves, curves, curves],
        metric_id="move.rmse_zero", draws=10)
    assert same["n_draws_nonpositive"] == 10
    assert same["p_one_sided"] == pytest.approx(1.0)
    assert same["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_shared_intersection_wrapper_is_marked_audit_only():
    curves = {curve: 0.5 for curve in curves_for(0.5)}
    out = stats.paired_bootstrap(candidate_tables=[curves],
                                 baseline_tables=[curves],
                                 metric_id="move.rmse_zero", draws=5)
    assert out["mode_role"].startswith("AUDIT_ONLY")
    assert "paired_curve_bootstrap " in out["delegated_to"] + " "


def test_h3_confirmatory_path_uses_multiseed_when_seeds_are_disjoint(tmp_path):
    """The decisive switch: no curve is shared across seeds at all.

    The shared-intersection audit cannot run (empty intersection) yet the H3
    verdict must be produced — which is only possible if the confirmatory path
    is the multi-seed one.
    """
    seeds = (17, 29, 43)
    base_curves = {s: curves_like(BASE_METRICS, prefix=f"S{s}_")
                   for s in seeds}
    cand_curves = {s: curves_like(BETTER_METRICS, prefix=f"S{s}_")
                   for s in seeds}
    doc = baseline_doc(curves_by_seed=base_curves)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s,
                      report=eval_report(metrics=BETTER_METRICS,
                                         curves=cand_curves[s]))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles, baseline=doc)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=50)
    candidate = report["candidates"]["M1"]
    endpoint = candidate["endpoints"]["move_rmse"]
    assert endpoint["curve_pairing"]["n_curves_shared_across_all_seeds"] == 0
    assert endpoint["audit_shared_intersection"]["status"] == "unavailable"
    assert endpoint["audit_shared_intersection"]["used_for_verdict"] is False
    assert endpoint["bootstrap_mode"] == "per_seed_independent"
    assert endpoint["n_curves_per_seed"] == [4, 4, 4]
    assert endpoint["n_curves_total"] == 12
    assert endpoint["ci_lower_gt_zero"] is True
    assert endpoint["p_one_sided"] < 0.025
    assert candidate["move_qualified"] is True
    assert candidate["h3_qualified"] is True


def test_h3_reports_both_modes_with_shared_intersection_as_audit(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s,
                      report=eval_report(metrics=BETTER_METRICS,
                                         curves=curves_like(BETTER_METRICS)))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=50)
    endpoint = report["candidates"]["M1"]["endpoints"]["move_rmse"]
    audit = endpoint["audit_shared_intersection"]
    assert audit["audit_only"] is True
    assert audit["used_for_verdict"] is False
    assert audit["status"] != "unavailable"
    assert audit["n_curves_shared"] == 4          # the synthetic intersection
    assert audit["n_curves"] == 4                 # the under-powered reading
    assert audit["mode_role"].startswith("AUDIT_ONLY")
    # the confirmatory reading keeps every seed's own cluster set
    assert endpoint["bootstrap_mode"] == "per_seed_independent"
    assert endpoint["n_curves"] == 4
    assert endpoint["n_curves_per_seed"] == [4, 4, 4]
    assert endpoint["n_curves_total"] == 12
    modes = report["bootstrap_modes"]
    assert modes["confirmatory"]["used_for_verdict"] is True
    assert modes["audit"]["used_for_verdict"] is False
    assert modes["audit"]["n_curves_shared"]["M1:move_rmse"] == 4
    assert report["statistics_contract"]["confirmatory_mode"].startswith(
        "per_seed_independent")
    assert "paired_curve_bootstrap_multiseed" in report[
        "statistics_contract"]["delegated_implementations"][0]
    assert report["bootstrap_audit_shared_intersection"]["M1:move_rmse"][
        "mode_role"].startswith("AUDIT_ONLY")


def test_h3_holm_consumes_the_multiseed_p(tmp_path):
    seeds = (17, 29, 43)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s,
                      report=eval_report(metrics=BETTER_METRICS,
                                         curves=curves_like(BETTER_METRICS)))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    report = stats.compute_h3(table=table, draws=50)
    endpoint = report["candidates"]["M1"]["endpoints"]["move_rmse"]
    family = report["holm_families"]["move_rmse_exit"]
    assert family["p_raw"]["M1"] == pytest.approx(endpoint["p_one_sided"])
    assert family["p_raw"]["M1"] == pytest.approx(
        report["bootstrap_by_endpoint"]["M1:move_rmse"]["p_one_sided"])
    assert family["p_adjusted"]["M1"] == pytest.approx(
        endpoint["p_adjusted_holm"])
    # the audit block never feeds a family
    assert report["bootstrap_audit_shared_intersection"]["M1:move_rmse"][
        "used_for_verdict"] is False


def test_h3_reads_higher_is_better_from_the_producer_and_refuses_contradiction(
        tmp_path):
    seeds = (17, 29, 43)
    doc = baseline_doc()
    doc["bootstrap_inputs"]["metrics"]["move.rmse_zero"][
        "higher_is_better"] = True         # contradicts §9.1 :876
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s,
                      report=eval_report(metrics=BETTER_METRICS,
                                         curves=curves_like(BETTER_METRICS)))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles, baseline=doc)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    assert table["bootstrap_contract"]["metrics_higher_is_better"][
        "move.rmse_zero"] is True
    with pytest.raises(stats.V543StatsError) as exc:
        stats.compute_h3(table=table, draws=10)
    assert exc.value.code == "V543_STATS_BOOTSTRAP_DIRECTION"


def test_h3_uses_the_contract_seed_order(tmp_path):
    seeds = (17, 29, 43)
    order = [43, 29, 17]
    base_curves = {s: curves_like(BASE_METRICS, prefix=f"S{s}_")
                   for s in seeds}
    cand_curves = {s: curves_like(BETTER_METRICS, prefix=f"S{s}_")
                   for s in seeds}
    doc = baseline_doc(curves_by_seed=base_curves, seed_order=order)
    bundles = [bundle(run_id=f"V543_M1_S{s}", candidate_id="M1", track="move",
                      seed=s,
                      report=eval_report(metrics=BETTER_METRICS,
                                         curves=cand_curves[s]))
               for s in seeds]
    root = run_extract_campaign(tmp_path, bundles=bundles, baseline=doc)
    table = stats.load_table(root / stats.METRICS_TABLE_REL)
    assert table["bootstrap_contract"]["seed_order"] == order
    report = stats.compute_h3(table=table, draws=20)
    assert report["statistics_contract"]["seed_order"] == order
    endpoint = report["candidates"]["M1"]["endpoints"]["move_rmse"]
    assert endpoint["curve_pairing"]["seed_order"] == order
    assert list(endpoint["per_seed_bootstrap"]) == ["43", "29", "17"]
    # the wiring must hand the tables to the frozen function in that order
    candidate_rows = stats.rows_by_seed(table, "M1")
    cand_tables = [candidate_rows[s]["curve_values"]["move.rmse_zero"]
                   for s in order]
    base_tables = [stats.baseline_row(table, s)["curve_values"][
        "move.rmse_zero"] for s in order]
    direct = stats.paired_bootstrap_multiseed(
        candidate_tables=cand_tables, baseline_tables=base_tables,
        metric_id="move.rmse_zero", higher_is_better=False,
        draws=20, seed_root=stats.BOOTSTRAP_SEED_ROOT)
    assert endpoint["p_one_sided"] == pytest.approx(direct["p_one_sided"])
    assert endpoint["ci95"] == pytest.approx(direct["ci95"])


def test_table_refuses_to_mix_instrument_generations(tmp_path):
    root = run_extract_campaign(tmp_path)
    doc = baseline_doc()
    doc["evaluator"]["code_sha256"] = "code-0002"
    (root / "audit/baseline.json").write_text(json.dumps(doc),
                                             encoding="utf-8")
    with pytest.raises(stats.V543StatsError) as exc:
        stats.run_extract(cfg_stub(), campaign_root=root,
                          baseline_path=root / "audit/baseline.json")
    assert exc.value.code == "V543_STATS_TABLE_IDENTITY_MIX"


# ---------------------------------------------------------------------------
# 7. config vs 附录 B, and the plan-ambiguity register
# ---------------------------------------------------------------------------
def test_config_consistency_matches_appendix_b():
    pytest.importorskip("yaml")
    cfg = stats._load_cfg(stats.DEFAULT_CONFIG)
    report = stats.config_consistency(cfg)
    assert report["mismatched"] == []
    assert report["n_checked"] == 21


def test_config_consistency_reports_a_disagreement_without_patching_it():
    cfg = {"thresholds": {"move_rmse_regression_rel_max_pct": 3.0},
           "evaluation": {"bootstrap_draws": 10000,
                          "bootstrap_seed_root": 20261102}}
    report = stats.config_consistency(cfg)
    assert "move_rmse_regression_rel_max_pct" in report["mismatched"]
    check = report["checks"]["move_rmse_regression_rel_max_pct"]
    assert check["plan_value"] == 2.0
    assert check["config_value"] == 3.0
    assert check["matches"] is False


def test_ambiguities_are_declared_as_data():
    ids = {entry["id"] for entry in stats.ambiguities()}
    assert {"h0_item_count", "zero_baseline_epsilon",
            "screen_two_seed_aggregation", "cross_seed_cluster_intersection",
            "endpoint_list_for_H3_holm"} <= ids


def test_frozen_constants_match_appendix_b():
    assert stats.MOVE_RMSE_REGRESSION_REL_MAX == pytest.approx(0.02)
    assert stats.PROCESS_Q_MAE_REGRESSION_REL_MAX == pytest.approx(0.05)
    assert stats.PROCESS_H3_AUPRC_ABS_MIN == pytest.approx(0.02)
    assert stats.MOVE_EXIT_ALPHA_EACH == pytest.approx(0.025)
    assert stats.PROCESS_FAMILY_ALPHA == pytest.approx(0.025)
    assert stats.MAX_CORE_TIME_RATIO == pytest.approx(2.0)
    assert stats.MAX_LATENCY_RATIO_P95 == pytest.approx(2.0)
    assert stats.BOOTSTRAP_DRAWS == 10000
    assert stats.BOOTSTRAP_SEED_ROOT == 20261102
    assert stats.FROZEN_SEEDS == (17, 29, 43)
    assert stats.SCREEN_SEEDS == (17, 29)


def test_cli_refuses_a_missing_table(tmp_path):
    code = stats.main(["h2", "--campaign-root", str(tmp_path)])
    assert code == 2
