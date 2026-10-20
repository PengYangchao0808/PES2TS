"""R1 analysis/consolidation tooling gates (V5_1A_R1_REPAIR, CPU only).

Synthetic-cell coverage of the PURE logic in
``scripts/campaigns/r1_dev_analysis.py`` and
``scripts/campaigns/r1_consolidate.py`` — no checkpoints, no real run
dirs, no network:

* cell discovery: A0 legacy-mapping resolution (7 legacy cells first,
  fallback r1 root), baseline-fill/dev arms resolve to the r1 root,
  absent cells stay absent; availability predicate needs run_record +
  metrics + checkpoint_best;
* aggregation: per (arm, fold) medians over whatever seeds landed
  (missing seeds reported, never fabricated), A2/A3-vs-A0 fold deltas;
* seed pooling + cluster bootstrap: per-cluster medians over the COMMON
  seeds only, seeded reproducibility of the 2000-draw paired bootstrap
  (frozen seed 20261018) and its point delta / CI ordering;
* futility (protocol §futility_rules): SRE regression > 0.03 on >= 2/3
  folds OR RMSE/0 regression > 5% on >= 2/3 folds ⇒ arm_stop; 1/3 folds
  never fires; insufficient data never fires;
* A4 gate: a2/a3_positive criteria (SRE improves >= 0.01 on >= 2/3 dev
  folds with RMSE within 1%) and complementarity tolerances (A2
  dead-spurious +0.02 absolute, A3 strong-amplitude −0.1);
* smoke checker tolerance logic;
* consolidation: stage/source classification for legacy and R1 runs,
  wall-time extraction from both decision logs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.campaigns import r1_consolidate as con  # noqa: E402
from scripts.campaigns import r1_dev_analysis as ana  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic cell helpers
# ---------------------------------------------------------------------------

def _cell(arm: str, fold: int, seed: int, *, scalars: dict,
          per_cluster: dict | None = None,
          reported: dict | None = None,
          canonical: dict | None = None) -> dict:
    cell = {
        "arm": arm, "fold": fold, "seed": seed,
        "run_id": f"{'B0' if arm == 'A0' else arm}_f{fold}_s{seed}_v51",
        "run_dir": f"outputs/pesn/v5_1/runs/{arm}_{fold}_{seed}",
        "source": "legacy" if arm == "A0" else "r1",
        "scalars": dict(scalars),
    }
    if per_cluster is not None:
        cell["per_cluster"] = per_cluster
    if reported is not None:
        cell["reported"] = reported
    if canonical is not None:
        cell["canonical"] = canonical
    return cell


def _s(rmse: float = 0.7, sre: float = 0.3, **extra) -> dict:
    base = {
        "reported_rmse_over_zero": rmse,
        "reported_sre_median": sre,
        "canonical_rmse_over_zero": rmse,
        "canonical_sre_cluster_median": sre,
        "canonical_near_ts_cos": 0.85,
        "dead_spurious_ge_0.005": 0.10,
        "strong_amplitude_ratio_median": 0.90,
    }
    base.update(extra)
    return base


def _agg_entry(scalars: dict) -> dict:
    return {"seeds": [17, 29, 43], "n_seeds": 3, "missing_seeds": [],
            "scalars": dict(scalars)}


def _aggregates(fold_values: dict) -> dict:
    """fold_values: {arm: {fold: scalar-dict}} → aggregates shape."""
    return {
        arm: {str(fold): _agg_entry(scalars)
              for fold, scalars in folds.items()}
        for arm, folds in fold_values.items()
    }


# ---------------------------------------------------------------------------
# cell discovery
# ---------------------------------------------------------------------------

def _mk_run(root: Path, run_id: str, *, record: bool = True,
            metrics: bool = False, ckpt: bool = False) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if record:
        (run_dir / "run_record.json").write_text(
            '{"run_id": "%s", "status": "complete"}' % run_id,
            encoding="utf-8")
    if metrics:
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    if ckpt:
        ck = run_dir / "checkpoints"
        ck.mkdir(exist_ok=True)
        (ck / "checkpoint_best.pt").write_bytes(b"pt")
    return run_dir


def test_resolve_cell_a0_legacy_mapping(tmp_path):
    legacy = tmp_path / "legacy"
    r1 = tmp_path / "r1"
    # legacy cell exists only in the legacy root
    _mk_run(legacy, "B0_f0_s17_v51")
    got = ana.resolve_cell("A0", 0, 17, legacy_root=legacy, r1_root=r1)
    assert got is not None
    assert got[0] == legacy / "B0_f0_s17_v51" and got[1] == "legacy"
    # non-legacy A0 cell resolves to the r1 root
    _mk_run(r1, "B0_f1_s29_v51")
    got = ana.resolve_cell("A0", 1, 29, legacy_root=legacy, r1_root=r1)
    assert got is not None
    assert got[0] == r1 / "B0_f1_s29_v51" and got[1] == "r1"
    # legacy cell vanished → fallback to r1 root (driver behaviour)
    _mk_run(r1, "B0_f3_s17_v51")
    got = ana.resolve_cell("A0", 3, 17, legacy_root=legacy, r1_root=r1)
    assert got is not None and got[0] == r1 / "B0_f3_s17_v51"
    # absent everywhere → None (never fabricated)
    assert ana.resolve_cell(
        "A0", 4, 43, legacy_root=legacy, r1_root=r1) is None


def test_resolve_cell_dev_arms_and_absence(tmp_path):
    r1 = tmp_path / "r1"
    _mk_run(r1, "A2_f0_s17_v51")
    got = ana.resolve_cell("A2", 0, 17,
                           legacy_root=tmp_path / "legacy", r1_root=r1)
    assert got is not None and got[1] == "r1"
    # dev arms never resolve out of the r1 root
    assert ana.resolve_cell(
        "A3", 2, 43, legacy_root=tmp_path, r1_root=r1) is None


def test_cell_availability_requires_all_three(tmp_path):
    run_dir = _mk_run(tmp_path, "X", record=True)
    ok, reason = ana.cell_availability(run_dir)
    assert not ok and "checkpoint_best" in reason
    _mk_run(tmp_path, "X", ckpt=True)          # record + ckpt, no metrics
    ok, reason = ana.cell_availability(run_dir)
    assert not ok and "metrics" in reason
    _mk_run(tmp_path, "X", metrics=True)       # all three present
    ok, reason = ana.cell_availability(run_dir)
    assert ok and reason == ""
    # an in-flight run reports its status as the reason
    run_dir2 = _mk_run(tmp_path, "Y", record=True, ckpt=True)
    rec = json.loads((run_dir2 / "run_record.json").read_text(
        encoding="utf-8"))
    rec["status"] = "running"
    (run_dir2 / "run_record.json").write_text(json.dumps(rec),
                                              encoding="utf-8")
    ok, reason = ana.cell_availability(run_dir2)
    assert not ok and "running" in reason


# ---------------------------------------------------------------------------
# aggregation + deltas
# ---------------------------------------------------------------------------

def test_aggregate_cells_medians_and_missing_seed():
    cells = [
        _cell("A0", 0, 17, scalars=_s(rmse=0.70, sre=0.30)),
        _cell("A0", 0, 29, scalars=_s(rmse=0.72, sre=0.32)),
        # seed 43 missing → reported, not padded
        _cell("A2", 0, 17, scalars=_s(rmse=0.68, sre=0.26)),
        _cell("A2", 0, 29, scalars=_s(rmse=0.74, sre=0.24)),
        _cell("A2", 0, 43, scalars=_s(rmse=0.70, sre=0.22)),
    ]
    agg = ana.aggregate_cells(cells, folds=[0], seeds=[17, 29, 43])
    a0 = agg["A0"]["0"]
    assert a0["n_seeds"] == 2 and a0["missing_seeds"] == [43]
    assert a0["scalars"]["canonical_rmse_over_zero"] == pytest.approx(0.71)
    a2 = agg["A2"]["0"]
    assert a2["n_seeds"] == 3
    assert a2["scalars"]["canonical_sre_cluster_median"] == \
        pytest.approx(0.24)
    # cells outside requested folds never enter the fold aggregates
    stray = cells + [_cell("A0", 1, 17, scalars=_s(rmse=9.9))]
    agg = ana.aggregate_cells(stray, folds=[0], seeds=[17, 29, 43])
    assert "1" not in agg["A0"]


def test_fold_deltas_candidate_minus_baseline():
    agg = _aggregates({
        "A0": {0: _s(rmse=0.70, sre=0.30), 1: _s(rmse=0.80, sre=0.40)},
        "A2": {0: _s(rmse=0.75, sre=0.26), 1: _s(rmse=0.78, sre=0.35)},
        "A3": {0: _s(rmse=0.69, sre=0.31)},
    })
    deltas = ana.fold_deltas(agg)
    assert deltas["A2"]["0"]["delta_candidate_minus_baseline"][
        "canonical_rmse_over_zero"] == pytest.approx(0.05)
    assert deltas["A2"]["0"]["delta_candidate_minus_baseline"][
        "canonical_sre_cluster_median"] == pytest.approx(-0.04)
    assert deltas["A2"]["1"]["delta_candidate_minus_baseline"][
        "canonical_sre_cluster_median"] == pytest.approx(-0.05)
    assert deltas["A3"]["0"]["delta_candidate_minus_baseline"][
        "canonical_rmse_over_zero"] == pytest.approx(-0.01)
    # a fold missing the baseline is skipped, not fabricated
    agg = _aggregates({"A2": {2: _s()}})
    assert ana.fold_deltas(agg) == {}


# ---------------------------------------------------------------------------
# seed pooling + paired cluster bootstrap
# ---------------------------------------------------------------------------

def _pc(values: dict) -> dict:
    return {cid: {"rmse_over_zero": v[0], "sre_cluster_median": v[1]}
            for cid, v in values.items()}


def test_pool_cluster_values_paired_common_seeds():
    cand = [
        _cell("A2", 0, 17, scalars=_s(),
              per_cluster=_pc({"C1": (0.6, 0.2), "C2": (0.5, 0.4)})),
        _cell("A2", 0, 29, scalars=_s(),
              per_cluster=_pc({"C1": (0.7, 0.3), "C2": (0.6, 0.5)})),
        _cell("A2", 0, 43, scalars=_s(),
              per_cluster=_pc({"C1": (0.8, 0.4), "C2": (0.7, 0.6)})),
    ]
    base = [
        _cell("A0", 0, 17, scalars=_s(),
              per_cluster=_pc({"C1": (0.9, 0.5), "C2": (0.8, 0.7)})),
        _cell("A0", 0, 29, scalars=_s(),
              per_cluster=_pc({"C1": (0.9, 0.5), "C2": (0.8, 0.7)})),
        # baseline seed 43 absent → excluded from BOTH pools (paired)
    ]
    cand_pool, base_pool, common = ana.pool_cluster_values(
        cand, base, metric_key="sre_cluster_median")
    assert common == [17, 29]
    assert cand_pool == {"C1": pytest.approx(0.25),
                         "C2": pytest.approx(0.45)}
    assert base_pool == {"C1": pytest.approx(0.5),
                         "C2": pytest.approx(0.7)}
    # a cluster present on one side only never leaks through
    cand_extra = cand[:2] + [
        _cell("A2", 0, 43, scalars=_s(),
              per_cluster=_pc({"C1": (0.8, 0.4), "C2": (0.7, 0.6),
                               "C3": (1.0, 1.0)})),
    ]
    cand_pool, base_pool, common = ana.pool_cluster_values(
        cand_extra, base, metric_key="rmse_over_zero")
    assert common == [17, 29]          # seed 43 missing on baseline
    assert set(cand_pool) == set(base_pool) == {"C1", "C2"}
    # cluster only ever seen on the candidate side stays excluded
    cand_only = cand[:2] + [
        _cell("A2", 0, 43, scalars=_s(),
              per_cluster=_pc({"C3": (1.0, 1.0)})),
    ]
    cand_pool, base_pool, common = ana.pool_cluster_values(
        cand_only, base, metric_key="rmse_over_zero")
    assert common == [17, 29]
    assert set(cand_pool) == set(base_pool) == {"C1", "C2"}


def test_bootstrap_pooled_deterministic_and_point_delta():
    rng_a = {"C1": 0.20, "C2": 0.22, "C3": 0.24, "C4": 0.26,
             "C5": 0.28, "C6": 0.30, "C7": 0.32, "C8": 0.34}
    rng_b = {c: v + 0.05 for c, v in rng_a.items()}
    b1 = ana.bootstrap_pooled(rng_a, rng_b, draws=2000, seed=20261018)
    b2 = ana.bootstrap_pooled(rng_a, rng_b, draws=2000, seed=20261018)
    assert b1["n_clusters"] == 8
    assert b1["draws"] == 2000 and b1["seed"] == 20261018
    # constant delta → every draw reproduces it exactly
    assert b1["point_delta"] == pytest.approx(-0.05)
    assert b1["median_delta"] == pytest.approx(-0.05)
    assert b1["ci_lo_2_5"] == pytest.approx(-0.05, abs=1e-12)
    assert b1["ci_hi_97_5"] == pytest.approx(-0.05, abs=1e-12)
    # seeded reproducibility
    assert b1 == b2
    assert b1["ci_lo_2_5"] <= b1["median_delta"] <= b1["ci_hi_97_5"]
    assert set(b1["per_cluster_deltas"]) == set(rng_a)
    assert b1["per_cluster_deltas"]["C1"] == pytest.approx(-0.05)
    # heterogeneous deltas → CI is an interval around the point estimate
    rng_c = {"C1": 0.10, "C2": 0.20, "C3": 0.30, "C4": 0.40,
             "C5": 0.50, "C6": 0.60, "C7": 0.70, "C8": 0.80}
    b3 = ana.bootstrap_pooled(rng_a, rng_c, draws=2000, seed=20261018)
    assert b3["ci_lo_2_5"] < b3["point_delta"] < b3["ci_hi_97_5"]
    # empty input → status, not an exception
    assert ana.bootstrap_pooled({}, {})["n_clusters"] == 0


# ---------------------------------------------------------------------------
# futility (protocol §futility_rules)
# ---------------------------------------------------------------------------

def _futility_agg(fold_sre_delta: dict, fold_rmse_rel: dict) -> dict:
    """fold_{...}: {fold: delta}; build A0/candidate aggregates."""
    folds = sorted(set(fold_sre_delta) | set(fold_rmse_rel))
    a0, cand = {}, {}
    for f in folds:
        a0[f] = {"canonical_sre_cluster_median": 0.30,
                 "canonical_rmse_over_zero": 0.70}
        cand[f] = {
            "canonical_sre_cluster_median":
                0.30 + fold_sre_delta.get(f, 0.0),
            "canonical_rmse_over_zero":
                0.70 * (1.0 + fold_rmse_rel.get(f, 0.0)),
        }
    return _aggregates({"A0": a0, "A2": cand})


def test_futility_sre_regression_two_of_three_fires():
    agg = _futility_agg({0: 0.04, 1: 0.04, 2: 0.01}, {})
    res = ana.futility_check(agg, folds=[0, 1, 2])
    assert res["A2"]["sre_regression_folds"] == [0, 1]
    assert res["A2"]["n_sre_regressions"] == 2
    assert res["A2"]["arm_stop"] is True
    assert res["A2"]["insufficient_data"] is False


def test_futility_single_fold_regression_never_fires():
    agg = _futility_agg({0: 0.04}, {})
    res = ana.futility_check(agg, folds=[0, 1, 2])
    assert res["A2"]["n_sre_regressions"] == 1
    assert res["A2"]["arm_stop"] is False
    assert res["A2"]["insufficient_data"] is True  # only fold 0 evaluable


def test_futility_rmse_regression_rule_and_boundary():
    # > 5% on two folds → stop
    agg = _futility_agg({}, {0: 0.06, 1: 0.06})
    res = ana.futility_check(agg, folds=[0, 1, 2])
    assert res["A2"]["rmse_regression_folds"] == [0, 1]
    assert res["A2"]["arm_stop"] is True
    # exactly 5% is NOT > 5% → no fire
    agg = _futility_agg({}, {0: 0.05, 1: 0.05})
    res = ana.futility_check(agg, folds=[0, 1, 2])
    assert res["A2"]["n_rmse_regressions"] == 0
    assert res["A2"]["arm_stop"] is False
    # 5% on two folds + big SRE regression on one fold → SRE rule needs 2
    agg = _futility_agg({0: 0.04}, {1: 0.06})
    res = ana.futility_check(agg, folds=[0, 1, 2])
    assert res["A2"]["n_sre_regressions"] == 1
    assert res["A2"]["n_rmse_regressions"] == 1
    assert res["A2"]["arm_stop"] is False


def test_futility_healthy_arm_not_flagged():
    agg = _futility_agg({0: -0.01, 1: 0.005, 2: 0.02}, {0: -0.02})
    res = ana.futility_check(agg, folds=[0, 1, 2])
    assert res["A2"]["arm_stop"] is False
    assert res["A2"]["n_sre_regressions"] == 0


# ---------------------------------------------------------------------------
# A4 gate
# ---------------------------------------------------------------------------

def test_a4_gate_positive_and_complementary():
    # A2: SRE improves 0.02/0.015 on folds 0/1, RMSE +0.5% → positive
    # A3: SRE improves 0.03/0.02 on folds 0/1 → positive
    # complementarity within tolerance for both
    agg = _aggregates({
        "A0": {0: _s(sre=0.30, rmse=0.70, **{"dead_spurious_ge_0.005": 0.10,
                                              "strong_amplitude_ratio_median":
                                                  0.90}),
               1: _s(sre=0.30, rmse=0.70, **{"dead_spurious_ge_0.005": 0.10,
                                              "strong_amplitude_ratio_median":
                                                  0.90}),
               2: _s(sre=0.30, rmse=0.70)},
        "A2": {0: _s(sre=0.28, rmse=0.7035,
                     **{"dead_spurious_ge_0.005": 0.11}),
               1: _s(sre=0.285, rmse=0.698,
                     **{"dead_spurious_ge_0.005": 0.12}),
               2: _s(sre=0.29, rmse=0.701)},
        "A3": {0: _s(sre=0.27, rmse=0.704,
                     **{"strong_amplitude_ratio_median": 0.85}),
               1: _s(sre=0.28, rmse=0.702,
                     **{"strong_amplitude_ratio_median": 0.87}),
               2: _s(sre=0.30, rmse=0.705)},
    })
    gate = ana.a4_gate(agg, folds=[0, 1, 2])
    assert gate["a2_positive"] is True
    assert gate["a3_positive"] is True
    assert gate["complementarity"]["a2_dead_spurious"]["ok"] is True
    assert gate["complementarity"]["a3_strong_amplitude_ratio"]["ok"] \
        is True
    assert gate["a4_gate_open"] is True


def test_a4_gate_rmse_outside_one_percent_disqualifies():
    # SRE improves on 2 folds but RMSE +2% on both → not positive
    agg = _aggregates({
        "A0": {0: _s(sre=0.30, rmse=0.70), 1: _s(sre=0.30, rmse=0.70)},
        "A2": {0: _s(sre=0.25, rmse=0.714), 1: _s(sre=0.26, rmse=0.716)},
    })
    gate = ana.a4_gate(agg, folds=[0, 1, 2])
    assert gate["a2_positive"] is False
    a2 = gate["per_arm"]["A2"]
    assert a2["folds_evaluated"] == [0, 1] and a2["folds_qualifying"] == []
    # improvement counted but RMSE gate failed
    assert a2["sre_improvement_A0_minus_arm"]["0"] == pytest.approx(0.05)


def test_a4_gate_needs_two_qualifying_folds():
    agg = _aggregates({
        "A0": {0: _s(sre=0.30), 1: _s(sre=0.30), 2: _s(sre=0.30)},
        "A2": {0: _s(sre=0.27), 1: _s(sre=0.31), 2: _s(sre=0.31)},
    })
    gate = ana.a4_gate(agg, folds=[0, 1, 2])
    assert gate["per_arm"]["A2"]["folds_qualifying"] == [0]
    assert gate["a2_positive"] is False


def test_a4_gate_complementarity_vetoes():
    # A2 positive but dead-spurious degrades by +0.05 > 0.02 → not
    # complementary → gate closed
    agg = _aggregates({
        "A0": {0: _s(sre=0.30, **{"dead_spurious_ge_0.005": 0.10}),
               1: _s(sre=0.30, **{"dead_spurious_ge_0.005": 0.10})},
        "A2": {0: _s(sre=0.25, **{"dead_spurious_ge_0.005": 0.13}),
               1: _s(sre=0.25, **{"dead_spurious_ge_0.005": 0.15})},
        "A3": {0: _s(sre=0.25, **{"strong_amplitude_ratio_median": 0.89}),
               1: _s(sre=0.25, **{"strong_amplitude_ratio_median": 0.88})},
    })
    gate = ana.a4_gate(agg, folds=[0, 1, 2])
    assert gate["a2_positive"] is True
    assert gate["a3_positive"] is True
    comp = gate["complementarity"]
    assert comp["a2_dead_spurious"]["ok"] is False
    assert comp["a2_dead_spurious"]["worst_fold_delta"] == \
        pytest.approx(0.05)
    assert comp["a3_strong_amplitude_ratio"]["ok"] is True
    assert gate["a4_gate_open"] is False
    assert any("A2 not complementary" in n for n in comp["notes"])

    # A3 strong amplitude drop beyond 0.1 → veto
    agg["A3"]["1"]["scalars"]["strong_amplitude_ratio_median"] = 0.75
    gate = ana.a4_gate(agg, folds=[0, 1, 2])
    assert gate["complementarity"]["a3_strong_amplitude_ratio"]["ok"] \
        is False
    assert gate["a4_gate_open"] is False


def test_a4_gate_insufficient_data_is_false_not_fabricated():
    agg = _aggregates({"A0": {0: _s(sre=0.30)},
                       "A2": {0: _s(sre=0.20)}})
    gate = ana.a4_gate(agg, folds=[0, 1, 2])
    assert gate["a2_positive"] is False
    assert gate["per_arm"]["A2"]["folds_evaluated"] == [0]
    assert "only 1/3" in gate["per_arm"]["A2"]["note"]
    assert gate["a3_positive"] is False
    assert gate["a4_gate_open"] is False
    assert any("no A3-vs-A0" in n
               for n in gate["complementarity"]["notes"])


# ---------------------------------------------------------------------------
# smoke checker
# ---------------------------------------------------------------------------

def _smoke_cell(rmse_diff: float, sre_diff: float) -> dict:
    rep_r, can_r = 0.774, 0.774 + rmse_diff
    rep_s, can_s = 0.293, 0.293 + sre_diff
    return {
        "run_id": "B0_f0_s17_v51", "fold": 0, "seed": 17,
        "reported": {"inner_val_score": {
            "rmse_over_zero": rep_r, "sre_median": rep_s}},
        "canonical": {"rmse_over_zero": can_r,
                      "sre_cluster_median": can_s},
        "reported_vs_canonical_abs_diff": {
            "rmse_over_zero": abs(rmse_diff),
            "sre_cluster_median": abs(sre_diff)},
    }


def test_smoke_check_tolerance():
    ok = ana.smoke_check([_smoke_cell(1e-6, 3e-9)], tol=2e-3)
    assert ok["all_ok"] is True and ok["cells"][0]["ok"] is True
    bad = ana.smoke_check([_smoke_cell(1e-6, 3e-9),
                           _smoke_cell(0.0, 5e-3)], tol=2e-3)
    assert bad["all_ok"] is False
    assert bad["cells"][0]["ok"] is True
    assert bad["cells"][1]["ok"] is False


def test_smoke_check_missing_diff_is_failure():
    cell = _smoke_cell(0.0, 0.0)
    cell["reported_vs_canonical_abs_diff"] = {
        "rmse_over_zero": None, "sre_cluster_median": None}
    assert ana.smoke_check([cell])["all_ok"] is False


# ---------------------------------------------------------------------------
# consolidation: classification + decision logs
# ---------------------------------------------------------------------------

def test_classify_legacy_and_r1():
    assert con.classify_legacy(0)[0] == "dev_reference"
    assert con.classify_legacy(2)[0] == "dev_reference"
    assert con.classify_legacy(3)[0] == "confirm_reference"
    assert con.classify_legacy(4)[0] == "confirm_reference"
    # R1: decision-log stage wins
    assert con.classify_r1("A2", 0, "dev") == ("dev", "r1_dev")
    assert con.classify_r1("A4", 3, "confirm") == ("confirm",
                                                    "r1_confirm")
    # fallbacks when the log has no row
    assert con.classify_r1("B0", 1, None) == ("baseline_fill",
                                              "r1_baseline_fill")
    assert con.classify_r1("A3", 2, None) == ("dev", "r1_dev")
    assert con.classify_r1("A3", 4, None) == ("confirm", "r1_confirm")


def test_wall_time_extraction():
    legacy_rows = [
        {"decision": "subprocess", "wall_s": 12.5,
         "argv": ["python", "-m", "trainer", "--out",
                  "outputs/pesn/v5_1/runs/B0_f0_s17_v51"]},
        {"decision": "prepare_complete", "wall_s": 99.0},
        {"decision": "subprocess", "wall_s": 7.0,
         "argv": ["python", "-m", "trainer"]},  # no --out → ignored
    ]
    assert con.legacy_wall_times(legacy_rows) == {
        "B0_f0_s17_v51": 12.5}
    r1_rows = [
        {"decision": "launch", "run_id": "A2_f0_s17_v51",
         "stage": "dev", "wall_s": 33.0},
        {"decision": "stage_status", "stage": "dev"},
        {"decision": "reuse_a0", "run_id": "B0_f0_s17_v51",
         "stage": "baseline_fill"},  # no wall_s
        {"decision": "launch", "run_id": "B0_f1_s29_v51",
         "stage": "baseline_fill", "wall_s": 44.5},
    ]
    meta = con.r1_stage_and_wall(r1_rows)
    assert meta["A2_f0_s17_v51"] == {"stage": "dev", "wall_s": 33.0}
    assert meta["B0_f0_s17_v51"] == {"stage": "baseline_fill"}
    assert meta["B0_f1_s29_v51"]["wall_s"] == 44.5


def test_build_row_empty_fields_never_fabricated(tmp_path):
    run_dir = tmp_path / "B0_f0_s17_v51"
    run_dir.mkdir()
    record = {
        "run_id": "B0_f0_s17_v51", "fold": 0, "seed": 17,
        "updates_budget": 3000, "arm": {"arm_id": "B0"},
        "inner_val_score": {"rmse_over_zero": 0.77, "sre_median": 0.29,
                            "near_ts_cos": 0.85},
        "sign_accuracy": 0.75, "alpha_median": 0.55,
        "dead_zone_fraction": 0.15, "best_update": 550,
        "gv": {"config_sha256": "abc", "data_manifest_sha256": "def"},
        "loss_schema_version": "pesn_key_atom_bc_loss_v2",
    }
    row = con.build_row(
        run_record=record, run_dir=run_dir, source=con.SOURCE_LEGACY,
        stage="dev_reference", campaign="V5_1A_SPRINT", wall_s=12.5)
    assert row["inner_rmse_over_zero"] == 0.77
    assert row["reused_from"].endswith("B0_f0_s17_v51")
    assert row["config_sha256"] == "abc"
    assert row["wall_s"] == 12.5
    # a running run without metrics → None stays None, status preserved
    running = {"run_id": "A2_f0_s29_v51", "fold": 0, "seed": 29,
               "updates_budget": 3000, "arm": {"arm_id": "A2"},
               "status": "running"}
    row = con.build_row(
        run_record=running, run_dir=run_dir, source=con.SOURCE_DEV,
        stage="dev", campaign="V5_1A_R1_REPAIR", wall_s=None)
    assert row["inner_rmse_over_zero"] is None
    assert row["best_update"] is None
    assert row["status"] == "running"
    assert row["reused_from"] == ""  # not a legacy reuse row


def test_collect_rows_matrix_and_manifest(tmp_path):
    legacy = tmp_path / "legacy"
    r1 = tmp_path / "r1"
    rec0 = {"run_id": "B0_f0_s17_v51", "fold": 0, "seed": 17,
            "updates_budget": 3000, "arm": {"arm_id": "B0"},
            "campaign_id": "V5_1A_SPRINT",
            "inner_val_score": {"rmse_over_zero": 0.7,
                                "sre_median": 0.3, "near_ts_cos": 0.8},
            "gv": {"config_sha256": "c0", "data_manifest_sha256": "d0"}}
    rec3 = {"run_id": "B3_f4_s17_v51", "fold": 4, "seed": 17,
            "updates_budget": 3000, "arm": {"arm_id": "B3"},
            "campaign_id": "V5_1A_SPRINT",
            "gv": {"config_sha256": "c1", "data_manifest_sha256": "d1"}}
    rec_a2 = {"run_id": "A2_f0_s17_v51", "fold": 0, "seed": 17,
              "updates_budget": 3000, "arm": {"arm_id": "A2"},
              "campaign_id": "V5_1A_R1_REPAIR",
              "loss_schema_version": "pesn_key_atom_bc_loss_v3",
              "gv": {"config_sha256": "c2", "data_manifest_sha256": "d2"}}
    for root, doc in ((legacy, rec0), (legacy, rec3), (r1, rec_a2)):
        run_dir = root / doc["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_record.json").write_text(json_dumps(doc),
                                                 encoding="utf-8")
    # r1 run WITHOUT metrics/checkpoint still lands as a row
    rows, entries = con.collect_rows(
        legacy_root=legacy, r1_root=r1,
        legacy_decisions=tmp_path / "nope.jsonl",
        r1_decisions=tmp_path / "nope2.jsonl")
    assert [r["run_id"] for r in rows] == [
        "B0_f0_s17_v51", "B3_f4_s17_v51", "A2_f0_s17_v51"]
    assert rows[0]["stage"] == "dev_reference"
    assert rows[0]["source"] == con.SOURCE_LEGACY
    assert rows[2]["source"] == con.SOURCE_DEV and rows[2]["stage"] == "dev"
    assert rows[1]["stage"] == "confirm_reference"
    assert rows[0]["notes"].startswith("v5_1a A0 reuse cell")
    assert "reference arm" in rows[1]["notes"]
    assert rows[2]["notes"].endswith("inner metrics pending")
    assert len(entries) == 3
    assert entries[0]["files"]["run_record.json"]["sha256"]
    assert set(entries[0]["files"]) == {"run_record.json"}


def json_dumps(doc: dict) -> str:
    import json
    return json.dumps(doc)


def test_matrix_writer_field_order(tmp_path):
    out = tmp_path / "experiment_matrix.csv"
    con.write_matrix([], out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == con.MATRIX_FIELDS
