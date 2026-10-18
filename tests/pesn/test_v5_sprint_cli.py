"""Lane D acceptance: V5 sprint gate protocols + campaign CLI (plan §5/§6).

Covers hermetically (no GPU, no real data, no training):

* stage routing + unknown-stage exit 2; dry-run writes NOTHING;
* downgrade-ladder arithmetic and the §5.4 t_run timing formula;
* §5.3 screen ranking + tie-break chain on synthetic metrics;
* the EXACT §5.7 status vocabulary;
* the censored-budget path under an injected fake clock;
* the sealed-eval typed REFUSED echo (this sprint never unseals);
* G5.4 synthetic static checks (clash incl. bonded pairs, protected
  stretch/shorten, strict-copy trip, oracle ceiling pass, follower
  counterfactual marking);
* G5.3 synthetic sign accuracy + SRE.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (  # noqa: E402
    KeyAtomProtocolError,
    assert_strict_copy,
    clash_pairs_of,
    evaluate_g52,
    evaluate_g53,
    evaluate_g54,
    exact_delta_d,
    first_order_pair_projection,
    follower_counterfactual_diagnostics,
    follower_h_rows,
    oracle_static_ceiling,
    patch_strict_copy,
    protected_edges_of,
    protected_violations_of,
    sealed_eval_request,
    write_gate_report,
)

# ---------------------------------------------------------------------------
# Synthetic G5.4 geometries
# ---------------------------------------------------------------------------


def _h2_sample(distance: float = 0.74) -> dict:
    """H2 with one H key-atom: bonded pair + terminal-H follower setup."""
    return {
        "group_id": "RXN_G54_H2",
        "pos": np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        "pos_start": np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        "atomic_numbers": [1, 1],
        "key_rows": [1],
        "k_pairs": [(0, 1)],
        "start_bonds": [(0, 1, 1.0)],
    }


def _water_like_sample() -> dict:
    """O(0) key atom with two O-H protected bonds (stretch/shorten tests)."""
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
    ])
    return {
        "group_id": "RXN_G54_WATER",
        "pos": pos,
        "pos_start": pos.copy(),
        "atomic_numbers": [8, 1, 1],
        "key_rows": [0],
        "k_pairs": [],
        "start_bonds": [(0, 1, 1.0), (0, 2, 1.0)],
    }


# ---------------------------------------------------------------------------
# 1. Stage routing + unknown stage exit 2
# ---------------------------------------------------------------------------


def test_unknown_stage_exits_2() -> None:
    result = subprocess.run(
        [sys.executable,
         str(PROJECT_ROOT / "scripts/campaigns/run_v5_sprint.py"),
         "--config", str(PROJECT_ROOT / "config/campaigns/v5_sprint.yaml"),
         "--stage", "bogus"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120)
    assert result.returncode == 2, result.stderr
    assert "invalid choice" in result.stderr


def test_stage_routing_table_is_complete() -> None:
    import scripts.campaigns.run_v5_sprint as drv

    assert tuple(drv.STAGE_ORDER) == (
        "prepare-audit", "label-screen", "recipe-recheck", "crossfit",
        "report")
    # every stage maps to a handler method
    for stage in drv.STAGE_ORDER:
        handler = drv.SprintDriver.STAGE_HANDLERS[stage]
        assert hasattr(drv.SprintDriver, handler), (stage, handler)


# ---------------------------------------------------------------------------
# 2. Dry-run writes nothing (tmp cwd)
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = str(PROJECT_ROOT / "config/campaigns/v5_sprint.yaml")
    for stage in ("prepare-audit", "label-screen", "crossfit", "report"):
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts/campaigns/run_v5_sprint.py"),
             "--config", config, "--stage", stage, "--dry-run"],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=120)
        assert result.returncode == 0, (stage, result.stderr)
        assert "[dry-run]" in result.stdout
    # nothing was created in the fresh cwd
    assert list(tmp_path.iterdir()) == []


def test_dry_run_driver_creates_no_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    import scripts.campaigns.run_v5_sprint as drv

    driver = drv.SprintDriver(
        config_path=PROJECT_ROOT / "config/campaigns/v5_sprint.yaml",
        out_root=tmp_path / "out", dry_run=True)
    assert driver.run_stage("report") == "dry_run"
    assert not (tmp_path / "out").exists()
    assert not driver.manifest_path.exists()


# ---------------------------------------------------------------------------
# 3. Downgrade ladder arithmetic + t_run formula (plan §5.4)
# ---------------------------------------------------------------------------


def test_t_run_formula_matches_plan_example() -> None:
    import scripts.campaigns.run_v5_sprint as drv

    # plan §5.4 example: step=0.10s, eval=10s, save=15s
    assert drv.t_run_seconds(1500, 0.10, 10.0, 15.0) == pytest.approx(225.0)
    assert drv.t_run_seconds(3000, 0.10, 10.0, 15.0) == pytest.approx(435.0)
    # ceil(U/250) evals
    assert drv.t_run_seconds(251, 0.0, 10.0, 0.0) == pytest.approx(20.0)
    assert drv.t_run_seconds(250, 0.0, 10.0, 0.0) == pytest.approx(10.0)


def test_downgrade_ladder_arithmetic() -> None:
    import scripts.campaigns.run_v5_sprint as drv

    timing_fast = {"t_step_p90_s": 0.10, "t_eval_s": 10.0,
                   "t_start_save_s": 15.0}
    # 6 runs x 225 s = 22.5 min <= 108 -> full 1500 budget
    budget = drv.resolve_screen_updates(timing_fast)
    assert budget["updates"] == 1500
    assert budget["exploratory_only"] is False
    assert budget["feasible"] is True
    assert budget["total_minutes"] == pytest.approx(22.5)

    # slower step: 1500 exceeds 108 min -> uniform downgrade to 1000
    timing_mid = {"t_step_p90_s": 0.80, "t_eval_s": 10.0,
                  "t_start_save_s": 15.0}
    budget = drv.resolve_screen_updates(timing_mid)
    assert budget["updates"] == 1000
    assert budget["exploratory_only"] is False
    assert budget["total_minutes"] <= 108.0

    # even 1000 exceeds -> 500 rung, EXPLORATORY ONLY (recorded)
    timing_slow = {"t_step_p90_s": 1.20, "t_eval_s": 10.0,
                   "t_start_save_s": 15.0}
    budget = drv.resolve_screen_updates(timing_slow)
    assert budget["updates"] == 500
    assert budget["exploratory_only"] is True

    # nothing fits -> infeasible, updates None (caller must refuse)
    timing_absurd = {"t_step_p90_s": 30.0, "t_eval_s": 60.0,
                     "t_start_save_s": 60.0}
    budget = drv.resolve_screen_updates(timing_absurd)
    assert budget["feasible"] is False
    assert budget["updates"] is None

    # recheck: sub-1000 budget cannot declare a stability winner
    recheck = drv.resolve_recheck_updates(timing_slow)
    assert recheck["feasible"] is True
    assert recheck["updates"] == 500
    assert recheck["stable_winner_allowed"] is False
    recheck_ok = drv.resolve_recheck_updates(timing_fast)
    assert recheck_ok["updates"] == 1500
    assert recheck_ok["stable_winner_allowed"] is True

    # crossfit feasibility ladder
    full = drv.resolve_crossfit_mode(timing_fast)
    assert full["mode"] == "full_15"
    assert full["status_on_completion"] == "complete_oof_3seed"
    single = drv.resolve_crossfit_mode(
        {"t_step_p90_s": 0.45, "t_eval_s": 10.0, "t_start_save_s": 15.0})
    assert single["mode"] == "single_seed_5"
    assert single["status_on_completion"] == "complete_oof_1seed"
    none = drv.resolve_crossfit_mode(
        {"t_step_p90_s": 5.0, "t_eval_s": 30.0, "t_start_save_s": 30.0})
    assert none["mode"] == "archive_only"
    assert none["feasible"] is False


# ---------------------------------------------------------------------------
# 4. Ranking + tie-break on synthetic metrics (plan §5.3)
# ---------------------------------------------------------------------------


def test_ranking_primary_then_tie_break_chain() -> None:
    import scripts.campaigns.run_v5_sprint as drv

    rows = [
        {"arm_id": "C05", "rmse_over_zero": 0.55, "near_ts_cos": 0.80,
         "static_violation_rate": 0.004},
        {"arm_id": "A08", "rmse_over_zero": 0.54, "near_ts_cos": 0.30,
         "static_violation_rate": 0.001},
        {"arm_id": "A12", "rmse_over_zero": 0.545, "near_ts_cos": 0.90,
         "static_violation_rate": 0.002},
        {"arm_id": "C12", "rmse_over_zero": 0.90, "near_ts_cos": 0.70,
         "static_violation_rate": 0.000},
        {"arm_id": "A05", "rmse_over_zero": None, "censored": True},
    ]
    ranked = drv.rank_screen_arms(rows)
    by_arm = {r["arm_id"]: r for r in ranked}
    # A08/A12/C05 all lie within 0.02 of the best primary (0.54..0.56):
    # the contending set is re-ordered by near-TS cos (higher first):
    # A12 (0.90) > C05 (0.80) > A08 (0.30); C12 stays on primary order.
    assert by_arm["A12"]["rank"] == 1
    assert by_arm["C05"]["rank"] == 2
    assert by_arm["A08"]["rank"] == 3
    assert by_arm["C12"]["rank"] == 4
    assert by_arm["A05"]["rank"] is None
    assert by_arm["A12"]["tie_break"] == "near_ts_cos+static_violation"
    # second tie-break level: near-TS equal -> lower static violation wins
    tied = drv.rank_screen_arms([
        {"arm_id": "X", "rmse_over_zero": 0.50, "near_ts_cos": 0.70,
         "static_violation_rate": 0.010},
        {"arm_id": "Y", "rmse_over_zero": 0.50, "near_ts_cos": 0.70,
         "static_violation_rate": 0.002},
    ])
    assert [r["arm_id"] for r in tied] == ["Y", "X"]
    # large primary gap disables the tie-break: raw ordering preserved
    far = drv.rank_screen_arms([
        {"arm_id": "P", "rmse_over_zero": 0.50, "near_ts_cos": 0.10},
        {"arm_id": "Q", "rmse_over_zero": 0.60, "near_ts_cos": 0.99},
    ])
    assert [r["arm_id"] for r in far] == ["P", "Q"]
    assert far[0]["tie_break"] == "primary_rmse_over_zero"


def test_eval_curve_stability_flag() -> None:
    import scripts.campaigns.run_v5_sprint as drv

    assert drv.is_eval_curve_stable([0.9, 0.7, 0.6, 0.55]) is True
    assert drv.is_eval_curve_stable([0.9, 0.5, 0.8]) is False  # regressed
    assert drv.is_eval_curve_stable([0.9, float("nan")]) is False
    assert drv.is_eval_curve_stable([]) is False


# ---------------------------------------------------------------------------
# 5. Status vocabulary (plan §5.7 — EXACT)
# ---------------------------------------------------------------------------


def test_status_vocabulary_is_exact() -> None:
    import scripts.campaigns.run_v5_sprint as drv

    assert drv.STAGE_STATUS_VALUES == (
        "complete_label_screen",
        "complete_recipe_recheck",
        "complete_oof_3seed",
        "complete_oof_1seed",
        "censored_budget",
        "rejected_data",
        "rejected_gate",
        "insufficient_evidence",
    )
    for status in drv.STAGE_STATUS_VALUES:
        assert drv.validate_status(status) == status
    with pytest.raises(ValueError):
        drv.validate_status("complete_prepare_audit")
    with pytest.raises(ValueError):
        drv.validate_status("done")


# ---------------------------------------------------------------------------
# 6. Censored path with injected fake clock
# ---------------------------------------------------------------------------


def _fake_clock_driver(tmp_path, monkeypatch, offset_s: float):
    import time as _time

    import scripts.campaigns.run_v5_sprint as drv

    monkeypatch.chdir(tmp_path)
    # minimal parent stubs so the precheck passes
    (tmp_path / "data/manifests/v5").mkdir(parents=True)
    (tmp_path / "data/manifests/v5/split_manifest.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    # T0 far in the past (default: 400 min) -> every stage window expired
    (out / "campaign_manifest.json").write_text(json.dumps({
        "schema_version": drv.MANIFEST_SCHEMA,
        "campaign_id": drv.CAMPAIGN_ID,
        "T0_unix": _time.time() - float(offset_s or 400 * 60.0),
        "stages": {},
        "parents": {},
    }))
    (out / "timing_probe.json").write_text(json.dumps({
        "t_step_p90_s": 0.1, "t_eval_s": 1.0, "t_start_save_s": 1.0}))
    return drv.SprintDriver(
        config_path=PROJECT_ROOT / "config/campaigns/v5_sprint.yaml",
        out_root=out, clock=_time.time)


def test_censored_path_saves_state_and_exits_typed(tmp_path,
                                                   monkeypatch) -> None:
    import scripts.campaigns.run_v5_sprint as drv

    driver = _fake_clock_driver(tmp_path, monkeypatch, 0.0)
    # any actual process launch is a test failure; the expired window must
    # censor BEFORE subprocess.run is reached
    monkeypatch.setattr(
        drv.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("no subprocess may launch after window end")))
    status = driver.run_stage("label-screen")
    assert status == "censored_budget"
    # censored state saved: manifest stage entry + decision row
    manifest = json.loads(driver.manifest_path.read_text())
    assert manifest["stages"]["label-screen"]["status"] == "censored_budget"
    decisions = [json.loads(line) for line in
                 driver.decisions_path.read_text().splitlines() if line.strip()]
    assert any(d.get("stage") == "label-screen"
               and d.get("status") == "censored_budget" for d in decisions)


def test_launch_guard_stops_subprocess_before_window_end(tmp_path,
                                                        monkeypatch) -> None:
    import time as _time

    import scripts.campaigns.run_v5_sprint as drv

    # T0 such that label-screen is inside its last-5-minute guard
    t0 = _time.time() - (210 - 3) * 60.0
    window = drv.StageWindow(t0, "label-screen")
    assert window.may_launch() is False
    assert window.expired() is False
    with pytest.raises(drv.StageCensored):
        window.ensure_launch()
    # fully expired window
    expired = drv.StageWindow(_time.time() - 400 * 60.0, "label-screen")
    assert expired.expired() is True
    with pytest.raises(drv.StageCensored):
        expired.ensure_launch()


# ---------------------------------------------------------------------------
# 7. Sealed eval refusal echo (plan §5.5; ADR-0026 §4)
# ---------------------------------------------------------------------------


def test_sealed_eval_refusal_echoes_registration() -> None:
    registration = {"campaign_id": "V5_SPRINT",
                    "split_sha256": "deadbeef", "fold": 0}
    decision = sealed_eval_request(
        registration, crossfit_complete_runs=15,
        model_hash="a" * 64, config_hash="b" * 64, protocol_hash="c" * 64,
        unseal=True, allow_unseal=False)
    assert decision["status"] == "REFUSED"
    assert decision["code"] == "sealed_confirm_not_opened"
    assert decision["registration"] == registration
    assert any("policy" in reason for reason in decision["reasons"])
    # missing qualification dimensions are itemized
    weak = sealed_eval_request(
        {}, crossfit_complete_runs=5, model_hash=None, config_hash=None,
        protocol_hash=None, unseal=False, allow_unseal=False)
    assert weak["status"] == "REFUSED"
    assert len(weak["reasons"]) >= 4
    # fully-qualified request under an unsealing protocol (NOT this sprint)
    qualified = sealed_eval_request(
        registration, crossfit_complete_runs=15,
        model_hash="a" * 64, config_hash="b" * 64, protocol_hash="c" * 64,
        unseal=True, allow_unseal=True)
    assert qualified["status"] == "QUALIFIED"
    assert qualified["code"] is None


def test_cli_unseal_flag_still_refused(tmp_path, monkeypatch) -> None:
    driver = _fake_clock_driver(tmp_path, monkeypatch, 0.0)
    decision = sealed_eval_request(
        {"campaign_id": "V5_SPRINT"}, crossfit_complete_runs=15,
        model_hash="a" * 64, config_hash="b" * 64, protocol_hash="c" * 64,
        unseal=driver.unseal, allow_unseal=False)
    assert decision["status"] == "REFUSED"
    assert decision["code"] == "sealed_confirm_not_opened"


# ---------------------------------------------------------------------------
# 8. G5.4 synthetic static checks (plan §6 G5.4)
# ---------------------------------------------------------------------------


def test_g54_clash_includes_bonded_pairs() -> None:
    # H2 compressed to 0.40 A: below 0.75*(0.31+0.31)=0.465 A -> clash even
    # though the pair is BONDED (no pre-registered exception exists).
    sample = _h2_sample(distance=0.40)
    clashes = clash_pairs_of(sample["pos"], sample["atomic_numbers"])
    assert clashes == [[0, 1]]
    # relaxed geometry at the start distance: no clash
    ok = clash_pairs_of(_h2_sample(0.74)["pos"], [1, 1])
    assert ok == []


def test_g54_protected_stretch_and_shorten() -> None:
    sample = _water_like_sample()
    protected = protected_edges_of(sample["start_bonds"], sample["k_pairs"])
    assert protected == [(0, 1), (0, 2)]
    # stretch O-H1 beyond 1.35*d_eq
    stretched = sample["pos"].copy()
    stretched[1, 0] = 0.96 * 1.40
    violations = protected_violations_of(stretched, protected,
                                         sample["pos_start"])
    assert [v["kind"] for v in violations] == ["stretch"]
    assert violations[0]["edge"] == [0, 1]
    # shorten O-H2 below 0.70*d_eq
    shortened = sample["pos"].copy()
    shortened[2] = np.array([0.96 * 0.60, 0.0, 0.0])
    violations = protected_violations_of(shortened, protected,
                                         sample["pos_start"])
    assert [v["kind"] for v in violations] == ["shorten"]
    # in-range geometry: clean
    assert protected_violations_of(sample["pos"], protected,
                                   sample["pos_start"]) == []


def test_g54_strict_copy_trip() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    key_rows = [0]
    # honest patch: only the key atom moves
    patched = patch_strict_copy(pos, np.array([[0.0, 0.1, 0.0]]), key_rows)
    assert np.array_equal(patched[1:], pos[1:])
    assert np.allclose(patched[0], pos[0] + np.array([0.0, 0.1, 0.0]))
    # dishonest patch: a non-key atom moved -> typed refusal
    forged = patched.copy()
    forged[2, 0] = 2.5
    with pytest.raises(KeyAtomProtocolError) as excinfo:
        assert_strict_copy(pos, forged, key_rows)
    assert excinfo.value.code == "STRICT_COPY_VIOLATION"


def test_g54_gate_on_synthetic_violations() -> None:
    # input is CLEAN; the prediction compresses H2 into a bonded clash
    # -> new-clash rate 1.0 -> gate fails (input violations are never
    # laundered, but only NEW violations count against the model)
    samples = [_h2_sample(0.74) for _ in range(4)]
    clashing = [np.array([[-0.40, 0.0, 0.0]]) for _ in range(4)]
    report = evaluate_g54(samples, clashing)
    assert report["rates"]["new_clash_rate"] == pytest.approx(1.0)
    assert report["gate"]["pass"] is False
    # an ALREADY-clashing input with a zero prediction: no NEW violation,
    # so the model is not blamed (the input flaw stays in the denominator)
    dirty = [_h2_sample(0.40) for _ in range(4)]
    report = evaluate_g54(dirty, [np.zeros((1, 3)) for _ in range(4)])
    assert report["rates"]["new_clash_rate"] == 0.0
    assert report["rates"]["input_clash_rate"] == pytest.approx(1.0)
    # benign bounded prediction on the relaxed geometry -> clean pass
    good = [_h2_sample(0.74) for _ in range(4)]
    benign = [np.array([[0.02, 0.0, 0.0]]) for _ in range(4)]
    report = evaluate_g54(good, benign)
    assert report["rates"]["new_clash_rate"] == 0.0
    assert report["rates"]["new_protected_rate"] == 0.0
    assert report["gate"]["pass"] is True
    # refusal accounting: None prediction counts as refusal in denominator
    refused = evaluate_g54(good, [None] * 4)
    assert refused["n_refused"] == 4
    assert refused["rates"]["refusal_rate"] == pytest.approx(1.0)
    assert refused["gate"]["pass"] is False


def test_oracle_static_ceiling_pass_and_failure_typing() -> None:
    # teacher displacements that keep the geometry clean -> OK
    good = [_h2_sample(0.74) for _ in range(3)]
    teachers = [np.array([[0.01, 0.0, 0.0]]) for _ in range(3)]
    ceiling = oracle_static_ceiling(good, teachers)
    assert ceiling["status"] == "ORACLE_STATIC_OK"
    assert ceiling["action_representation_flagged"] is False
    # teacher that compresses H2 into a clash -> typed failure flags the
    # ACTION REPRESENTATION, never the network
    bad = [_h2_sample(0.50) for _ in range(3)]
    teachers = [np.array([[-0.12, 0.0, 0.0]]) for _ in range(3)]
    ceiling = oracle_static_ceiling(bad, teachers)
    assert ceiling["status"] == "ORACLE_STATIC_FAILED"
    assert ceiling["action_representation_flagged"] is True
    assert ceiling["report"]["gate"]["pass"] is False


def test_follower_counterfactual_is_diagnostic_only() -> None:
    sample = _h2_sample(0.74)
    # H1 is terminal, bonded to key heavy atom H0?  H0 is hydrogen, so no
    # HEAVY key atom exists -> no follower (H is never "heavy")
    assert follower_h_rows(sample["atomic_numbers"], sample["key_rows"],
                           sample["start_bonds"]) == []
    # rebuild with a heavy key atom: H2O-like where O is key and H terminal
    water = _water_like_sample()
    followers = follower_h_rows(water["atomic_numbers"], water["key_rows"],
                                water["start_bonds"])
    assert followers == [(1, 0), (2, 0)]
    preds = [np.array([[0.02, 0.01, 0.0]])]
    diag = follower_counterfactual_diagnostics([water], preds)
    assert diag[0]["contract_counterfactual"] is True
    assert diag[0]["n_followers"] == 2
    assert "excluded from the G5.4 gate" in diag[0]["note"]
    # bridging H (degree 2) is excluded: add an H bonded to two heavy atoms
    bridging = {
        "group_id": "RXN_BRIDGE",
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.5, 0.9, 0.0], [0.5, -0.9, 0.0]]),
        "pos_start": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [0.5, 0.9, 0.0], [0.5, -0.9, 0.0]]),
        "atomic_numbers": [6, 6, 1, 1],
        "key_rows": [0],
        "k_pairs": [],
        "start_bonds": [(0, 1, 1.0), (2, 0, 1.0), (2, 1, 1.0), (3, 0, 1.0)],
    }
    # row 2 is bonded to BOTH heavy atoms -> bridging, excluded; row 3 is
    # terminal on key atom 0 -> included
    assert follower_h_rows(bridging["atomic_numbers"], bridging["key_rows"],
                           bridging["start_bonds"]) == [(3, 0)]


# ---------------------------------------------------------------------------
# 9. G5.3 synthetic sign accuracy + SRE (plan §6 G5.3)
# ---------------------------------------------------------------------------


def _ab_sample(pos, key_rows, pairs):
    return {
        "group_id": "RXN_G53", "reaction_id": "RXN_G53", "branch": 0,
        "pos": np.asarray(pos, dtype=np.float64),
        "key_rows": list(key_rows),
        "pairs": list(pairs),
    }


def test_g53_exact_delta_d_and_first_order() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    key_rows = [0, 1]
    # stretch the bond by 0.10 A along x via u_j
    u = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    dd = exact_delta_d(pos, key_rows, [(0, 1)], u)
    assert dd[0] == pytest.approx(0.10, abs=1e-12)
    j = first_order_pair_projection(pos, key_rows, [(0, 1)], u)
    assert j[0] == pytest.approx(0.10, abs=1e-12)
    # transverse displacement: J = 0 (first order) while the EXACT Δd is
    # second-order positive — this curvature is what G5.3's J-vs-Δd
    # deviation report captures
    u_trans = np.array([[0.0, 0.0, 0.0], [0.0, 0.10, 0.0]])
    dd_t = exact_delta_d(pos, key_rows, [(0, 1)], u_trans)
    j_t = first_order_pair_projection(pos, key_rows, [(0, 1)], u_trans)
    assert j_t[0] == pytest.approx(0.0, abs=1e-12)
    assert dd_t[0] == pytest.approx(
        math.sqrt(1.5 ** 2 + 0.01) - 1.5, abs=1e-12)
    assert dd_t[0] > 0.003  # second-order curvature, invisible to J


def test_g53_sign_accuracy_synthetic() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    y = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])   # stretches (+)
    good_u = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])  # same sign
    bad_u = np.array([[0.0, 0.0, 0.0], [-0.03, 0.0, 0.0]])  # opposite
    abstain_u = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])  # |Δd|<0.005
    base = _ab_sample(pos, [0, 1], [(0, 1)])
    report = evaluate_g53([{**base, "u": good_u, "y": y}])
    assert report["aggregate"]["n_sign_pairs"] == 1
    assert report["aggregate"]["sign_accuracy_cluster_median"] == 1.0
    report = evaluate_g53([{**base, "u": bad_u, "y": y}])
    assert report["aggregate"]["sign_accuracy_cluster_median"] == 0.0
    # abstention (|Δd(u)|<0.005) counts as INCORRECT in the main denominator
    report = evaluate_g53([{**base, "u": abstain_u, "y": y}])
    assert report["aggregate"]["n_sign_pairs"] == 1
    assert report["aggregate"]["n_sign_pairs_correct"] == 0
    assert report["aggregate"]["sign_accuracy_cluster_median"] == 0.0
    # a sub-threshold teacher change contributes NO sign pair
    tiny_y = y * 0.1  # |Δd(y)| = 0.002 < 0.005
    report = evaluate_g53([{**base, "u": good_u, "y": tiny_y}])
    assert report["aggregate"]["n_sign_pairs"] == 0


def test_g53_sre_synthetic() -> None:
    # two-bond frame: teacher moves 0.03/0.01 -> p = (0.75, 0.25)
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    key_rows = [0, 1, 2]
    pairs = [(0, 1), (1, 2)]
    y = np.array([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    proportional = y * 2.0            # same proportions -> SRE = 0
    zero_pred = np.zeros_like(y)      # fully zero -> SRE = 1
    lopsided = np.array([[-0.04, 0.0, 0.0], [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])  # all mass on bond 1
    base = _ab_sample(pos, key_rows, pairs)
    report = evaluate_g53([{**base, "u": proportional, "y": y}])
    assert report["aggregate"]["n_sre_frames"] == 1
    assert report["aggregate"]["sre_cluster_median"] == pytest.approx(0.0,
                                                                      abs=1e-12)
    report = evaluate_g53([{**base, "u": zero_pred, "y": y}])
    assert report["aggregate"]["sre_cluster_median"] == pytest.approx(1.0)
    report = evaluate_g53([{**base, "u": lopsided, "y": y}])
    # p_hat = (1, 0); SRE = 0.5*(|1-0.75| + |0-0.25|) = 0.25
    assert report["aggregate"]["sre_cluster_median"] == pytest.approx(0.25)
    # single-bond frames are EXCLUDED from SRE (needs >=2 pairs)
    single = _ab_sample(pos, [0, 1], [(0, 1)])
    report = evaluate_g53([{**single, "u": proportional[:2], "y": y[:2]}])
    assert report["aggregate"]["n_sre_frames"] == 0


def test_g53_gate_pass_and_fail() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    pairs = [(0, 1), (1, 2)]
    y = np.array([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    base = _ab_sample(pos, [0, 1, 2], pairs)
    good = evaluate_g53([{**base, "u": y, "y": y}])
    assert good["gate"]["pass"] is True
    bad = evaluate_g53([{**base, "u": -y, "y": y}])
    assert bad["gate"]["pass"] is False


# ---------------------------------------------------------------------------
# 10. G5.2 + gate report writer smoke (plan §6/§8)
# ---------------------------------------------------------------------------


def test_g52_undefined_denominator_listed_not_zero() -> None:
    # one cluster whose teacher is (near) zero -> ratio undefined
    y = np.zeros((2, 1, 3))
    y[0, 0, 0] = 0.5
    u = y * 0.5
    report = evaluate_g52(
        u, y, groups=["A", "B"], classic=None,
        bootstrap_draws=50, bootstrap_seed=7)
    undefined = report["raw"]["undefined"]["rmse_over_zero"]
    assert "B" in undefined
    per_cluster = report["raw"]["per_cluster"]
    assert per_cluster["B"]["rmse_over_zero"] is None
    # never coerced to 0
    assert per_cluster["B"]["rmse_over_zero"] != 0.0
    assert report["raw"]["aggregate"]["n_clusters"] == 2


def test_g52_raw_and_bounded_blocks() -> None:
    y = np.ones((4, 2, 3)) * 0.05
    u_raw = y * 2.0
    u_bounded = y * 1.0
    report = evaluate_g52(
        {"raw": u_raw, "bounded": u_bounded}, y,
        groups=["G1"] * 4, classic=np.zeros_like(y),
        bootstrap_draws=50, bootstrap_seed=7)
    assert report["raw"]["aggregate"]["median_rmse_over_zero"] > \
        report["bounded"]["aggregate"]["median_rmse_over_zero"]
    assert report["gate"] is report["raw"]["gate"]


def test_gate_report_writer_distinguishes_statuses(tmp_path) -> None:
    gates = {
        "G5.1": {"status": "measured", "verdict": "pass"},
        "G5.2": {"status": "censored", "verdict": "—"},
        "G5.3": {"status": "planned", "verdict": "—"},
        "G5.4": {"status": "not_executed", "verdict": "—"},
    }
    artifacts = [{"name": "timing_probe", "schema": "s1",
                  "path": None, "status": "measured"}]
    paths = write_gate_report(
        tmp_path, gates=gates, artifacts=artifacts,
        stage_statuses={"label-screen": "censored_budget"},
        qc="QC not scheduled")
    doc = json.loads(Path(paths["gate_report_json"]["path"]).read_text())
    assert doc["schema_version"] == "pesn_v5_gate_report_v1"
    assert doc["gates"]["G5.2"]["status"] == "censored"
    assert doc["qc"] == "QC not scheduled"
    md = Path(paths["gate_report_md"]["path"]).read_text()
    assert "QC not scheduled" in md
    # invalid status -> typed refusal
    with pytest.raises(KeyAtomProtocolError):
        write_gate_report(tmp_path / "bad",
                          gates={"G5.1": {"status": "done"}},
                          artifacts=[])
