"""V5.1a campaign driver acceptance: CLI + protocol freeze + stage enumeration.

Covers hermetically (no GPU, no real data, no training — ``_spawn`` is
monkeypatched everywhere a subprocess would launch):

* stage routing + unknown-stage exit 2; dry-run writes NOTHING;
* ``t_run_seconds(..., eval_every=50)`` arithmetic (V5 default 250 intact);
* the frozen protocol: payload imports gate thresholds from
  ``key_atom_protocols`` (never retyped), freezes BEFORE diagnose, and
  refuses later stages when absent or config-sha-mismatched — with zero
  spawn attempts;
* diagnose spawns exactly the D1 legacy run (B0/fold0/seed17/3000/mode v51);
* ablate enumerates exactly 12 runs (B0–B3 × {17,29,43}, mode v51) with
  confirmatory-only marks for magnitude-head arms under a
  "checkpoint problem only" D1 verdict;
* confirm enumerates 5 confirm + 4 B0 reference runs and freezes the recipe;
* sealed eval stays REFUSED under policy ``v5_1a_never_unseals`` (even with
  ``--unseal``); no stage ever passes ``--unseal`` to a subprocess;
* the report labels every OOF claim development-evidence-only and never
  writes a ``complete_oof*`` status anywhere.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (  # noqa: E402
    G52_BETTER_THAN_ZERO_FRACTION_MIN,
    G52_COS_NEAR_TS_MIN,
    G52_COS_OVERALL_MIN,
    G52_MAE_OVER_ZERO_MAX,
    G52_Q75_RMSE_OVER_ZERO_MAX,
    G52_RMSE_OVER_CLASSIC_MAX,
    G52_RMSE_OVER_ZERO_MAX,
    G53_SIGN_MEDIAN_MIN,
    G53_SIGN_Q25_MIN,
    G53_SRE_MEDIAN_MAX,
    G53_SRE_Q75_MAX,
    G54_CLUSTER_MAX,
    G54_NEW_CLASH_MAX,
    G54_NEW_PROTECTED_MAX,
    G54_REFUSAL_MAX,
    G54_UPPER_BOUND_MAX,
    sealed_eval_request,
)

import scripts.campaigns.run_v5_1_sprint as drv  # noqa: E402
import scripts.campaigns.run_v5_sprint as v5  # noqa: E402

CONFIG = PROJECT_ROOT / "config/campaigns/v5_1_sprint.yaml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_env(tmp_path: Path, monkeypatch) -> Path:
    """Fresh cwd with a stub frozen V5 split manifest (parents verified)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/manifests/v5").mkdir(parents=True)
    (tmp_path / "data/manifests/v5/split_manifest.json").write_text(
        '{"schema_version": "pesn_v5_split_manifest_v1"}', encoding="utf-8")
    return tmp_path


def _patch_heavy(monkeypatch) -> None:
    """Stub the GPU timing probe and the G5.1 pytest subprocess."""

    def fake_g51(self) -> dict:
        return {"returncode": 0,
                "files": list(drv.G51_TEST_FILES),
                "stdout_tail": ["stubbed G5.1 green"],
                "v51_trainer_test": "absent — skipped (stubbed)",
                "g51_pass": True}

    def fake_probe(self) -> dict:
        probe = {"schema_version": "pesn_v51_timing_probe_v1",
                 "t_step_p90_s": 0.10, "t_eval_s": 1.0,
                 "t_start_save_s": 1.0,
                 "eval_every_updates": self.eval_every}
        self.out_root.mkdir(parents=True, exist_ok=True)
        (self.out_root / "timing_probe.json").write_text(
            json.dumps(probe), encoding="utf-8")
        return probe

    monkeypatch.setattr(drv.V51SprintDriver, "_g51_evidence", fake_g51)
    monkeypatch.setattr(drv.V51SprintDriver, "_timing_probe", fake_probe)


def _prepared_driver(tmp_path: Path, monkeypatch, *,
                     unseal: bool = False) -> "drv.V51SprintDriver":
    _make_env(tmp_path, monkeypatch)
    _patch_heavy(monkeypatch)
    driver = drv.V51SprintDriver(config_path=CONFIG, unseal=unseal)
    assert driver.run_stage("prepare") == "complete_prepare_v51"
    return driver


def _fake_spawn(argvs: list, *, rmse: float = 0.45):
    """Monkeypatched ``_spawn``: records argv, fakes a complete run."""

    def fake(self, argv, *, stage, window):
        argvs.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        updates = int(argv[argv.index("--updates") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "status.json").write_text(json.dumps({
            "status": "complete", "detail": "", "updates_run": updates,
            "mode": argv[argv.index("--mode") + 1]}), encoding="utf-8")
        (out / "eval.jsonl").write_text(json.dumps({
            "update": updates, "wall_s": 1.5,
            "aggregate": {"rmse_over_zero": rmse,
                          "cos_eff_near_ts_median": 0.80,
                          "sre": 0.05, "sign_accuracy": 0.95},
            "per_cluster": {}, "aggregate_bounded": {},
            "per_cluster_bounded": {}}) + "\n", encoding="utf-8")

        class _Proc:
            returncode = 0

        return _Proc()

    return fake


def _argv_value(argv, flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _no_unseal(argvs: list) -> None:
    for argv in argvs:
        assert "--unseal" not in argv, argv


# ---------------------------------------------------------------------------
# 1. Stage routing + unknown stage exit 2
# ---------------------------------------------------------------------------


def test_unknown_stage_exits_2() -> None:
    result = subprocess.run(
        [sys.executable,
         str(PROJECT_ROOT / "scripts/campaigns/run_v5_1_sprint.py"),
         "--config", str(CONFIG), "--stage", "bogus"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120)
    assert result.returncode == 2, result.stderr
    assert "invalid choice" in result.stderr


def test_stage_routing_table_is_complete() -> None:
    assert drv.V51_STAGE_ORDER == (
        "prepare", "diagnose", "ablate", "confirm", "report")
    for stage in drv.V51_STAGE_ORDER:
        handler = drv.V51SprintDriver.STAGE_HANDLERS[stage]
        assert hasattr(drv.V51SprintDriver, handler), (stage, handler)
    # V5.1a NEVER writes a formal-OOF status — the vocabulary forbids it.
    assert not any(status.startswith("complete_oof")
                   for status in drv.V51_STATUS_VALUES)
    assert drv.V51_MANIFEST_SCHEMA == "pesn_v5_1_sprint_manifest_v1"
    assert drv.DEFAULT_CAMPAIGN_ID == "V5_1A_SPRINT"


# ---------------------------------------------------------------------------
# 2. Dry-run writes nothing
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for stage in drv.V51_STAGE_ORDER:
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts/campaigns/run_v5_1_sprint.py"),
             "--config", str(CONFIG), "--stage", stage, "--dry-run"],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=120)
        assert result.returncode == 0, (stage, result.stderr)
        assert "[dry-run]" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_dry_run_driver_creates_no_manifest(tmp_path, monkeypatch) -> None:
    _make_env(tmp_path, monkeypatch)
    driver = drv.V51SprintDriver(config_path=CONFIG, dry_run=True)
    assert driver.run_stage("report") == "dry_run"
    assert not driver.manifest_path.exists()
    assert not Path("outputs/pesn/v5_1/sprint").exists()


# ---------------------------------------------------------------------------
# 3. t_run_seconds eval-interval arithmetic (additive V5 edit)
# ---------------------------------------------------------------------------


def test_t_run_seconds_eval_every_arithmetic() -> None:
    # V5 default (eval_every=250) — byte-identical to the original formula.
    assert v5.t_run_seconds(3000, 0.10, 10.0, 15.0) == pytest.approx(435.0)
    assert v5.t_run_seconds(251, 0.0, 10.0, 0.0) == pytest.approx(20.0)
    # V5.1 inner-val cadence: ceil(U/50) evals.
    assert v5.t_run_seconds(3000, 0.10, 10.0, 15.0,
                            eval_every=50) == pytest.approx(300 + 600 + 15)
    assert v5.t_run_seconds(251, 0.0, 10.0, 0.0,
                            eval_every=50) == pytest.approx(60.0)
    assert v5.t_run_seconds(50, 0.0, 10.0, 0.0,
                            eval_every=50) == pytest.approx(10.0)
    assert v5.t_run_seconds(1, 0.0, 10.0, 0.0,
                            eval_every=50) == pytest.approx(10.0)
    # re-exported by the V5.1 driver
    assert drv.t_run_seconds(3000, 0.10, 10.0, 15.0,
                             eval_every=50) == pytest.approx(915.0)
    with pytest.raises(drv.SprintRefused) as excinfo:
        v5.t_run_seconds(100, 0.1, 1.0, 1.0, eval_every=0)
    assert excinfo.value.code == "TIMING_EVAL_EVERY"


# ---------------------------------------------------------------------------
# 4. Frozen protocol payload (thresholds imported, never retyped)
# ---------------------------------------------------------------------------


def test_protocol_payload_thresholds_imported(tmp_path, monkeypatch) -> None:
    _make_env(tmp_path, monkeypatch)
    driver = drv.V51SprintDriver(config_path=CONFIG)
    payload = driver._protocol_payload()

    g52 = payload["gate_thresholds"]["G5.2"]
    assert g52["mae_over_zero_max"] == G52_MAE_OVER_ZERO_MAX == 0.80
    assert g52["rmse_over_zero_max"] == G52_RMSE_OVER_ZERO_MAX == 0.80
    assert g52["rmse_over_classic_max"] == G52_RMSE_OVER_CLASSIC_MAX == 0.90
    assert g52["q75_rmse_over_zero_max"] == G52_Q75_RMSE_OVER_ZERO_MAX == 1.00
    assert (g52["frac_better_than_zero_min"]
            == G52_BETTER_THAN_ZERO_FRACTION_MIN == 0.70)
    assert g52["cos_overall_min"] == G52_COS_OVERALL_MIN == 0.60
    assert g52["cos_near_ts_min"] == G52_COS_NEAR_TS_MIN == 0.50
    g53 = payload["gate_thresholds"]["G5.3"]
    assert g53["sign_median_min"] == G53_SIGN_MEDIAN_MIN == 0.90
    assert g53["sign_q25_min"] == G53_SIGN_Q25_MIN == 0.80
    assert g53["sre_median_max"] == G53_SRE_MEDIAN_MAX == 0.15
    assert g53["sre_q75_max"] == G53_SRE_Q75_MAX == 0.25
    assert g53["dead_zone_abstention_max"] == 0.10
    g54 = payload["gate_thresholds"]["G5.4"]
    assert g54["new_clash_max"] == G54_NEW_CLASH_MAX == 0.005
    assert g54["new_protected_max"] == G54_NEW_PROTECTED_MAX == 0.01
    assert g54["bootstrap_upper95_max"] == G54_UPPER_BOUND_MAX == 0.02
    assert g54["per_cluster_max"] == G54_CLUSTER_MAX == 0.05
    assert g54["refusal_max"] == G54_REFUSAL_MAX == 0.05

    full = payload["g52_v51_full_oof_protocol"]
    assert "49-cluster cluster-median" in full["aggregation"]
    assert full["bootstrap_draws"] == 2000
    assert full["bootstrap_seed"] == 20261018
    assert full["classic_improvement_lower95_gt"] == 0.0
    assert full["folds_point_estimate_pass_min"] == "4/5"
    assert full["fold_rmse_over_zero_median_max"] == 1.0
    assert "secondary/historical" in full["v5_9_cluster_gate"]
    assert "NO formal OOF" in full["status"]

    m1 = payload["m1_amplitude_gate"]
    assert m1["alpha_median_pass_range"] == [0.85, 1.05]
    assert m1["alpha_q25_min"] == 0.70 and m1["alpha_q75_max"] == 1.30
    assert payload["s1_stability_gate"]["seeds_pass_min"] == "2/3"
    assert payload["s1_stability_gate"]["ensemble_required"] is True
    stretch = payload["stretch_targets"]
    assert stretch["alpha_median_pass_range"] == [0.90, 1.05]
    assert stretch["sign_median_min"] == 0.92 and stretch["sign_q25_min"] == 0.85
    assert stretch["sre_median_max"] == 0.12 and stretch["sre_q75_max"] == 0.20
    assert stretch["dead_zone_abstention_max"] == 0.05

    tree = payload["d1_decision_tree"]
    assert tree["alpha_best_ge_0.85"] == "checkpoint_problem_only"
    assert tree["0.70_le_alpha_best_lt_0.85"] == "post_hoc_calibration_first"
    assert tree["alpha_best_lt_0.70"] == "magnitude_head_justified"

    lambdas = payload["frozen_lambdas"]
    assert lambdas["lambda_cos"] == 0.02
    assert lambdas["lambda_mag"] == 0.5
    assert lambdas["lambda_sign"] == 0.2
    assert lambdas["lambda_dead"] == 0.1
    assert lambdas["ramp"] == {"start_update": 250, "ramp_updates": 250}

    inner_val = payload["inner_val_rule"]
    assert inner_val["n_train_clusters"] == 32
    assert inner_val["n_val_clusters"] == 8
    assert inner_val["stratify"] == ["family", "size"]
    assert inner_val["seed"] == 20261019
    assert inner_val["eval_every_updates"] == 50
    assert inner_val["rmse_tolerance"] == 1.02
    assert inner_val["patience_checks"] == 8
    assert inner_val["min_delta"] == 0.005
    assert inner_val["min_update_floor"] == 500

    matrix = payload["arm_matrix"]
    assert matrix["arms"] == ["B0", "B1", "B2", "B3"]
    table = matrix["arm_table"]
    assert table["B0"]["magnitude_head"] is False
    assert table["B0"]["pair_loss"] == "legacy_j"
    assert table["B1"]["magnitude_head"] is True
    assert table["B1"]["pair_loss"] == "legacy_j"
    assert table["B2"]["magnitude_head"] is False
    assert table["B2"]["pair_loss"] == "exact_trio"
    assert table["B3"]["magnitude_head"] is True
    assert table["B3"]["pair_loss"] == "exact_trio"
    assert all(entry["d0"] == 0.05 for entry in table.values())

    rule = payload["best_config_selection_rule"]
    assert "inner-val SRE" in rule["primary"]
    assert "1.05" in rule["constraint"]
    assert rule["tie_break"] == ["|alpha - 1|",
                                 "sign accuracy (higher first)"]

    hashes = payload["hashes"]
    assert hashes["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()).hexdigest()
    assert hashes["split_manifest_sha256"] is not None  # frozen V5 split
    assert "code_commit" in hashes

    assert payload["statement"] == (
        "thresholds frozen pre-training, no post-hoc relaxation")
    assert payload["thresholds_frozen_pre_training"] is True
    assert payload["campaign_id"] == "V5_1A_SPRINT"


def test_select_best_arm_frozen_rule() -> None:
    def rows(sre, rmse, alpha, sign, seeds=(17, 29, 43)):
        return [{"arm": arm, "fold": 0, "seed": seed,
                 "inner_val_sre": sre[arm],
                 "inner_val_rmse_over_zero": rmse[arm],
                 "alpha_median": alpha[arm],
                 "inner_val_sign_accuracy": sign[arm]}
                for arm in ("B0", "B1", "B2", "B3") for seed in seeds]

    all_rows = rows(
        sre={"B0": 0.20, "B1": 0.05, "B2": 0.08, "B3": 0.06},
        rmse={"B0": 0.50, "B1": 0.50, "B2": 0.55, "B3": 0.51},
        alpha={"B0": 0.50, "B1": 1.00, "B2": 0.95, "B3": 1.02},
        sign={"B0": 0.70, "B1": 0.95, "B2": 0.90, "B3": 0.93})
    verdict = drv.select_best_arm(all_rows)
    # B2 violates the rmse ≤ B0×1.05 constraint (0.55 > 0.525);
    # among eligible arms B1 wins on primary SRE (0.05 < 0.06).
    assert verdict["per_arm"]["B2"]["eligible"] is False
    assert verdict["best_arm"] == "B1"
    assert verdict["ranking"][:2] == ["B1", "B3"]
    assert verdict["rmse_limit_over_b0"] == pytest.approx(0.525)
    # tie on SRE → |α−1| decides
    tied = rows(sre={"B0": 0.10, "B1": 0.10, "B2": 0.10, "B3": 0.10},
                rmse={"B0": 0.50, "B1": 0.50, "B2": 0.50, "B3": 0.50},
                alpha={"B0": 0.50, "B1": 1.01, "B2": 1.30, "B3": 0.90},
                sign={"B0": 0.70, "B1": 0.90, "B2": 0.95, "B3": 0.95})
    assert drv.select_best_arm(tied)["best_arm"] == "B1"
    # missing metrics → no fabrication, no winner
    empty = drv.select_best_arm([{"arm": "B0", "seed": 17}])
    assert empty["best_arm"] is None


# ---------------------------------------------------------------------------
# 5. Protocol freeze gates diagnose (absent / mismatched → typed refusal)
# ---------------------------------------------------------------------------


def test_protocol_freeze_gates_diagnose(tmp_path, monkeypatch) -> None:
    _make_env(tmp_path, monkeypatch)
    driver = drv.V51SprintDriver(config_path=CONFIG)
    argvs: list = []
    monkeypatch.setattr(drv.V51SprintDriver, "_spawn", _fake_spawn(argvs))

    # (a) protocol ABSENT → typed refusal, zero spawn attempts
    with pytest.raises(drv.SprintRefused) as excinfo:
        driver.run_stage("diagnose")
    assert excinfo.value.code == "PROTOCOL_NOT_FROZEN"
    assert argvs == []
    manifest = json.loads(driver.manifest_path.read_text())
    assert manifest["stages"]["diagnose"]["status"] == "rejected_gate"

    # (b) freeze → protocol written with the config sha
    driver._freeze_protocol()
    assert driver.protocol_path.is_file()
    payload = json.loads(driver.protocol_path.read_text())
    assert payload["hashes"]["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()).hexdigest()
    assert payload["statement"] == (
        "thresholds frozen pre-training, no post-hoc relaxation")

    # (c) config sha MISMATCH vs the frozen record → typed refusal again,
    #     still zero spawn attempts
    payload["hashes"]["config_sha256"] = "0" * 64
    driver.protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(drv.SprintRefused) as excinfo:
        driver.run_stage("diagnose")
    assert excinfo.value.code == "PROTOCOL_HASH_MISMATCH"
    assert argvs == []
    manifest = json.loads(driver.manifest_path.read_text())
    assert manifest["stages"]["diagnose"]["status"] == "rejected_gate"


def test_prepare_freezes_protocol_last(tmp_path, monkeypatch) -> None:
    driver = _prepared_driver(tmp_path, monkeypatch)
    summary = json.loads(
        (driver.out_root / "prepare_summary_v51.json").read_text())
    names = [c["checkpoint"] for c in summary["checkpoints"]]
    # G5.1 evidence → timing probe → protocol frozen BEFORE any training
    assert names == ["g51_evidence", "timing_probe", "protocol_frozen"]
    assert summary["g51_pass"] is True
    assert summary["protocol_sha256"] is not None
    assert driver.protocol_path.is_file()
    assert summary["eval_every_updates"] == 50
    manifest = json.loads(driver.manifest_path.read_text())
    assert manifest["stages"]["prepare"]["status"] == "complete_prepare_v51"
    assert "complete_oof" not in json.dumps(manifest)


# ---------------------------------------------------------------------------
# 6. Diagnose: D1 argv contract
# ---------------------------------------------------------------------------


def test_diagnose_spawns_d1_argv(tmp_path, monkeypatch) -> None:
    driver = _prepared_driver(tmp_path, monkeypatch)
    argvs: list = []
    monkeypatch.setattr(drv.V51SprintDriver, "_spawn", _fake_spawn(argvs))
    monkeypatch.setattr(
        drv.V51SprintDriver, "_alpha_diagnostics",
        lambda self, run_dir: {
            "schema_version": "pesn_v51_d1_alpha_v1", "status": "measured",
            "alpha_best_median": 0.90,
            "decision_tree_verdict": "checkpoint_problem_only",
            "checkpoint_policy": "checkpoint_best.pt",
            "post_hoc_only": True,
            "development_evidence_only": True})

    assert driver.run_stage("diagnose") == "complete_diagnose_v51"
    assert len(argvs) == 1
    argv = argvs[0]
    assert _argv_value(argv, "--mode") == "v51"
    assert _argv_value(argv, "--arm") == "B0"
    assert _argv_value(argv, "--fold") == "0"
    assert _argv_value(argv, "--seed") == "17"
    assert _argv_value(argv, "--updates") == "3000"
    assert argv[argv.index("--out") + 1].endswith(
        str(Path("outputs/pesn/v5_1/runs/B0_f0_s17_d1")))
    assert _argv_value(argv, "--config") == str(CONFIG)
    _no_unseal(argvs)

    diag_path = (driver.diagnostics_root
                 / "d1_best_checkpoint_alpha.json")
    assert diag_path.is_file()
    diag = json.loads(diag_path.read_text())
    assert diag["decision_tree_verdict"] == "checkpoint_problem_only"
    assert diag["development_evidence_only"] is True
    assert diag["post_hoc_only"] is True


# ---------------------------------------------------------------------------
# 7. Ablate: exactly 12 runs, correct argv, confirmatory-only marks
# ---------------------------------------------------------------------------


def test_ablate_enumerates_exactly_12_runs(tmp_path, monkeypatch) -> None:
    driver = _prepared_driver(tmp_path, monkeypatch)
    # D1 verdict present BEFORE ablate → magnitude arms must be marked
    driver.diagnostics_root.mkdir(parents=True, exist_ok=True)
    (driver.diagnostics_root / "d1_best_checkpoint_alpha.json").write_text(
        json.dumps({"schema_version": "pesn_v51_d1_alpha_v1",
                    "status": "measured", "alpha_best_median": 0.93,
                    "decision_tree_verdict": "checkpoint_problem_only"}),
        encoding="utf-8")

    argvs: list = []
    monkeypatch.setattr(drv.V51SprintDriver, "_spawn", _fake_spawn(argvs))
    assert driver.run_stage("ablate") == "complete_ablate_v51"

    assert len(argvs) == 12
    arms = [_argv_value(a, "--arm") for a in argvs]
    assert arms == (["B0"] * 3 + ["B1"] * 3 + ["B2"] * 3 + ["B3"] * 3)
    assert sorted(set(arms)) == ["B0", "B1", "B2", "B3"]
    assert all(_argv_value(a, "--mode") == "v51" for a in argvs)
    assert all(_argv_value(a, "--fold") == "0" for a in argvs)
    assert all(_argv_value(a, "--updates") == "3000" for a in argvs)
    seeds_by_arm = {arm: sorted(int(_argv_value(a, "--seed"))
                                for a in argvs if _argv_value(a, "--arm") == arm)
                    for arm in ("B0", "B1", "B2", "B3")}
    assert seeds_by_arm == {arm: [17, 29, 43] for arm in
                            ("B0", "B1", "B2", "B3")}
    assert all(Path(_argv_value(a, "--out")).is_relative_to(
        driver.runs_root) for a in argvs)
    _no_unseal(argvs)

    parquet = driver.out_root / "ablation_v51.parquet"
    assert parquet.is_file()
    doc = json.loads((driver.out_root / "ablation_v51.json").read_text())
    assert doc["n_runs"] == 12
    assert doc["decision_tree_verdict"] == "checkpoint_problem_only"
    for row in doc["rows"]:
        assert row["development_evidence_only"] is True
        assert row["selection_exposed"] is True
        assert row["status"] == "complete"
        if row["arm"] in ("B1", "B3"):
            assert row["confirmatory_only"] is True
            assert "decision tree" in row["confirmatory_reason"]
        else:
            assert row["confirmatory_only"] is False
    # no formal-OOF status anywhere in the campaign bookkeeping
    assert "complete_oof" not in driver.manifest_path.read_text()
    assert "complete_oof" not in driver.decisions_path.read_text()


# ---------------------------------------------------------------------------
# 8. Confirm: 5 confirm + 4 B0 reference runs + frozen recipe
# ---------------------------------------------------------------------------


def _write_synthetic_ablation(driver) -> None:
    spec = {  # arm: (sre, rmse, alpha, sign)
        "B0": (0.20, 0.50, 0.50, 0.70),
        "B1": (0.05, 0.50, 1.00, 0.95),
        "B2": (0.08, 0.55, 0.95, 0.90),
        "B3": (0.06, 0.51, 1.02, 0.93),
    }
    rows = []
    for arm, (sre, rmse, alpha, sign) in spec.items():
        for seed in (17, 29, 43):
            rows.append({"arm": arm, "fold": 0, "seed": seed,
                         "inner_val_sre": sre,
                         "inner_val_rmse_over_zero": rmse,
                         "alpha_median": alpha,
                         "inner_val_sign_accuracy": sign,
                         "status": "complete"})
    driver.out_root.mkdir(parents=True, exist_ok=True)
    (driver.out_root / "ablation_v51.json").write_text(
        json.dumps({"schema_version": "pesn_v51_ablation_v1",
                    "rows": rows}), encoding="utf-8")


def test_confirm_enumerates_5_plus_4_runs(tmp_path, monkeypatch) -> None:
    driver = _prepared_driver(tmp_path, monkeypatch)
    _write_synthetic_ablation(driver)
    argvs: list = []
    monkeypatch.setattr(drv.V51SprintDriver, "_spawn", _fake_spawn(argvs))

    assert driver.run_stage("confirm") == "complete_confirm_v51"
    assert len(argvs) == 9
    _no_unseal(argvs)
    assert all(_argv_value(a, "--mode") == "v51" for a in argvs)
    assert all(_argv_value(a, "--seed") == "17" for a in argvs)
    assert all(_argv_value(a, "--updates") == "3000" for a in argvs)

    references = [a for a in argvs
                  if _argv_value(a, "--arm") == "B0"
                  and _argv_value(a, "--fold") in ("1", "2", "3", "4")]
    confirms = [a for a in argvs if a not in references]
    assert len(references) == 4
    assert len(confirms) == 5
    # frozen selection rule picks B1 (lowest SRE among eligible arms)
    assert all(_argv_value(a, "--arm") == "B1" for a in confirms)
    assert sorted(_argv_value(a, "--fold") for a in confirms) == \
        ["0", "1", "2", "3", "4"]
    # reference runs are marked in the decision log
    decisions = [json.loads(line) for line in
                 driver.decisions_path.read_text().splitlines()
                 if line.strip()]
    assert any(d.get("decision") == "reference_run"
               and d.get("reference") is True
               and d.get("fold") in (1, 2, 3, 4)
               and d.get("stage") == "confirm" for d in decisions)

    # frozen recipe record
    frozen = json.loads(
        (driver.out_root / "frozen_recipe_v51.json").read_text())
    assert frozen["arm"] == "B1"
    assert frozen["lambdas"] == {"lambda_cos": 0.02, "lambda_pair": 0.5,
                                 "lambda_pair_reg": 0.5, "lambda_mag": 0.5,
                                 "lambda_sign": 0.2, "lambda_dead": 0.1}
    assert frozen["ramp"] == {"start_update": 250, "ramp_updates": 250}
    assert frozen["inner_val_rule"]["seed"] == 20261019
    assert frozen["inner_val_rule"]["eval_every_updates"] == 50
    assert frozen["hashes"]["config_sha256"] is not None
    assert frozen["hashes"]["split_manifest_sha256"] is not None
    assert frozen["hashes"]["protocol_sha256"] is not None
    assert frozen["selection_exposed"] is True
    assert frozen["confirmation"]["no_fold_regressed"] is True
    assert frozen["confirmation"]["approved_for_v5_1b"] is True

    confirm = json.loads(
        (driver.out_root / "confirm_v51.json").read_text())
    assert confirm["recipe"] == "B1"
    assert confirm["n_confirm_runs"] == 5
    assert confirm["n_reference_runs"] == 4
    assert confirm["regressed_folds"] == []
    assert confirm["no_fold_regressed"] is True
    assert (driver.out_root / "confirm_v51.parquet").is_file()
    assert "complete_oof" not in driver.manifest_path.read_text()


def test_confirm_refuses_without_ablation(tmp_path, monkeypatch) -> None:
    driver = _prepared_driver(tmp_path, monkeypatch)
    argvs: list = []
    monkeypatch.setattr(drv.V51SprintDriver, "_spawn", _fake_spawn(argvs))
    with pytest.raises(drv.SprintRefused) as excinfo:
        driver.run_stage("confirm")
    assert excinfo.value.code == "ABLATION_MISSING"
    assert argvs == []


# ---------------------------------------------------------------------------
# 9. Sealed eval stays REFUSED (policy v5_1a_never_unseals)
# ---------------------------------------------------------------------------


def test_sealed_stays_refused_v51_policy() -> None:
    registration = {"campaign_id": "V5_1A_SPRINT",
                    "formal_oof": "none — V5.1a does no formal OOF"}
    decision = sealed_eval_request(
        registration, crossfit_complete_runs=0,
        model_hash=None, config_hash="a" * 64, protocol_hash="b" * 64,
        unseal=True, allow_unseal=False,
        sprint_policy="v5_1a_never_unseals")
    assert decision["status"] == "REFUSED"
    assert decision["code"] == "sealed_confirm_not_opened"
    assert any("v5_1a_never_unseals" in reason
               for reason in decision["reasons"])
    assert decision["registration"] == registration
    # never-allow path: even fully qualified inputs cannot unseal
    fully_qualified = sealed_eval_request(
        registration, crossfit_complete_runs=15,
        model_hash="a" * 64, config_hash="b" * 64, protocol_hash="c" * 64,
        unseal=True, allow_unseal=False,
        sprint_policy="v5_1a_never_unseals")
    assert fully_qualified["status"] == "REFUSED"


# ---------------------------------------------------------------------------
# 10. Report: development evidence labels, no complete_oof, sealed REFUSED
# ---------------------------------------------------------------------------


def test_report_labels_development_evidence(tmp_path, monkeypatch) -> None:
    driver = _prepared_driver(tmp_path, monkeypatch, unseal=True)
    argvs: list = []
    monkeypatch.setattr(drv.V51SprintDriver, "_spawn", _fake_spawn(argvs))

    assert driver.run_stage("report") == "complete_report_v51"
    _no_unseal(argvs)

    root = driver.report_root          # outputs/pesn/v5_1
    # sealed eval: REFUSED even though --unseal was requested
    sealed = json.loads(
        (driver.out_root / "sealed_eval.json").read_text())
    assert sealed["status"] == "REFUSED"
    assert sealed["allow_unseal"] is False
    assert sealed["unseal_requested"] is True
    assert sealed["sprint_policy"] == "v5_1a_never_unseals"
    assert sealed["crossfit_complete_runs"] == 0

    json_text = (root / "gate_report_v51.json").read_text()
    md_text = (root / "gate_report_v51.md").read_text()
    doc = json.loads(json_text)
    assert doc["campaign_id"] == "V5_1A_SPRINT"
    assert "development evidence" in json_text
    assert "development evidence" in md_text
    # formal gates are NOT claimed — deferred, labelled development-only
    assert doc["gates"]["G5.2"]["status"] == "not_executed"
    assert doc["gates"]["G5.3"]["status"] == "not_executed"
    assert "development evidence only" in doc["gates"]["G5.2"]["note"]
    assert doc["gates"]["G5.1"]["status"] == "measured"
    assert doc["gates"]["G5.1"]["verdict"] == "pass"
    assert doc["gates"]["sealed_eval"]["verdict"] == "REFUSED"
    assert doc["gates"]["M1"]["status"] == "not_executed"  # no diagnose here
    assert doc["extra"]["oof_claims"].startswith("none")

    # ABSENCE of any complete_oof status write, anywhere
    for text in (json_text, md_text,
                 driver.manifest_path.read_text(),
                 driver.decisions_path.read_text()):
        assert "complete_oof" not in text
    for stage_status in doc["stage_statuses"].values():
        assert not str(stage_status).startswith("complete_oof")

    # no subprocess argv ever carries --unseal
    for line in driver.decisions_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("argv"):
            assert "--unseal" not in row["argv"]

    # run_records merger + scale readiness
    assert (root / "run_records.jsonl").is_file()
    readiness = json.loads(
        (root / "scale_readiness_v51.json").read_text())
    assert readiness["expansion_allowed"] is False
    s0 = readiness["gates"]["S0_model_repair"]
    assert s0["status"] == "pending"
    assert s0["can_pass_in_v5_1a"] is False
    criteria = s0["criteria"]
    assert criteria["g51_green"]["met"] is True          # satisfied in prepare
    for key in ("g52_original_thresholds_formal_oof",
                "m1_amplitude_0p85_1p05_formal",
                "g53_original_thresholds_formal_oof",
                "s1_stability_2of3_seed_plus_ensemble",
                "g54_true_amplitude_formal"):
        assert criteria[key]["met"] is False
        assert criteria[key]["status"] == "pending"
    assert readiness["gates"]["S4_confirm_ood_one_shot_unseal"][
        "unsealed_in_v5_1a"] is False
    assert readiness["gates"]["S1_data_quality"]["status"] == "not_started"

    # stage statuses recorded, still no formal-OOF vocabulary
    manifest = json.loads(driver.manifest_path.read_text())
    assert manifest["stages"]["prepare"]["status"] == "complete_prepare_v51"
    assert manifest["stages"]["report"]["status"] == "complete_report_v51"
    assert manifest["status_vocabulary"] == list(drv.V51_STATUS_VALUES)
    assert manifest["campaign_id"] == "V5_1A_SPRINT"
    assert manifest["schema_version"] == "pesn_v5_1_sprint_manifest_v1"


def test_run_metrics_uses_inner_val_best_row_not_dev_final(tmp_path) -> None:
    """Leak guard: ablation/selection metrics must come from the INNER-VAL
    row at best_update, never the post-hoc dev_dense_final row (plan §2:
    outer fold must not select checkpoints or arms)."""
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    rows = [
        {"update": 50, "split": "inner_val", "wall_s": 10.0,
         "aggregate": {"rmse_over_zero": 0.99,
                       "sre_cluster_median": 0.40,
                       "sign_accuracy_cluster_median": 0.60,
                       "cos_eff_near_ts_median": 0.60,
                       "alpha_median": 0.30}},
        {"update": 100, "split": "inner_val", "wall_s": 20.0,
         "aggregate": {"rmse_over_zero": 0.80,
                       "sre_cluster_median": 0.25,
                       "sign_accuracy_cluster_median": 0.75,
                       "cos_eff_near_ts_median": 0.75,
                       "alpha_median": 0.60}},
        {"update": 100, "split": "dev_dense_best", "wall_s": 25.0,
         "aggregate": {"rmse_over_zero": 0.55,
                       "sre_cluster_median": 0.10,
                       "sign_accuracy_cluster_median": 0.95,
                       "cos_eff_near_ts_median": 0.90,
                       "alpha_median": 0.95}},
        {"update": 100, "split": "dev_dense_final", "wall_s": 30.0,
         "aggregate": {"rmse_over_zero": 0.50,
                       "sre_cluster_median": 0.08,
                       "sign_accuracy_cluster_median": 0.96,
                       "cos_eff_near_ts_median": 0.91,
                       "alpha_median": 0.99}},
    ]
    (run_dir / "eval.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "schema_version": "pesn_key_atom_bc_metrics_v1",
        "best_update": 100,
        "alpha_median": 0.60,
        "dead_zone_fraction": 0.12,
        "raw": rows[:2],
    }), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "complete", "best_update": 100}), encoding="utf-8")

    driver = drv.V51SprintDriver(config_path=CONFIG)
    got = driver._run_metrics(run_dir)
    assert got["rmse_over_zero"] == pytest.approx(0.80)
    assert got["sre"] == pytest.approx(0.25)
    assert got["sign_accuracy"] == pytest.approx(0.75)
    assert got["near_ts_cos"] == pytest.approx(0.75)
    assert got["metrics_source_split"] == "inner_val"
    assert got["metrics_source_update"] == 100
    # dev_dense rows would have given 0.50/0.08/0.96 — none may leak in
    assert got["rmse_over_zero"] != pytest.approx(0.50)
    assert got["sre"] != pytest.approx(0.08)


def test_run_metrics_legacy_series_without_split_tags(tmp_path) -> None:
    """V5-style runs (no split tags) keep the legacy last-row behavior."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    rows = [{"update": 250, "wall_s": 5.0,
             "aggregate": {"rmse_over_zero": 0.7, "sre_cluster_median": 0.2}}]
    (run_dir / "eval.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    driver = drv.V51SprintDriver(config_path=CONFIG)
    got = driver._run_metrics(run_dir)
    assert got["rmse_over_zero"] == pytest.approx(0.7)
    assert got["final_update"] == 250


def test_run_delivered_accepts_designed_patience_stop(tmp_path) -> None:
    """censored_patience + checkpoint_best.pt = delivered v51 protocol
    outcome (plan §3.4); missing best checkpoint or budget censoring is
    not delivered."""
    d = tmp_path / "run"
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "checkpoint_best.pt").write_bytes(b"x")
    assert drv._run_delivered({"status": "complete"}, d)
    assert drv._run_delivered({"status": "censored_patience"}, d)
    assert not drv._run_delivered({"status": "censored_patience"},
                                  tmp_path / "empty")
    (d / "checkpoints" / "checkpoint_best.pt").unlink()
    assert not drv._run_delivered({"status": "censored_patience"}, d)
    assert not drv._run_delivered({"status": "censored_budget"}, d)
    assert not drv._run_delivered(None, d)
