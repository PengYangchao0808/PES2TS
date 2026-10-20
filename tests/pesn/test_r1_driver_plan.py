"""V5.1a-R1 driver planner gates (planning logic only — NO training).

Hermetic coverage of ``scripts/campaigns/run_v5_1_r1.py``:

* stage plans: baseline_fill (7 reused A0 cells + the exact 8 missing
  B0 cells with the ORIGINAL v5_1_sprint.yaml), dev (A2/A3 × folds 0–2),
  confirm (--arm × folds 3–4), screen (no run cells);
* HARD refusals: folds/seeds/arms outside the stage plan, any mode other
  than v51 (formal OOF), unknown plan keys, ``--unseal``/``--fold``
  rejected by argparse (nonzero exit), confirm without ``--arm``;
* the delivered predicate (status ∈ {complete, censored_patience,
  censored_screen} AND checkpoint_best.pt exists);
* the budget guard (no new launch inside the final 5 minutes / past the
  limit) with an injected clock;
* --dry-run prints the plan and writes NOTHING;
* the summary-row reader over a metrics.json/status.json fixture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.campaigns import run_v5_1_r1 as drv  # noqa: E402

R1_CONFIG = PROJECT_ROOT / "config/campaigns/v5_1_r1.yaml"
SPRINT_CONFIG = PROJECT_ROOT / "config/campaigns/v5_1_sprint.yaml"


def _paths(tmp_path: Path) -> dict:
    return {
        "r1_config_path": R1_CONFIG,
        "sprint_config_path": SPRINT_CONFIG,
        "runs_root": tmp_path / "runs",
        "legacy_runs_root": tmp_path / "legacy_runs",
    }


def _plan(stage: str, tmp_path: Path, **kw):
    p = _paths(tmp_path)
    return drv.plan_stage(
        stage,
        plan=kw.pop("plan", None),
        arm=kw.pop("arm", None),
        r1_config_path=p["r1_config_path"],
        sprint_config_path=p["sprint_config_path"],
        runs_root=p["runs_root"],
        legacy_runs_root=p["legacy_runs_root"],
        **kw)


# ---------------------------------------------------------------------------
# Stage plans
# ---------------------------------------------------------------------------

def test_plan_baseline_fill_full_a0_matrix(tmp_path):
    cells = _plan("baseline_fill", tmp_path)
    # 5 folds × 3 seeds = 15 A0 cells: 7 reused + 8 trained
    assert len(cells) == 15
    reuse = [c for c in cells if c.reuse_from is not None]
    fresh = [c for c in cells if c.reuse_from is None]
    assert len(reuse) == 7 and len(fresh) == 8
    assert {(c.fold, c.seed) for c in reuse} == set(drv.A0_REUSE_CELLS)
    assert {(c.fold, c.seed) for c in fresh} == set(drv.BASELINE_FILL_CELLS)
    for c in cells:
        assert c.arm == "B0" and c.mode == "v51" and c.updates == 3000
        assert c.config_path == SPRINT_CONFIG
        assert c.run_id == f"B0_f{c.fold}_s{c.seed}_v51"
        assert c.out_dir.name == c.run_id
    # reuse cells point at the LEGACY v5_1a run dirs; fresh cells at R1
    assert all(c.out_dir == c.reuse_from for c in reuse)
    assert all(str(c.out_dir).startswith(str(tmp_path / "runs"))
               for c in fresh)
    assert all(str(c.out_dir).startswith(str(tmp_path / "legacy_runs"))
               for c in reuse)


def test_plan_baseline_fill_uses_v5_1_sprint_config_sha(tmp_path):
    """A0 cells must train with the ORIGINAL v5_1_sprint.yaml so every
    A0 cell shares the frozen effective config sha (1d8101a4...)."""
    cells = _plan("baseline_fill", tmp_path)
    import hashlib
    h = hashlib.sha256(SPRINT_CONFIG.read_bytes()).hexdigest()
    assert h.startswith("1d8101a4")
    assert all(c.config_path == SPRINT_CONFIG for c in cells)


def test_plan_dev_matrix(tmp_path):
    cells = _plan("dev", tmp_path)
    assert len(cells) == 18  # 2 arms × 3 folds × 3 seeds
    assert {c.arm for c in cells} == {"A2", "A3"}
    assert {c.fold for c in cells} == {0, 1, 2}
    assert {c.seed for c in cells} == {17, 29, 43}
    for c in cells:
        assert c.mode == "v51" and c.updates == 3000
        assert c.config_path == R1_CONFIG
        assert c.reuse_from is None
        assert c.out_dir == tmp_path / "runs" / c.run_id
    run_ids = [c.run_id for c in cells]
    assert len(set(run_ids)) == 18
    assert "A2_f0_s17_v51" in run_ids and "A3_f2_s43_v51" in run_ids


def test_plan_confirm_requires_arm_and_covers_folds_3_4(tmp_path):
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("confirm", tmp_path)
    assert excinfo.value.code == "ARM_REQUIRED"
    cells = _plan("confirm", tmp_path, arm="A4")
    assert len(cells) == 6  # 1 arm × 2 folds × 3 seeds
    assert {c.arm for c in cells} == {"A4"}
    assert {c.fold for c in cells} == {3, 4}
    assert {c.seed for c in cells} == {17, 29, 43}
    assert all(c.config_path == R1_CONFIG for c in cells)
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("confirm", tmp_path, arm="A0")
    assert excinfo.value.code == "ARM_NOT_IN_PLAN"


def test_plan_screen_has_no_cells(tmp_path):
    assert _plan("screen", tmp_path) == []


def test_plan_updates_and_subsets_via_plan_json(tmp_path):
    cells = _plan("dev", tmp_path,
                  plan={"folds": [0], "seeds": [17], "arms": ["A2"],
                        "updates": 60})
    assert len(cells) == 1
    (cell,) = cells
    assert (cell.arm, cell.fold, cell.seed, cell.updates) == (
        "A2", 0, 17, 60)


# ---------------------------------------------------------------------------
# HARD refusals
# ---------------------------------------------------------------------------

def test_plan_refuses_fold_outside_stage_plan(tmp_path):
    # dev may never widen to folds 3/4 (confirm folds)
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("dev", tmp_path, plan={"folds": [4]})
    assert excinfo.value.code == "FOLD_NOT_IN_PLAN"
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("confirm", tmp_path, arm="A2", plan={"folds": [0]})
    assert excinfo.value.code == "FOLD_NOT_IN_PLAN"
    # baseline_fill never invents new folds
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("baseline_fill", tmp_path, plan={"folds": [5]})
    assert excinfo.value.code == "FOLD_NOT_IN_PLAN"


def test_plan_refuses_seed_and_arm_outside_plan(tmp_path):
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("dev", tmp_path, plan={"seeds": [999]})
    assert excinfo.value.code == "SEED_NOT_IN_PLAN"
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("dev", tmp_path, plan={"arms": ["B1"]})
    assert excinfo.value.code == "ARM_NOT_IN_PLAN"
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("baseline_fill", tmp_path, plan={"arms": ["A2"]})
    assert excinfo.value.code == "ARM_NOT_IN_PLAN"


def test_plan_refuses_formal_oof_mode(tmp_path):
    for bad in ("formal", "oof", "screen"):
        with pytest.raises(drv.DriverRefused) as excinfo:
            _plan("dev", tmp_path, plan={"mode": bad})
        assert excinfo.value.code == "FORMAL_OOF_REFUSED"
        assert "formal OOF" in str(excinfo.value)


def test_plan_refuses_unknown_keys_and_bad_values(tmp_path):
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("dev", tmp_path, plan={"foldz": [0]})
    assert excinfo.value.code == "PLAN_KEY_UNKNOWN"
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("dev", tmp_path, plan={"updates": 0})
    assert excinfo.value.code == "PLAN_INVALID"
    # r1 lambda overrides only apply to the r1 training stages
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("baseline_fill", tmp_path, plan={"lambda_strong": 0.1})
    assert excinfo.value.code == "PLAN_KEY_NOT_APPLICABLE"
    # screen takes no run-cell plan at all
    with pytest.raises(drv.DriverRefused) as excinfo:
        _plan("screen", tmp_path, plan={"folds": [0]})
    assert excinfo.value.code == "PLAN_KEY_NOT_APPLICABLE"
    # but a positive lambda override on dev/confirm is admissible
    cells = _plan("dev", tmp_path,
                  plan={"lambda_strong": 0.3, "lambda_dead2": 0.03})
    assert len(cells) == 18


def test_cli_lambda_override_refused_outside_training_stages(tmp_path):
    """--lambda-* overrides are a dev/confirm config-patch mechanism —
    refused (typed) on baseline_fill/screen even via the CLI."""
    driver = drv.R1QueueDriver(
        sprint_config_path=SPRINT_CONFIG, r1_config_path=R1_CONFIG,
        root=tmp_path / "root", runs_root=tmp_path / "runs",
        legacy_runs_root=tmp_path / "legacy",
        dry_run=True, arm=None, plan=None,
        lambda_strong=0.1, lambda_dead2=None)
    with pytest.raises(drv.DriverRefused) as excinfo:
        driver.run_stage("baseline_fill")
    assert excinfo.value.code == "PLAN_KEY_NOT_APPLICABLE"
    # dev/confirm accept the override (dry-run prints the plan)
    driver_dev = drv.R1QueueDriver(
        sprint_config_path=SPRINT_CONFIG, r1_config_path=R1_CONFIG,
        root=tmp_path / "root", runs_root=tmp_path / "runs",
        legacy_runs_root=tmp_path / "legacy",
        dry_run=True, plan={"folds": [0], "seeds": [17], "arms": ["A2"]},
        lambda_strong=0.1, lambda_dead2=0.3)
    assert driver_dev.run_stage("dev") == drv.EXIT_OK


def test_cli_has_no_unseal_and_no_fold(tmp_path):
    parser = drv.build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--stage", "dev", "--unseal"])
    assert excinfo.value.code != 0
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--stage", "dev", "--fold", "4"])
    assert excinfo.value.code != 0
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--stage", "oof"])  # no formal/OOF stage either
    assert excinfo.value.code != 0
    # and no unseal/fold attributes sneak into the parser
    args = parser.parse_args(["--stage", "confirm", "--arm", "A2"])
    assert not hasattr(args, "unseal") and not hasattr(args, "fold")


# ---------------------------------------------------------------------------
# Delivered predicate + budget guard
# ---------------------------------------------------------------------------

def _mk_run(root: Path, *, status: str | None, best: bool) -> Path:
    run = root / "run"
    if status is not None:
        (run).mkdir(parents=True, exist_ok=True)
        (run / "status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8")
    if best:
        (run / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run / "checkpoints" / "checkpoint_best.pt").write_bytes(b"pt")
    return run


def test_run_delivered_semantics(tmp_path):
    rd = drv.run_delivered
    assert rd(None, None) is False
    assert rd({"status": "complete"}, None) is True
    for good in ("complete", "censored_patience", "censored_screen"):
        run = _mk_run(tmp_path / good, status=good, best=True)
        assert rd({"status": good}, run) is True
        run2 = _mk_run(tmp_path / f"{good}_nobest", status=good, best=False)
        assert rd({"status": good}, run2) is False, good
    for bad in ("running", "aborted_nan", "aborted_data", "missing_status"):
        run = _mk_run(tmp_path / bad, status=bad, best=True)
        assert rd({"status": bad}, run) is False, bad
    assert rd({"status": "complete"}, tmp_path / "nonexistent") is False


def test_budget_guard_gate():
    clock = {"t": 1000.0}
    guard = drv.BudgetGuard(
        1000.0, limit_minutes=240.0, clock=lambda: clock["t"])
    assert guard.may_launch() and not guard.expired()
    clock["t"] = 1000.0 + 236.0 * 60  # inside the final guard window
    assert not guard.may_launch() and not guard.expired()
    clock["t"] = 1000.0 + 240.0 * 60
    assert not guard.may_launch() and guard.expired()
    # guard window is configurable
    guard2 = drv.BudgetGuard(
        1000.0, limit_minutes=10.0, launch_guard_minutes=0.0,
        clock=lambda: clock["t"])
    clock["t"] = 1000.0
    assert guard2.may_launch()
    clock["t"] = 1000.0 + 601.0
    assert not guard2.may_launch() and guard2.expired()


# ---------------------------------------------------------------------------
# Dry-run: prints the plan, writes NOTHING
# ---------------------------------------------------------------------------

def test_cli_dry_run_prints_plan_writes_nothing(tmp_path, capsys):
    root = tmp_path / "r1"
    rc = drv.main([
        "--stage", "dev", "--dry-run",
        "--root", str(root),
        "--runs-root", str(tmp_path / "runs"),
        "--legacy-runs-root", str(tmp_path / "legacy")])
    assert rc == drv.EXIT_OK
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["stage"] == "dev" and doc["dry_run"] is True
    assert len(doc["plan"]) == 18
    assert not root.exists()          # dry-run wrote NOTHING
    assert not (tmp_path / "runs").exists()
    # confirm dry-run too (arm via --arm)
    rc = drv.main([
        "--stage", "confirm", "--arm", "A3", "--dry-run",
        "--root", str(root), "--runs-root", str(tmp_path / "runs")])
    assert rc == drv.EXIT_OK
    doc = json.loads(capsys.readouterr().out)
    assert len(doc["plan"]) == 6 and doc["plan"][0]["arm"] == "A3"
    assert not root.exists()


def test_cli_plan_file_refusals(tmp_path, capsys):
    bad = tmp_path / "plan.json"
    bad.write_text(json.dumps({"folds": [7]}), encoding="utf-8")
    rc = drv.main([
        "--stage", "dev", "--dry-run", "--plan", str(bad),
        "--root", str(tmp_path / "r1")])
    assert rc == drv.EXIT_REFUSED
    assert "FOLD_NOT_IN_PLAN" in capsys.readouterr().err
    missing = tmp_path / "nope.json"
    rc = drv.main([
        "--stage", "dev", "--dry-run", "--plan", str(missing),
        "--root", str(tmp_path / "r1")])
    assert rc == drv.EXIT_REFUSED
    assert "PLAN_NOT_FOUND" in capsys.readouterr().err
    # stage routing: unknown stage rejected by argparse (exit != 0)
    with pytest.raises(SystemExit) as excinfo:
        drv.build_arg_parser().parse_args(["--stage", "report"])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# Summary-row reader (metrics.json contract)
# ---------------------------------------------------------------------------

def test_summary_row_from_metrics(tmp_path):
    run = tmp_path / "A2_f0_s17_v51"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps(
        {"status": "censored_patience"}), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({
        "best_update": 1450,
        "inner_val_score": {"rmse_over_zero": 0.71,
                            "sre_median": 0.24,
                            "near_ts_cos": 0.88},
        "sign_accuracy": 0.91,
        "alpha_median": 0.62,
        "dead_zone_fraction": 0.14,
    }), encoding="utf-8")
    cell = drv.RunCell(
        stage="dev", run_id="A2_f0_s17_v51", arm="A2", fold=0, seed=17,
        mode="v51", updates=3000, config_path=R1_CONFIG, out_dir=run)
    row = drv.R1QueueDriver._summary_row(cell, run)
    assert row["run_id"] == "A2_f0_s17_v51"
    assert row["status"] == "censored_patience"
    assert row["best_update"] == 1450
    assert row["inner_rmse_over_zero"] == repr(0.71)
    assert row["inner_sre_median"] == repr(0.24)
    assert row["inner_near_ts_cos"] == repr(0.88)
    assert row["inner_sign_accuracy"] == repr(0.91)
    assert row["alpha_median"] == repr(0.62)
    assert row["dead_zone_fraction"] == repr(0.14)
    assert tuple(row) == drv.R1QueueDriver.SUMMARY_FIELDS
    # missing metrics degrade to "" (never fabricated)
    empty = tmp_path / "missing"
    empty.mkdir()
    row2 = drv.R1QueueDriver._summary_row(cell, empty)
    assert row2["status"] == "missing"
    assert row2["inner_rmse_over_zero"] == ""
    assert row2["best_update"] == ""
