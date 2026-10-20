"""WP-L3 monitor tests (mission: PlanNotFrozen capture, lane W-LOSS).

Covers E0 residual #3: the supervisor monitor must catch
:class:`PlanNotFrozenError` and record ``blocked_with_reason`` on the
affected run record instead of crashing.  Regression tests:

* launch against a tampered/unpinned plan -> blocked report + annotation,
  never an exception out of ``attempt_launch``;
* launch against a legitimately PENDING (stage-e0) plan -> blocked with the
  PLAN_NOT_FROZEN reason, annotation recorded;
* a later successful transition clears the annotation;
* non-plan ledger errors (budget cap, illegal transition) still propagate
  (only plan-freeze conditions are monitor-captured);
* ``monitor_scan`` reports plan drift read-only;
* the CLI exits 3 (blocked) rather than crashing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agent_ops.ledger.run_ledger import (
    BudgetCapExceeded,
    RunLedger,
    sha256_file,
)
from scripts.agent_ops.runplan import build_r3_runplan as builder
from scripts.agent_ops.supervisor.monitor import attempt_launch, monitor_scan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "PES2TS_30H_R3_V52_V53"
EVIDENCE_SHA = "ab" * 32

FULL_SHAS = {
    "config_sha256": "e5" * 32,
    "data_sha256": "a1" * 32,
    "teacher_sha256": "b2" * 32,
    "features_sha256": "c3" * 32,
    "split_sha256": "d4" * 32,
}


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def make_record(**overrides):
    record = {
        "campaign_id": CAMPAIGN,
        "phase": "R3",
        "arm": "F0",
        "fold": 0,
        "seed": 17,
        "variant": "forward",
        "n_updates": 3000,
        "data_sha256": "a1" * 32,
        "teacher_sha256": "b2" * 32,
        "features_sha256": "c3" * 32,
        "split_sha256": "d4" * 32,
        "config_sha256": "e5" * 32,
        "checkpoint_update": None,
        "checkpoint_sha256": None,
        "status": "planned",
        "metrics_path": None,
        "gate_id": "G2-Move",
        "admissibility": "admissible",
        "evidence_sha": EVIDENCE_SHA,
        "attempt": 1,
        "parent_run_id": None,
    }
    record.update(overrides)
    return record


def read_lines(path: Path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture
def frozen_plan(tmp_path) -> Path:
    plan = tmp_path / "run_plan.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)
    return plan


@pytest.fixture
def pin(frozen_plan):
    return {"sha256": sha256_file(frozen_plan), "campaign_mode": "full"}


@pytest.fixture
def ledger(tmp_path, frozen_plan, pin) -> RunLedger:
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=frozen_plan, plan_pin=pin
    )
    led.init()
    return led


# ---------------------------------------------------------------------------
# attempt_launch: never crashes on plan-freeze conditions
# ---------------------------------------------------------------------------


def test_attempt_launch_tampered_plan_blocked_not_crash(ledger, frozen_plan, tmp_path):
    rec = make_record()
    ledger.append(rec)  # registration happens while the plan is pristine
    # tamper the plan AFTER registration
    frozen_plan.write_bytes(frozen_plan.read_bytes() + b"\n")

    report = attempt_launch(ledger, rec)  # must NOT raise
    assert report["state"] == "blocked"
    assert "PLAN_SHA_MISMATCH" in report["blocked_with_reason"]
    assert report["action"] is None

    # the affected run record carries blocked_with_reason (status unchanged)
    entry = ledger.status(report["run_id"])
    assert entry["status"] == "planned"
    assert "PLAN_SHA_MISMATCH" in entry["blocked_with_reason"]
    assert entry["blocked_source"] == "supervisor.monitor"
    assert entry["blocked_at"]
    # audit event recorded
    events = [json.loads(ln) for ln in read_lines(ledger.events_path)]
    assert events[-1]["event"] == "blocked"
    assert events[-1]["run_id"] == report["run_id"]


def test_attempt_launch_pending_plan_blocked(ledger, tmp_path):
    """Stage-e0 plan (PENDING shas): blocked with PLAN_NOT_FROZEN, no crash."""
    pending = tmp_path / "pending_plan.jsonl"
    builder.write_plan(pending)  # PENDING_E1/E2 tokens
    pin = {"sha256": sha256_file(pending), "campaign_mode": "full"}
    led = RunLedger(tmp_path / "r.jsonl", plan_path=pending, plan_pin=pin)
    led.init()
    rec = make_record()
    led.append(rec)

    report = attempt_launch(led, rec)
    assert report["state"] == "blocked"
    assert "PLAN_NOT_FROZEN" in report["blocked_with_reason"]
    entry = led.status(report["run_id"])
    assert entry["status"] == "planned"  # NOT launched, just annotated
    assert "PLAN_NOT_FROZEN" in entry["blocked_with_reason"]


def test_attempt_launch_unregistered_run_blocked_unregistered(ledger, frozen_plan):
    """A blocked launch of a never-registered run is reported, not crashed."""
    frozen_plan.write_bytes(frozen_plan.read_bytes() + b"\n")
    report = attempt_launch(ledger, make_record())
    assert report["state"] == "blocked_unregistered"
    assert "PLAN_SHA_MISMATCH" in report["blocked_with_reason"]


def test_attempt_launch_success_clears_annotation(ledger, frozen_plan):
    rec = make_record()
    ledger.append(rec)
    pristine = frozen_plan.read_bytes()
    frozen_plan.write_bytes(pristine + b"\n")
    report = attempt_launch(ledger, rec)
    assert report["state"] == "blocked"
    # restore the pristine plan bytes -> launch succeeds and clears the annotation
    frozen_plan.write_bytes(pristine)
    report = attempt_launch(ledger, rec)
    assert report["state"] == "running"
    assert report["action"] == "transitioned"
    entry = ledger.status(report["run_id"])
    assert entry["status"] == "running"
    assert "blocked_with_reason" not in entry
    assert "blocked_source" not in entry
    assert "blocked_at" not in entry


def test_attempt_launch_stage_g0_binding(ledger, tmp_path):
    """--stage g0: PENDING shas block; real shas launch."""
    pending = tmp_path / "pending_plan.jsonl"
    builder.write_plan(pending)
    pin = {"sha256": sha256_file(pending), "campaign_mode": "full"}
    led = RunLedger(tmp_path / "r.jsonl", plan_path=pending, plan_pin=pin)
    led.init()
    rec = make_record()
    led.append(rec)
    report = attempt_launch(led, rec, stage="g0")
    assert report["state"] == "blocked"
    assert "STAGE_G0_NOT_BOUND" in report["blocked_with_reason"]
    assert led.status(report["run_id"])["status"] == "planned"

    # the same check passes on a fully-bound plan
    rec2 = make_record(seed=29)
    ledger.append(rec2)  # register while the plan is pristine/frozen
    report2 = attempt_launch(ledger, rec2, stage="g0")
    assert report2["state"] == "running"


def test_non_plan_errors_still_propagate(ledger):
    """Only plan-freeze conditions are captured; other errors are NOT swallowed."""
    # budget-cap case: a second launch past cap=1 raises, not a blocked report
    led2 = RunLedger(
        ledger.path,
        plan_path=ledger.plan_path,
        plan_pin=ledger.plan_pin,
        caps={"R3": 1},
    )
    r1 = make_record(seed=17)
    r2 = make_record(seed=29)
    led2.append(r1)
    assert attempt_launch(led2, r1)["state"] == "running"
    led2.append(r2)
    with pytest.raises(BudgetCapExceeded):
        attempt_launch(led2, r2)  # cap 1 exceeded -> propagates


# ---------------------------------------------------------------------------
# monitor_scan: read-only drift visibility
# ---------------------------------------------------------------------------


def test_monitor_scan_reports_freeze_and_blocks(ledger, frozen_plan):
    rec = make_record()
    ledger.append(rec)
    scan = monitor_scan(ledger)
    assert scan["plan_frozen"] is True
    assert scan["plan_error"] is None
    assert scan["run_count"] == 1
    assert scan["runs"][0]["status"] == "planned"
    assert scan["runs"][0]["blocked_with_reason"] is None

    frozen_plan.write_bytes(frozen_plan.read_bytes() + b"\n")
    attempt_launch(ledger, rec)  # annotate via the captured path
    scan2 = monitor_scan(ledger)
    assert scan2["plan_frozen"] is False
    assert "PLAN_SHA_MISMATCH" in scan2["plan_error"]
    assert "PLAN_SHA_MISMATCH" in scan2["runs"][0]["blocked_with_reason"]
    # scan is read-only: no new ledger lines beyond append+blocked
    assert len(read_lines(ledger.path)) == 2


# ---------------------------------------------------------------------------
# mark_blocked primitives
# ---------------------------------------------------------------------------


def test_mark_blocked_idempotent_and_validation(ledger):
    from scripts.agent_ops.ledger.run_ledger import LedgerError, RunNotFound

    rec = make_record()
    outcome = ledger.append(rec)
    run_id = outcome.entry["run_id"]
    blocked = ledger.mark_blocked(run_id, "reason A")
    assert blocked.action == "blocked"
    assert blocked.entry["blocked_with_reason"] == "reason A"
    # idempotent: same reason -> noop
    assert ledger.mark_blocked(run_id, "reason A").action == "noop"
    # changed reason -> new annotation
    assert ledger.mark_blocked(run_id, "reason B").entry["blocked_with_reason"] == "reason B"
    # invalid reasons / unknown runs refuse
    with pytest.raises(LedgerError) as excinfo:
        ledger.mark_blocked(run_id, "  ")
    assert "INVALID_BLOCK_REASON" in str(excinfo.value)
    with pytest.raises(RunNotFound) as excinfo:
        ledger.mark_blocked("no-such-run", "x")
    assert "RUN_NOT_FOUND" in str(excinfo.value)
    # status is untouched by annotation
    assert ledger.status(run_id)["status"] == "planned"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_monitor_cli_blocked_exit_and_scan(tmp_path, frozen_plan, pin):
    ledger_path = tmp_path / "r.jsonl"
    subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.ledger.run_ledger",
         "--ledger", str(ledger_path), "init"],
        cwd=str(PROJECT_ROOT), env=_env(), check=True, capture_output=True,
    )
    rec = make_record()
    subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.ledger.run_ledger",
         "--ledger", str(ledger_path), "append", "--record-json", json.dumps(rec)],
        cwd=str(PROJECT_ROOT), env=_env(), check=True, capture_output=True,
    )
    frozen_plan.write_bytes(frozen_plan.read_bytes() + b"\n")
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps(pin), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.agent_ops.supervisor.monitor",
            "--ledger", str(ledger_path), "--plan", str(frozen_plan),
            "--plan-pin-file", str(pin_file),
            "launch", "--record-json", json.dumps(rec),
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 3, proc.stderr  # blocked, NOT a crash (exit 1/2/traceback)
    report = json.loads(proc.stdout)
    assert report["state"] == "blocked"
    assert "PLAN_SHA_MISMATCH" in report["blocked_with_reason"]

    scan_proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.agent_ops.supervisor.monitor",
            "--ledger", str(ledger_path), "--plan", str(frozen_plan),
            "--plan-pin-file", str(pin_file), "scan",
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert scan_proc.returncode == 0, scan_proc.stderr
    scan = json.loads(scan_proc.stdout)
    assert scan["plan_frozen"] is False
    assert "PLAN_SHA_MISMATCH" in scan["plan_error"]
