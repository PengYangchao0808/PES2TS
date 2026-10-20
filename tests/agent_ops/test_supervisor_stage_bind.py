"""WP-L3 stage-g0 binding tests (mission: five-sha launch gate, lane W-LOSS).

Covers E0 residual #5: the supervisor-only ``--stage g0`` verify path requires
ALL five identity shas (config/data/teacher/features/split) to be real 64-hex
digests on the plan lines before any launch.  Fixtures only -- no real E1 SHAs
are bound here (they land after E1; the supervisor re-verifies then).

Semantics under test (fail-closed):
* stage e0 plan (PENDING tokens) -> refuse at g0/launch;
* all-real-sha plan -> pass at g0 and launch;
* zero sha / malformed digest -> refuse at ANY stage;
* stage e0 itself refuses to gate a launch (it is not a launch stage);
* unknown run_id / missing plan / empty plan -> refuse;
* CLI exit codes 0/1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agent_ops.runplan import build_r3_runplan as builder
from scripts.agent_ops.supervisor.stage_bind import (
    StageBindingError,
    assert_stage_binding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.supervisor.stage_bind"] + args,
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )


def read_plan_lines(path: Path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def rewrite(path: Path, lines):
    path.write_text(
        "".join(json.dumps(ln, sort_keys=True, ensure_ascii=False) + "\n" for ln in lines),
        encoding="utf-8",
    )


@pytest.fixture
def bound_plan(tmp_path) -> Path:
    plan = tmp_path / "bound_plan.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)  # all five real (fixture) shas
    return plan


@pytest.fixture
def e0_plan(tmp_path) -> Path:
    plan = tmp_path / "e0_plan.jsonl"
    builder.write_plan(plan)  # PENDING_E1/PENDING_E2 tokens
    return plan


# ---------------------------------------------------------------------------
# positive
# ---------------------------------------------------------------------------


def test_all_real_shas_pass_g0_and_launch(bound_plan):
    for stage in ("g0", "launch"):
        summary = assert_stage_binding(bound_plan, stage=stage)
        assert summary["bound"] is True
        assert summary["stage"] == stage
        assert summary["checked_runs"] == 99
        assert summary["sha_fields"] == list(builder.SHA_FIELDS)


def test_single_run_check(bound_plan):
    lines = read_plan_lines(bound_plan)
    run_id = lines[1]["run_id"]
    summary = assert_stage_binding(bound_plan, stage="g0", run_id=run_id)
    assert summary["checked_runs"] == 1


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


def test_e0_pending_plan_refused_at_g0(e0_plan):
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(e0_plan, stage="g0")
    msg = str(excinfo.value)
    assert "STAGE_G0_NOT_BOUND" in msg
    assert "PENDING_E1" in msg or "PENDING_E2" in msg


def test_stage_e0_is_not_a_launch_stage(bound_plan):
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(bound_plan, stage="e0")
    assert "STAGE_NOT_LAUNCHABLE" in str(excinfo.value)


def test_zero_sha_refused(tmp_path):
    """An all-zeros sha is never valid at any stage."""
    plan = tmp_path / "zero.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)
    lines = read_plan_lines(plan)
    lines[1]["data_sha256"] = "0" * 64
    rewrite(plan, lines)
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(plan, stage="g0")
    assert "STAGE_G0_NOT_BOUND" in str(excinfo.value)
    assert "zero" in str(excinfo.value)


def test_malformed_sha_refused(tmp_path):
    plan = tmp_path / "bad.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)
    lines = read_plan_lines(plan)
    lines[2]["teacher_sha256"] = "not-a-sha"
    rewrite(plan, lines)
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(plan, stage="launch")
    assert "STAGE_LAUNCH_NOT_BOUND" in str(excinfo.value)


def test_unknown_run_id_and_missing_plan_refused(tmp_path, bound_plan):
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(bound_plan, stage="g0", run_id="deadbeefdeadbeef")
    assert "STAGE_BINDING_RUN_NOT_PLANNED" in str(excinfo.value)

    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(tmp_path / "absent.jsonl", stage="g0")
    assert "STAGE_BINDING_PLAN_MISSING" in str(excinfo.value)

    empty = tmp_path / "empty.jsonl"
    empty.write_text(
        json.dumps(
            {
                "record_type": "plan_header",
                "campaign_id": builder.CAMPAIGN_ID,
                "campaign_mode": "full",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(empty, stage="g0")
    assert "STAGE_BINDING_PLAN_EMPTY" in str(excinfo.value)


def test_corrupt_plan_refused(tmp_path):
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(StageBindingError) as excinfo:
        assert_stage_binding(corrupt, stage="g0")
    assert "STAGE_BINDING_PLAN_UNUSABLE" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_exit_codes(bound_plan, e0_plan, tmp_path):
    proc = _run_cli(["--plan", str(bound_plan), "--stage", "g0"])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["bound"] is True

    proc = _run_cli(["--plan", str(e0_plan), "--stage", "g0"])
    assert proc.returncode == 1
    assert "STAGE_G0_NOT_BOUND" in proc.stderr

    lines = read_plan_lines(bound_plan)
    proc = _run_cli(
        ["--plan", str(bound_plan), "--stage", "g0", "--run-id", lines[1]["run_id"]]
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["checked_runs"] == 1

    proc = _run_cli(["--plan", str(tmp_path / "absent.jsonl")])
    assert proc.returncode == 1
