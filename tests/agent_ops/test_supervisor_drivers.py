"""WP-L3 driver campaign-root tests (mission: r3 entrypoints, lane W-LOSS).

Covers E0 residual #4: the R3 driver entrypoints
(``scripts/campaigns/run_v5_1_r3.py`` / ``scripts/campaigns/r3_launch.py``)
default their output root to ``outputs/pesn/30h_campaign/**`` and
assert-refuse the sealed prefixes (``outputs/pesn/v5_1/``,
``outputs/pesn/v5_1_r1/``) plus anything outside the campaign root --
fail-closed at startup, before any output file can be opened.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agent_ops.supervisor.campaign_root import CampaignRootError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, str(PROJECT_ROOT / rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_v5_1_r3 = _load("run_v5_1_r3_drv", "scripts/campaigns/run_v5_1_r3.py")
r3_launch = _load("r3_launch_drv", "scripts/campaigns/r3_launch.py")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ---------------------------------------------------------------------------
# run_v5_1_r3: default root + accept/reject
# ---------------------------------------------------------------------------


def test_default_output_root_is_campaign():
    assert run_v5_1_r3.DEFAULT_OUTPUT_ROOT == "outputs/pesn/30h_campaign/r3_runs"
    assert run_v5_1_r3.resolve_output_root(None) == "outputs/pesn/30h_campaign/r3_runs"


def test_explicit_campaign_roots_accepted():
    for root in (
        "outputs/pesn/30h_campaign",
        "outputs/pesn/30h_campaign/r3_runs",
        "outputs/pesn/30h_campaign/runs/exp1",
        "./outputs/pesn/30h_campaign/r3_runs",
        "/mnt/data/x/outputs/pesn/30h_campaign/r3_runs",
    ):
        assert run_v5_1_r3.resolve_output_root(root).startswith(
            "outputs/pesn/30h_campaign"
        )


@pytest.mark.parametrize(
    "sealed",
    [
        "outputs/pesn/v5_1",
        "outputs/pesn/v5_1/",
        "outputs/pesn/v5_1/runs/B0_f0_s17_v51",
        "outputs/pesn/v5_1_r1",
        "outputs/pesn/v5_1_r1/checkpoints",
    ],
)
def test_sealed_roots_refused(sealed):
    with pytest.raises(CampaignRootError) as excinfo:
        run_v5_1_r3.resolve_output_root(sealed)
    assert "SEALED_ROOT_REFUSED" in str(excinfo.value)


@pytest.mark.parametrize(
    "outside", ["outputs/pesn/other_campaign", "outputs/pesn", "data/tables"],
)
def test_outside_campaign_roots_refused(outside):
    with pytest.raises(CampaignRootError) as excinfo:
        run_v5_1_r3.resolve_output_root(outside)
    assert "OUTSIDE_CAMPAIGN_ROOT" in str(excinfo.value)


def test_invalid_roots_refused():
    for garbage in (123, "", "   ", "outputs/pesn/30h_campaign/../v5_1"):
        with pytest.raises(CampaignRootError):
            run_v5_1_r3.resolve_output_root(garbage)


def test_cli_exit_codes():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/campaigns/run_v5_1_r3.py"), "--check"],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["output_root"] == "outputs/pesn/30h_campaign/r3_runs"

    proc = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts/campaigns/run_v5_1_r3.py"),
            "--output-root", "outputs/pesn/v5_1_r1", "--check",
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "SEALED_ROOT_REFUSED" in proc.stderr


# ---------------------------------------------------------------------------
# r3_launch: ledger/plan paths must live in the campaign
# ---------------------------------------------------------------------------


def test_r3_launch_refuses_sealed_ledger(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts/campaigns/r3_launch.py"),
            "--ledger", "outputs/pesn/v5_1/run_records.jsonl",
            "--record-json", "{}",
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "SEALED_ROOT_REFUSED" in proc.stderr


def test_r3_launch_refuses_outside_plan(tmp_path):
    outside_plan = tmp_path / "plan.jsonl"
    outside_plan.write_text("{}", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts/campaigns/r3_launch.py"),
            "--plan", str(outside_plan),
            "--record-json", "{}",
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "plan outside campaign" in proc.stderr


def test_r3_launch_requires_exactly_one_record(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts/campaigns/r3_launch.py"),
            "--ledger", "outputs/pesn/30h_campaign/run_records.jsonl",
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert "--record" in proc.stderr


def test_r3_launch_blocked_exit_code(tmp_path):
    """A non-pinned (fixture) plan -> blocked report, exit 3 (never a crash)."""
    from scripts.agent_ops.ledger.run_ledger import sha256_file
    from scripts.agent_ops.runplan import build_r3_runplan as builder

    plan = tmp_path / "outputs/pesn/30h_campaign/run_plan.jsonl"
    ledger = tmp_path / "outputs/pesn/30h_campaign/run_records.jsonl"
    builder.write_plan(plan)  # stage-e0 plan: NOT the pinned bytes

    record = {
        "campaign_id": "PES2TS_30H_R3_V52_V53",
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
        "evidence_sha": "ab" * 32,
        "attempt": 1,
        "parent_run_id": None,
    }
    assert sha256_file(plan) != (  # proves the fixture plan is not the pinned one
        "d2a63599727804e9ff407b30c7e5e1bdfc555c38fa7cafd5857a6cf273073163"
    )
    # register the planned run first (through the ledger CLI, unpinned fixture)
    subprocess.run(
        [
            sys.executable, "-m", "scripts.agent_ops.ledger.run_ledger",
            "--ledger", str(ledger), "init",
        ],
        cwd=str(PROJECT_ROOT), env=_env(), check=True, capture_output=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "scripts.agent_ops.ledger.run_ledger",
            "--ledger", str(ledger), "append", "--record-json", json.dumps(record),
        ],
        cwd=str(PROJECT_ROOT), env=_env(), check=True, capture_output=True,
    )
    proc = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts/campaigns/r3_launch.py"),
            "--ledger", str(ledger),
            "--plan", str(plan),
            "--record-json", json.dumps(record),
        ],
        cwd=str(PROJECT_ROOT), env=_env(), capture_output=True, text=True, timeout=60,
    )
    # r3_launch pins to the E0-frozen campaign pin: a non-pinned plan is a
    # blocked outcome (exit 3) with PLAN_SHA_MISMATCH -- never a traceback
    assert proc.returncode == 3, proc.stderr
    report = json.loads(proc.stdout)
    assert report["state"] == "blocked"
    assert "PLAN_SHA_MISMATCH" in report["blocked_with_reason"]
    assert report["ledger_root"] == "outputs/pesn/30h_campaign"
