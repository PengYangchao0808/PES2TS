"""Campaign deadline/time-control tests (mission: test_deadline.py, lane W-LOSS).

Covers ``scripts/agent_ops/supervisor/deadline.py``: absolute window pinning
into ``task_state.json`` via temp+rename with all other fields preserved,
``check()`` reporting overdue phases as ``phase:blocked`` plus
``remaining = hard_stop - now``, and ``guard_dispatch`` refusing new dispatch
when the remaining budget is below the task ETA -- every missing/invalid/
corrupt input refuses (no warn-and-pass paths).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.agent_ops.supervisor import deadline as dl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

START = "2026-10-20T09:00:00+00:00"
HARD_STOP = "2026-10-21T15:00:00+00:00"  # 30h campaign window
NOW = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)  # 27h remaining


def make_state() -> dict:
    return {
        "campaign_id": "PES2TS_30H_R3_V52_V53",
        "mode": "parallel_execution_v0.1",
        "budget": {"max_parallel_lanes": 3, "gpu_jobs": 1},
        "time_control": {
            "campaign_started_at_utc": None,
            "hard_stop_at_utc": None,
            "rule": "E0签认时写死绝对起止",
        },
        "phases": [
            {"id": "E0", "title": "freeze", "status": "planned"},  # no deadline_utc
            {"id": "E1", "status": "planned", "deadline_utc": "2026-10-20T11:00:00+00:00"},
            {"id": "E2", "status": "planned", "deadline_utc": "2026-10-21T00:00:00+00:00"},
        ],
    }


@pytest.fixture
def state_path(tmp_path) -> Path:
    path = tmp_path / "task_state.json"
    path.write_text(json.dumps(make_state(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _cli(state: Path, *args: str):
    return subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.supervisor.deadline", "--state", str(state)]
        + list(args),
        cwd=str(PROJECT_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# set_campaign_window
# ---------------------------------------------------------------------------


def test_set_window_writes_and_preserves_other_fields(state_path):
    control = dl.set_campaign_window(START, HARD_STOP, state_path)
    assert control["campaign_started_at_utc"] == START
    assert control["hard_stop_at_utc"] == HARD_STOP
    assert control["rule"] == "E0签认时写死绝对起止"  # other time_control keys kept

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["campaign_id"] == "PES2TS_30H_R3_V52_V53"  # other top-level fields kept
    assert state["budget"]["gpu_jobs"] == 1
    assert state["phases"][0]["id"] == "E0"
    assert state["time_control"]["campaign_started_at_utc"] == START
    assert state["time_control"]["hard_stop_at_utc"] == HARD_STOP


def test_set_window_accepts_z_suffix_and_naive_utc(state_path):
    dl.set_campaign_window("2026-10-20T09:00:00Z", "2026-10-21T15:00:00Z", state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stop = datetime.fromisoformat(
        state["time_control"]["hard_stop_at_utc"].replace("Z", "+00:00")
    )
    assert stop == datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)


def test_set_window_rejects_inverted_window(state_path):
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.set_campaign_window(HARD_STOP, START, state_path)
    assert "INVALID_WINDOW" in str(excinfo.value)
    with pytest.raises(dl.TimeControlError):
        dl.set_campaign_window(START, START, state_path)


def test_set_window_rejects_garbage_input(state_path):
    with pytest.raises(dl.TimeControlError):
        dl.set_campaign_window("not-a-date", HARD_STOP, state_path)
    with pytest.raises(dl.TimeControlError):
        dl.set_campaign_window(START, "", state_path)
    with pytest.raises(dl.TimeControlError):
        dl.set_campaign_window(None, HARD_STOP, state_path)


def test_set_window_missing_or_corrupt_state_refused(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.set_campaign_window(START, HARD_STOP, missing)
    assert "TASK_STATE_MISSING" in str(excinfo.value)

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{oops", encoding="utf-8")
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.set_campaign_window(START, HARD_STOP, corrupt)
    assert "TASK_STATE_CORRUPT" in str(excinfo.value)

    not_object = tmp_path / "list.json"
    not_object.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.set_campaign_window(START, HARD_STOP, not_object)
    assert "TASK_STATE_CORRUPT" in str(excinfo.value)


def test_set_window_rejects_non_object_time_control(state_path):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["time_control"] = "bogus"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.set_campaign_window(START, HARD_STOP, state_path)
    assert "time_control" in str(excinfo.value)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_reports_overdue_phase_as_blocked(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    report = dl.check(state_path, now=NOW)

    assert report["blocked_phases"] == ["E1"]  # deadline 11:00 < now 12:00
    by_id = {p["id"]: p for p in report["phases"]}
    assert by_id["E1"]["report"] == "phase:blocked"
    assert by_id["E2"]["report"] == "phase:ok"
    assert "E0" not in by_id  # phase without deadline_utc is not deadline-governed
    assert report["remaining_h"] == 27.0  # hard_stop - now
    assert report["campaign_report"] == "campaign:ok"
    assert report["hard_stop_at_utc"] == HARD_STOP


def test_check_after_hard_stop_reports_campaign(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    after = datetime.fromisoformat(HARD_STOP) + timedelta(minutes=1)
    report = dl.check(state_path, now=after)
    assert report["remaining_h"] < 0
    assert report["campaign_report"] == "campaign:hard_stop_reached"
    assert set(report["blocked_phases"]) == {"E1", "E2"}


def test_check_requires_pinned_window(tmp_path, state_path):
    # window never pinned -> refuse (fail-closed, not remaining=inf)
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.check(state_path, now=NOW)
    assert "TIME_CONTROL_UNPINNED" in str(excinfo.value)

    # time_control object entirely absent -> refuse
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["time_control"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.check(state_path, now=NOW)
    assert "TIME_CONTROL_MISSING" in str(excinfo.value)

    # inverted window -> refuse
    state = make_state()
    state["time_control"] = {
        "campaign_started_at_utc": HARD_STOP,
        "hard_stop_at_utc": START,
    }
    bad = tmp_path / "inverted.json"
    bad.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.check(bad, now=NOW)
    assert "TIME_CONTROL_INVALID" in str(excinfo.value)

    # missing / corrupt files -> refuse
    with pytest.raises(dl.TimeControlError):
        dl.check(tmp_path / "nope.json", now=NOW)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    with pytest.raises(dl.TimeControlError) as excinfo:
        dl.check(corrupt, now=NOW)
    assert "TASK_STATE_CORRUPT" in str(excinfo.value)


def test_check_rejects_bad_phase_deadline(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phases"][1]["deadline_utc"] = "tomorrow-ish"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(dl.TimeControlError):
        dl.check(state_path, now=NOW)

    state["phases"][1]["deadline_utc"] = 123
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(dl.TimeControlError):
        dl.check(state_path, now=NOW)


# ---------------------------------------------------------------------------
# guard_dispatch
# ---------------------------------------------------------------------------


def test_guard_dispatch_allows_when_eta_fits(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    report = dl.guard_dispatch(5, state_path, now=NOW)  # 5h task, 27h left
    assert report["remaining_h"] == 27.0


def test_guard_dispatch_refuses_when_eta_exceeds_remaining(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    with pytest.raises(dl.DispatchRefused) as excinfo:
        dl.guard_dispatch(28, state_path, now=NOW)
    assert "DISPATCH_REFUSED" in str(excinfo.value)
    assert "27" in str(excinfo.value) and "28" in str(excinfo.value)
    # eta exactly equal to remaining is still allowed (strict <)
    dl.guard_dispatch(27, state_path, now=NOW)


def test_guard_dispatch_refuses_after_hard_stop(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    after = datetime.fromisoformat(HARD_STOP)
    with pytest.raises(dl.DispatchRefused) as excinfo:
        dl.guard_dispatch(0.0, state_path, now=after)
    assert "hard stop reached" in str(excinfo.value)


def test_guard_dispatch_fails_closed_without_window(state_path):
    with pytest.raises(dl.TimeControlError):  # DispatchRefused base class
        dl.guard_dispatch(1, state_path, now=NOW)


def test_guard_dispatch_rejects_invalid_eta(state_path):
    dl.set_campaign_window(START, HARD_STOP, state_path)
    for bad in (-1, "5", None, True):
        with pytest.raises(dl.TimeControlError) as excinfo:
            dl.guard_dispatch(bad, state_path, now=NOW)
        assert "INVALID_ETA" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------


def test_cli_roundtrip(state_path):
    proc = _cli(state_path, "set", "--start", START, "--hard-stop", HARD_STOP)
    assert proc.returncode == 0, proc.stderr

    proc = _cli(state_path, "check", "--now", NOW.isoformat())
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["blocked_phases"] == ["E1"]

    proc = _cli(state_path, "guard-dispatch", "--eta-h", "5", "--now", NOW.isoformat())
    assert proc.returncode == 0, proc.stderr

    proc = _cli(state_path, "guard-dispatch", "--eta-h", "28", "--now", NOW.isoformat())
    assert proc.returncode == 1  # refused -> explicit refusal exit code
    assert "DISPATCH_REFUSED" in proc.stderr


def test_cli_fails_closed_without_window(state_path):
    proc = _cli(state_path, "check")
    assert proc.returncode == 2  # unpinned window is an error, never a pass
    assert "TIME_CONTROL_UNPINNED" in proc.stderr
