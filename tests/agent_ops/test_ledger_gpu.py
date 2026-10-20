"""Supervisor core tests: run ledger (idempotency/audit) + GPU mutex.

Covers the 30h campaign guard rails: no duplicate run_id resubmission,
no placeholder hashes in evidence records, legal status transitions with a
retry budget, a consistent events.jsonl audit trail, and single-holder GPU
locking with stale-holder reporting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest

from scripts.agent_ops.gpu_queue import gpu_lock
from scripts.agent_ops.ledger import run_ledger
from scripts.agent_ops.ledger.run_ledger import (
    IllegalTransition,
    PlaceholderHashError,
    RetryCapExceeded,
    RunIdConflict,
    RunLedger,
    compute_run_id,
    validate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Larger than any possible Linux pid (pid_max <= 2^22) but a valid pid_t:
# guaranteed "dead" without touching a real process.
DEAD_PID = 999_999_999

CAMPAIGN = "PES2TS_30H_R3_V52_V53"
RUN_ID = "{}:V5.2:QT:f0:s17".format(CAMPAIGN)


def make_record(**overrides):
    """A fully valid section 7.2 run_record with real (fake) hashes."""
    record = {
        "campaign_id": CAMPAIGN,
        "phase": "V5.2",
        "arm": "QT",
        "fold": 0,
        "seed": 17,
        "n_updates": 1900,
        "data_sha256": "a1" * 32,
        "teacher_sha256": "b2" * 32,
        "features_sha256": "c3" * 32,
        "split_sha256": "d4" * 32,
        "config_sha256": "e5" * 32,
        "checkpoint_update": None,
        "checkpoint_sha256": None,
        "status": "planned",
        "metrics_path": None,
    }
    record.update(overrides)
    return record


def done_overrides():
    return {
        "status": "done",
        "checkpoint_update": 1900,
        "checkpoint_sha256": "ab" * 32,
        "metrics_path": "outputs/pesn/30h_campaign/metrics.json",
    }


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture
def ledger(tmp_path) -> RunLedger:
    led = RunLedger(tmp_path / "run_records.jsonl")
    led.init()
    return led


# ---------------------------------------------------------------------------
# ledger: idempotency / conflicts / validation
# ---------------------------------------------------------------------------


def test_idempotent_duplicate_append(ledger):
    first = ledger.append(make_record())
    assert first.action == "created"
    assert first.entry["run_id"] == RUN_ID == compute_run_id(make_record())

    second = ledger.append(make_record())
    assert second.action == "noop"
    assert second.entry["run_id"] == first.entry["run_id"]

    # volatile metadata drift must still be a no-op
    third = ledger.append(make_record(updated_at="1999-01-01T00:00:00+00:00"))
    assert third.action == "noop"

    assert len(read_lines(ledger.path)) == 1
    assert len(read_lines(ledger.events_path)) == 1
    assert ledger.status(RUN_ID)["status"] == "planned"


def test_run_id_conflict(ledger):
    ledger.append(make_record())
    with pytest.raises(RunIdConflict) as excinfo:
        ledger.append(make_record(config_sha256="ff" * 32))
    assert "RUN_ID_CONFLICT" in str(excinfo.value)

    # same identity + same status but drifted non-identity content
    with pytest.raises(RunIdConflict) as excinfo:
        ledger.append(make_record(metrics_path="runs/metrics.json"))
    assert "RUN_ID_CONFLICT" in str(excinfo.value)

    # rejected appends must not touch the ledger
    assert len(read_lines(ledger.path)) == 1
    assert len(read_lines(ledger.events_path)) == 1


def test_placeholder_hash_rejected(tmp_path):
    led = RunLedger(tmp_path / "run_records.jsonl")
    led.init()

    with pytest.raises(PlaceholderHashError) as excinfo:
        led.append(make_record(data_sha256="REQUIRED_AT_RUNTIME"))
    assert "PLACEHOLDER_HASH" in str(excinfo.value)

    missing = make_record()
    del missing["features_sha256"]
    with pytest.raises(PlaceholderHashError) as excinfo:
        led.append(missing)
    assert "MISSING_FIELD" in str(excinfo.value)

    with pytest.raises(PlaceholderHashError):
        led.append(make_record(checkpoint_sha256="REQUIRED_AT_RUNTIME"))

    # standalone validate() behaves the same
    with pytest.raises(PlaceholderHashError):
        validate(make_record(split_sha256="REQUIRED_AT_RUNTIME"))

    # nothing was written for any rejected record
    assert read_lines(led.path) == []
    assert read_lines(led.events_path) == []


def test_illegal_transition_rejected(ledger):
    ledger.append(make_record())
    with pytest.raises(IllegalTransition) as excinfo:
        ledger.append(make_record(**done_overrides()))
    assert "ILLEGAL_TRANSITION" in str(excinfo.value)

    ledger.append(make_record(status="skipped"))  # planned -> skipped is legal
    with pytest.raises(IllegalTransition):
        ledger.append(make_record(status="running"))  # skipped is terminal

    assert len(read_lines(ledger.path)) == 2
    assert len(read_lines(ledger.events_path)) == 2


def test_retry_cap_enforced(ledger):
    def start():
        return ledger.append(make_record(status="running", pid=os.getpid()))

    def fail():
        return ledger.append(make_record(status="failed"))

    ledger.append(make_record())
    start()
    fail()
    assert start().entry["retry_count"] == 1
    fail()
    assert start().entry["retry_count"] == 2
    fail()

    before = len(read_lines(ledger.path))
    with pytest.raises(RetryCapExceeded) as excinfo:
        start()
    assert "RETRY_CAP_EXCEEDED" in str(excinfo.value)
    assert len(read_lines(ledger.path)) == before  # rejected attempt wrote nothing
    assert len(read_lines(ledger.events_path)) == before

    # 3rd retry attempt requires supervisor_override=true
    outcome = ledger.append(
        make_record(status="running", pid=os.getpid(), supervisor_override=True)
    )
    assert outcome.action == "transitioned"
    assert outcome.entry["retry_count"] == 3
    assert outcome.entry["supervisor_override"] is True


def test_events_jsonl_audit_consistent(ledger):
    ledger.append(make_record())
    ledger.append(make_record())  # idempotent no-op: no line, no event
    ledger.append(make_record(status="running", pid=os.getpid()))
    ledger.append(make_record(status="failed"))
    ledger.append(make_record(status="running", pid=os.getpid()))
    ledger.append(make_record(**done_overrides()))

    ledger_lines = read_lines(ledger.path)
    events = [json.loads(ln) for ln in read_lines(ledger.events_path)]

    assert len(ledger_lines) == len(events) == 5
    run_ids = {json.loads(ln)["run_id"] for ln in ledger_lines}
    assert run_ids == {RUN_ID}
    assert {ev["run_id"] for ev in events} == run_ids
    assert [ev["to_status"] for ev in events] == [
        "planned",
        "running",
        "failed",
        "running",
        "done",
    ]
    assert [ev["event"] for ev in events] == [
        "append",
        "transition",
        "transition",
        "transition",
        "transition",
    ]
    assert all(ev["ts"] for ev in events)
    assert ledger.status(RUN_ID)["status"] == "done"


def test_reconcile_marks_dead_pid_interrupted(ledger):
    with pytest.raises(ProcessLookupError):
        os.kill(DEAD_PID, 0)  # prove the fake pid is genuinely dead
    assert run_ledger.pid_alive(DEAD_PID) is False
    assert run_ledger.pid_alive(os.getpid()) is True

    ledger.append(make_record(pid=DEAD_PID))
    ledger.append(make_record(status="running", pid=DEAD_PID))
    ledger.append(make_record(fold=1, pid=os.getpid()))
    ledger.append(make_record(fold=1, status="running", pid=os.getpid()))
    ledger.append(make_record(fold=2))
    ledger.append(make_record(fold=2, status="running"))

    summary = ledger.reconcile()
    assert summary["interrupted"] == [RUN_ID]
    assert summary["alive"] == ["{}:f1:s17".format(RUN_ID.rsplit(":f0", 1)[0])]
    assert summary["unknown"] == ["{}:f2:s17".format(RUN_ID.rsplit(":f0", 1)[0])]

    entry = ledger.status(RUN_ID)
    assert entry["status"] == "interrupted"
    assert entry["interrupted_reason"] == "dead_pid:{}".format(DEAD_PID)
    assert entry["retry_count"] == 0  # planned -> running is the initial attempt

    # reconcile never deletes: 6 registrations + 1 interrupted event
    assert len(read_lines(ledger.path)) == 7
    assert len(read_lines(ledger.events_path)) == 7
    assert ledger.status(RUN_ID)["campaign_id"] == CAMPAIGN

    # second reconcile is a no-op (interrupted is not running)
    again = ledger.reconcile()
    assert again["interrupted"] == []
    assert len(read_lines(ledger.path)) == 7


# ---------------------------------------------------------------------------
# gpu_lock
# ---------------------------------------------------------------------------


CHILD_HOLDER = """
import os, sys, time
from scripts.agent_ops.gpu_queue.gpu_lock import acquire

lock, ready, flag = sys.argv[1], sys.argv[2], sys.argv[3]
handle = acquire(lock, blocking=False, lane="child", run_id="R-child")
if handle is None:
    sys.exit(3)
with open(ready, "w") as fh:
    fh.write("1")
deadline = time.time() + 20
while not os.path.exists(flag) and time.time() < deadline:
    time.sleep(0.05)
handle.release()
sys.exit(0)
"""


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(module: str, args: List[str]) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", module] + args,
        cwd=str(PROJECT_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_gpu_mutual_exclusion_two_processes(tmp_path):
    lock = tmp_path / "pes2ts_gpu.lock"
    ready = tmp_path / "child_ready"
    flag = tmp_path / "child_release"
    script = tmp_path / "child_holder.py"
    script.write_text(CHILD_HOLDER, encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script), str(lock), str(ready), str(flag)],
        cwd=str(PROJECT_ROOT),
        env=_env(),
    )
    try:
        deadline = time.time() + 30
        while not ready.exists() and time.time() < deadline and proc.poll() is None:
            time.sleep(0.02)
        assert ready.exists(), "child process never acquired the lock"

        # second process must be refused while the child holds the lock
        assert gpu_lock.acquire(lock, blocking=False, lane="parent") is None
        status = gpu_lock.probe_status(lock)
        assert status["state"] == "HELD"
        assert status["stale"] is False
        assert status["holder"]["pid"] == proc.pid
        assert status["holder"]["lane"] == "child"
        holder = gpu_lock.read_holder(lock)
        for key in ("pid", "lane", "run_id", "acquired_at", "cmdline"):
            assert key in holder

        # and the CLI reports the same thing
        payload = json.loads(
            _run_cli(
                "scripts.agent_ops.gpu_queue.gpu_lock",
                ["--status", "--lock", str(lock)],
            )
        )
        assert payload["state"] == "HELD"
        assert payload["holder"]["pid"] == proc.pid

        flag.write_text("go", encoding="utf-8")
        assert proc.wait(timeout=30) == 0
        proc = None

        # same-process re-entry is refused too (independent open file description)
        handle = gpu_lock.acquire(lock, blocking=False, lane="parent", run_id="R-parent")
        assert handle is not None
        try:
            assert gpu_lock.acquire(lock, blocking=False, lane="parent") is None
            assert handle.holder["lane"] == "parent"
            assert handle.holder["run_id"] == "R-parent"
            assert handle.holder["cmdline"]
            assert handle.holder["acquired_at"]
            assert gpu_lock.probe_status(lock)["state"] == "HELD"
        finally:
            handle.release()

        final = gpu_lock.probe_status(lock)
        assert final["state"] == "FREE"
        assert final["stale"] is False
        assert not gpu_lock.holder_path(lock).exists()
    finally:
        if proc is not None:
            if proc.poll() is None:
                flag.write_text("go", encoding="utf-8")
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


def test_gpu_stale_detection_fake_dead_pid(tmp_path):
    lock = tmp_path / "pes2ts_gpu.lock"
    assert gpu_lock.pid_alive(DEAD_PID) is False
    with pytest.raises(ProcessLookupError):
        os.kill(DEAD_PID, 0)  # fake dead pid, no real process involved

    gpu_lock.holder_path(lock).write_text(
        json.dumps(
            {
                "pid": DEAD_PID,
                "lane": "child",
                "run_id": "R-stale",
                "acquired_at": 0.0,
                "cmdline": "python train.py",
                "lock_path": str(lock),
            }
        ),
        encoding="utf-8",
    )

    status = gpu_lock.probe_status(lock)
    assert status["state"] == "STALE"
    assert status["stale"] is True
    assert status["holder"]["pid"] == DEAD_PID
    assert "NOT auto-released" in status["detail"]

    # stale detection never auto-releases / never deletes
    assert gpu_lock.holder_path(lock).exists()

    payload = json.loads(
        _run_cli("scripts.agent_ops.gpu_queue.gpu_lock", ["--status", "--lock", str(lock)])
    )
    assert payload["state"] == "STALE"
    assert payload["holder"]["pid"] == DEAD_PID
    assert gpu_lock.holder_path(lock).exists()


def test_gpu_status_cli_free_and_held(tmp_path):
    lock = tmp_path / "pes2ts_gpu.lock"

    def status_cli() -> dict:
        out = _run_cli(
            "scripts.agent_ops.gpu_queue.gpu_lock", ["--status", "--lock", str(lock)]
        )
        return json.loads(out)

    assert status_cli()["state"] == "FREE"

    handle = gpu_lock.acquire(lock, blocking=False, lane="parent", run_id="R-cli")
    assert handle is not None
    try:
        payload = status_cli()
        assert payload["state"] == "HELD"
        assert payload["holder"]["pid"] == os.getpid()
        assert payload["holder"]["run_id"] == "R-cli"
        assert payload["lock_path"] == str(lock)
    finally:
        handle.release()

    assert status_cli()["state"] == "FREE"


# ---------------------------------------------------------------------------
# ledger CLI
# ---------------------------------------------------------------------------


def _ledger_cli(ledger_path: Path, args: List[str], expect: int = 0) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.ledger.run_ledger", "--ledger", str(ledger_path)]
        + args,
        cwd=str(PROJECT_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == expect, proc.stderr
    return proc.stdout


def test_ledger_cli_roundtrip(tmp_path):
    ledger_path = tmp_path / "run_records.jsonl"
    init_payload = json.loads(_ledger_cli(ledger_path, ["init"]))
    assert init_payload["ledger"] == str(ledger_path)

    record_json = json.dumps(make_record())
    created = json.loads(
        _ledger_cli(ledger_path, ["append", "--record-json", record_json])
    )
    assert created["action"] == "created"
    assert created["entry"]["run_id"] == RUN_ID

    again = json.loads(_ledger_cli(ledger_path, ["append", "--record-json", record_json]))
    assert again["action"] == "noop"

    conflict = _ledger_cli(
        ledger_path,
        ["append", "--record-json", json.dumps(make_record(config_sha256="ff" * 32))],
        expect=2,
    )
    assert conflict == ""

    entry = json.loads(_ledger_cli(ledger_path, ["status", "--run-id", RUN_ID]))
    assert entry["status"] == "planned"

    listing = json.loads(_ledger_cli(ledger_path, ["list", "--status", "planned"]))
    assert [e["run_id"] for e in listing] == [RUN_ID]

    summary = json.loads(_ledger_cli(ledger_path, ["reconcile"]))
    assert summary == {"interrupted": [], "alive": [], "unknown": []}

    assert len(read_lines(ledger_path)) == 1
    assert len(read_lines(ledger_path.parent / "events.jsonl")) == 1
