"""Fail-closed gate guard tests (mission: test_gate_guard.py, lane W-LOSS).

Covers ``scripts/agent_ops/gpu_queue/gate_guard.py`` (assert_runnable refusals
for missing/invalid/corrupt gate input and protocol sha pinning) and the
``gpu_lock.acquire(..., run_meta=..., gate_id=...)`` integration: a formal
acquisition without a passing gate raises ``GateNotPassedError`` *before* the
lock is touched; legacy fixture acquisitions keep working.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.agent_ops.gpu_queue import gate_guard, gpu_lock
from scripts.agent_ops.gpu_queue.gate_guard import GateNotPassedError

CAMPAIGN = "PES2TS_30H_R3_V52_V53"


@pytest.fixture
def campaign(tmp_path) -> Path:
    """Minimal campaign root: frozen protocol file + gates directory."""
    root = tmp_path / "campaign"
    protocol = root / "outputs" / "pes2ts" / "30h_campaign" / "protocol_30h.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({"protocol": "R3-v3.2", "campaign": CAMPAIGN}), encoding="utf-8")
    (root / "docs" / "agent_runs" / "state" / "gates").mkdir(parents=True)
    return root


def protocol_sha(root: Path) -> str:
    data = (root / "outputs" / "pes2ts" / "30h_campaign" / "protocol_30h.json").read_bytes()
    return hashlib.sha256(data).hexdigest()


def write_gate(root: Path, gate_name: str, **overrides) -> Path:
    record = {
        "gate_id": gate_name,
        "status": "pass",
        "protocol_sha256": protocol_sha(root),
        "protocol_file": "outputs/pes2ts/30h_campaign/protocol_30h.json",
        "evidence": ["docs/agent_runs/fixtures/example.json"],
    }
    record.update(overrides)
    path = root / "docs" / "agent_runs" / "state" / "gates" / "{}.json".format(gate_name)
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# assert_runnable: the happy path
# ---------------------------------------------------------------------------


def test_assert_runnable_pass_returns_verified_record(campaign):
    write_gate(campaign, "G2-Move")
    record = gate_guard.assert_runnable("G2-Move", campaign)
    assert record["status"] == "pass"
    assert record["gate_id"] == "G2-Move"
    assert record["protocol_sha256_verified"] == protocol_sha(campaign)
    assert record["protocol_file_resolved"].endswith("protocol_30h.json")


# ---------------------------------------------------------------------------
# every refusal path (fail-closed: no warn-and-continue anywhere)
# ---------------------------------------------------------------------------


def test_missing_gate_file_refused(campaign):
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "GATE_FILE_MISSING" in str(excinfo.value)


@pytest.mark.parametrize("status", ["fail", "blocked", "partial", "not_run", "PASS", None])
def test_non_pass_status_refused(campaign, status):
    write_gate(campaign, "G1-Feature", status=status)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G1-Feature", campaign)
    assert "GATE_NOT_PASSED" in str(excinfo.value)


def test_corrupt_gate_file_refused(campaign):
    path = campaign / "docs" / "agent_runs" / "state" / "gates" / "G3-V52Supervision.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G3-V52Supervision", campaign)
    assert "GATE_FILE_CORRUPT" in str(excinfo.value)

    path.write_text("[]", encoding="utf-8")  # valid JSON, wrong shape
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G3-V52Supervision", campaign)
    assert "GATE_FILE_CORRUPT" in str(excinfo.value)


def test_protocol_sha_mismatch_refused(campaign):
    write_gate(campaign, "G4-Rollout", protocol_sha256="0" * 64)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G4-Rollout", campaign)
    assert "PROTOCOL_SHA_MISMATCH" in str(excinfo.value)


def test_protocol_sha_drift_after_gate_pass_refused(campaign):
    """Protocol edited after the gate passed -> pin no longer matches -> refuse."""
    gate_path = write_gate(campaign, "G0-Data")
    protocol = campaign / "outputs" / "pes2ts" / "30h_campaign" / "protocol_30h.json"
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n//tampered", encoding="utf-8")
    assert gate_path.exists()
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_SHA_MISMATCH" in str(excinfo.value)


def test_missing_protocol_file_refused(campaign):
    write_gate(campaign, "G2-Move")
    (campaign / "outputs" / "pes2ts" / "30h_campaign" / "protocol_30h.json").unlink()
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G2-Move", campaign)
    assert "PROTOCOL_FILE_MISSING" in str(excinfo.value)


@pytest.mark.parametrize("bad_sha", [None, "", "zz" * 32, "abc"])
def test_missing_or_malformed_protocol_sha_refused(campaign, bad_sha):
    overrides = {"protocol_sha256": bad_sha} if bad_sha is not None else {}
    if bad_sha is None:
        write_gate(campaign, "G0-Data")
        path = campaign / "docs" / "agent_runs" / "state" / "gates" / "G0-Data.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["protocol_sha256"]
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        write_gate(campaign, "G0-Data", **overrides)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_SHA_MISSING" in str(excinfo.value)


@pytest.mark.parametrize("bad_id", [None, "", "  ", "docs/gates/G0", "a\\b", "..", "."])
def test_invalid_gate_id_refused(campaign, bad_id):
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable(bad_id, campaign)
    assert "INVALID_GATE_ID" in str(excinfo.value)


def test_missing_campaign_root_refused(tmp_path):
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", tmp_path / "nope")
    assert "CAMPAIGN_ROOT_MISSING" in str(excinfo.value)


def test_gate_id_mismatch_refused(campaign):
    write_gate(campaign, "G2-Move", gate_id="G3-Label")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G2-Move", campaign)
    assert "GATE_ID_MISMATCH" in str(excinfo.value)


def test_protocol_file_escape_refused(campaign):
    write_gate(campaign, "G0-Data", protocol_file="../../../etc/passwd")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_FILE_INVALID" in str(excinfo.value)


# ---------------------------------------------------------------------------
# gpu_lock.acquire integration
# ---------------------------------------------------------------------------


def test_formal_acquire_with_passing_gate(campaign, tmp_path):
    write_gate(campaign, "G2-Move")
    lock = tmp_path / "gpu.lock"
    handle = gpu_lock.acquire(
        lock,
        blocking=False,
        lane="W-LOSS",
        run_id="R-run",
        run_meta={"formal": True, "campaign_id": CAMPAIGN},
        gate_id="G2-Move",
        campaign_root=campaign,
    )
    assert handle is not None
    try:
        assert handle.holder["gate_id"] == "G2-Move"
        assert handle.holder["gate_checked"] is True
        assert handle.holder["gate_status"] == "pass"
        assert handle.holder["protocol_sha256"] == protocol_sha(campaign)
        assert gpu_lock.probe_status(lock)["state"] == "HELD"
    finally:
        handle.release()
    assert gpu_lock.probe_status(lock)["state"] == "FREE"


def test_formal_acquire_failing_gate_never_touches_lock(campaign, tmp_path):
    write_gate(campaign, "G2-Move", status="blocked")
    lock = tmp_path / "gpu.lock"
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(lock, lane="W-LOSS", gate_id="G2-Move", campaign_root=campaign)
    assert "GATE_NOT_PASSED" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()
    assert gpu_lock.probe_status(lock)["state"] == "FREE"


def test_formal_run_meta_without_gate_id_refused(campaign, tmp_path):
    lock = tmp_path / "gpu.lock"
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(
            lock, run_meta={"formal": True, "run_id": "R-x"}, campaign_root=campaign
        )
    assert "GATE_REQUIRED" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()


def test_gate_id_inside_run_meta_is_enforced(campaign, tmp_path):
    lock = tmp_path / "gpu.lock"
    # no gate file at all -> refuse even though the caller passed no gate_id arg
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(
            lock, run_meta={"gate_id": "G0-Data", "campaign_root": str(campaign)}
        )
    assert "GATE_FILE_MISSING" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()


def test_legacy_fixture_acquire_stays_backward_compatible(campaign, tmp_path):
    """No run_meta / no gate_id -> old behaviour (fixture-level CUDA work)."""
    lock = tmp_path / "gpu.lock"
    handle = gpu_lock.acquire(lock, blocking=False, lane="fixture", run_id="R-fixture")
    assert handle is not None
    try:
        assert handle.holder["gate_checked"] is False
        assert handle.holder["gate_id"] is None
        # second same-process attempt still refused (independent open file desc)
        assert gpu_lock.acquire(lock, blocking=False, lane="fixture") is None
    finally:
        handle.release()
    assert gpu_lock.probe_status(lock)["state"] == "FREE"
