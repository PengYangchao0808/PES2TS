"""Fail-closed gate guard tests (mission: test_gate_guard.py, lane W-LOSS).

Covers ``scripts/agent_ops/gpu_queue/gate_guard.py``: campaign-yaml protocol
pinning, upstream G0->G1->G2->G3a->G3->G4 chain verification, task_state
formal authorization (ORA-2 blocker 4), and the
``gpu_lock.acquire(..., run_meta=..., gate_id=..., cuda=...)`` integration:
formal acquisitions without a passing chain/authorization refuse *before* the
lock is touched; the nvidia-smi precheck refuses formal runs on probe
failure/external occupancy (blocker 5) but only warns for fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.agent_ops.gpu_queue import gate_guard, gpu_lock
from scripts.agent_ops.gpu_queue.gate_guard import GateNotPassedError, GATE_ORDER

CAMPAIGN = "PES2TS_30H_R3_V52_V53"


@pytest.fixture
def campaign(tmp_path) -> Path:
    """Minimal campaign root: campaign yaml + task_state + gates directory."""
    root = tmp_path / "campaign"
    yaml_path = root / "config" / "campaigns" / "pes2ts_30h_20261020.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("schema: pes2ts_30h_campaign_v1\ncampaign_id: {}\n".format(CAMPAIGN), encoding="utf-8")
    write_task_state(root, authorized=False)
    (root / "docs" / "agent_runs" / "state" / "gates").mkdir(parents=True)
    return root


def write_task_state(root: Path, authorized: bool = False, raw: str = None) -> Path:
    path = root / "docs" / "agent_runs" / "state" / "task_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    payload = {
        "campaign_id": CAMPAIGN,
        "authorization": {
            "formal_training_authorized": authorized,
            "sealed_authorized": False,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def yaml_sha(root: Path) -> str:
    data = (root / "config" / "campaigns" / "pes2ts_30h_20261020.yaml").read_bytes()
    return hashlib.sha256(data).hexdigest()


def write_gate(root: Path, gate_name: str, **overrides) -> Path:
    record = {
        "gate_id": gate_name,
        "status": "pass",
        "protocol_sha256": yaml_sha(root),
        "evidence": ["docs/agent_runs/fixtures/example.json"],
    }
    record.update(overrides)
    path = root / "docs" / "agent_runs" / "state" / "gates" / "{}.json".format(gate_name)
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return path


def write_chain(root: Path, upto: str, **target_overrides) -> None:
    """Write every gate G0..upto as pass; the target gets ``target_overrides``."""
    for gate in GATE_ORDER[: GATE_ORDER.index(upto) + 1]:
        overrides = target_overrides if gate == upto else {}
        write_gate(root, gate, **overrides)


@pytest.fixture
def gpu_ok(monkeypatch):
    """Deterministic CUDA precheck: probe always clean."""
    monkeypatch.setattr(
        gpu_lock, "precheck_gpu", lambda *a, **k: {"ok": True, "external_memory_mib": 0, "error": None, "reason": None}
    )


# ---------------------------------------------------------------------------
# assert_runnable: the happy path + chain + authorization
# ---------------------------------------------------------------------------


def test_assert_runnable_pass_returns_verified_record(campaign):
    write_chain(campaign, "G2-Move")
    record = gate_guard.assert_runnable("G2-Move", campaign)
    assert record["status"] == "pass"
    assert record["gate_id"] == "G2-Move"
    assert record["protocol_sha256_verified"] == yaml_sha(campaign)
    assert record["upstream_chain_verified"] == ["G0-Data", "G1-Feature", "G2-Move"]
    assert record["formal_authorization_checked"] is False


def test_assert_runnable_formal_requires_authorization(campaign):
    write_chain(campaign, "G3-V52Supervision")
    # flags are false until user sign-off -> formal MUST refuse (ORA-2/4b)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G3-V52Supervision", campaign, formal=True)
    assert "AUTHORIZATION_REQUIRED" in str(excinfo.value)

    write_task_state(campaign, authorized=True)  # user signed off
    record = gate_guard.assert_runnable("G3-V52Supervision", campaign, formal=True)
    assert record["formal_authorization_checked"] is True


def test_formal_authorization_missing_or_corrupt_refused(campaign):
    write_chain(campaign, "G0-Data")
    state = campaign / "docs" / "agent_runs" / "state" / "task_state.json"
    state.unlink()
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign, formal=True)
    assert "AUTHORIZATION_MISSING" in str(excinfo.value)

    write_task_state(campaign, raw="{broken json")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign, formal=True)
    assert "AUTHORIZATION_CORRUPT" in str(excinfo.value)

    write_task_state(campaign, raw=json.dumps({"authorization": {"formal_training_authorized": "true"}}))
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign, formal=True)
    assert "AUTHORIZATION_REQUIRED" in str(excinfo.value)


def test_upstream_chain_missing_gate_refused(campaign):
    # G2 passes but G0/G1 records were never written -> refuse
    write_gate(campaign, "G2-Move")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G2-Move", campaign)
    assert "CHAIN_GATE_MISSING" in str(excinfo.value)
    assert "G0-Data" in str(excinfo.value)


def test_upstream_chain_failed_gate_refused(campaign):
    write_chain(campaign, "G1-Feature")
    write_gate(campaign, "G1-Feature", status="fail")  # upstream regressed
    write_gate(campaign, "G2-Move")  # target itself passes
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G2-Move", campaign)
    assert "CHAIN_GATE_NOT_PASSED" in str(excinfo.value)
    assert "G1-Feature" in str(excinfo.value)


def test_full_chain_required_for_last_gate(campaign):
    write_chain(campaign, "G4-Rollout")
    record = gate_guard.assert_runnable("G4-Rollout", campaign)
    assert record["upstream_chain_verified"] == list(GATE_ORDER)


# ---------------------------------------------------------------------------
# every refusal path (fail-closed: no warn-and-continue anywhere)
# ---------------------------------------------------------------------------


def test_missing_gate_file_refused(campaign):
    # G0 record itself is missing -> target-level refusal
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "GATE_FILE_MISSING" in str(excinfo.value)


@pytest.mark.parametrize("status", ["fail", "blocked", "partial", "not_run", "PASS", None])
def test_non_pass_status_refused(campaign, status):
    write_chain(campaign, "G1-Feature")
    write_gate(campaign, "G1-Feature", status=status)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G1-Feature", campaign)
    assert "GATE_NOT_PASSED" in str(excinfo.value)


def test_corrupt_gate_file_refused(campaign):
    write_chain(campaign, "G3-V52Supervision")
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
    write_chain(campaign, "G4-Rollout", protocol_sha256="0" * 64)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G4-Rollout", campaign)
    assert "PROTOCOL_SHA_MISMATCH" in str(excinfo.value)


def test_protocol_sha_drift_after_gate_pass_refused(campaign):
    """Campaign yaml edited after the gates passed -> pin no longer matches."""
    write_chain(campaign, "G0-Data")
    yaml_path = campaign / "config" / "campaigns" / "pes2ts_30h_20261020.yaml"
    yaml_path.write_text(yaml_path.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_SHA_MISMATCH" in str(excinfo.value)


def test_missing_protocol_file_refused(campaign):
    write_chain(campaign, "G0-Data")
    (campaign / "config" / "campaigns" / "pes2ts_30h_20261020.yaml").unlink()
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_FILE_MISSING" in str(excinfo.value)


@pytest.mark.parametrize("bad_sha", [None, "", "zz" * 32, "abc"])
def test_missing_or_malformed_protocol_sha_refused(campaign, bad_sha):
    if bad_sha is None:
        write_chain(campaign, "G0-Data")
        path = campaign / "docs" / "agent_runs" / "state" / "gates" / "G0-Data.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["protocol_sha256"]
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        write_chain(campaign, "G0-Data", protocol_sha256=bad_sha)
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_SHA_MISSING" in str(excinfo.value)


@pytest.mark.parametrize("bad_id", [None, "", "  ", "docs/gates/G0", "a\\b", "..", "."])
def test_invalid_gate_id_refused(campaign, bad_id):
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable(bad_id, campaign)
    assert "INVALID_GATE_ID" in str(excinfo.value)


def test_unknown_gate_outside_serial_chain_refused(campaign):
    write_gate(campaign, "G0-Data")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G7-Unknown", campaign)
    assert "not in the serial gate chain" in str(excinfo.value)


def test_missing_campaign_root_refused(tmp_path):
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", tmp_path / "nope")
    assert "CAMPAIGN_ROOT_MISSING" in str(excinfo.value)


def test_gate_id_mismatch_refused(campaign):
    write_chain(campaign, "G2-Move")
    write_gate(campaign, "G2-Move", gate_id="G3-Label")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G2-Move", campaign)
    assert "GATE_ID_MISMATCH" in str(excinfo.value)


def test_protocol_file_escape_refused(campaign):
    write_chain(campaign, "G0-Data", protocol_file="../../../etc/passwd")
    with pytest.raises(GateNotPassedError) as excinfo:
        gate_guard.assert_runnable("G0-Data", campaign)
    assert "PROTOCOL_FILE_INVALID" in str(excinfo.value)


# ---------------------------------------------------------------------------
# gpu_lock.acquire integration (gates + authorization + cuda precheck)
# ---------------------------------------------------------------------------


def test_formal_acquire_with_passing_gate(campaign, tmp_path, gpu_ok):
    write_chain(campaign, "G2-Move")
    write_task_state(campaign, authorized=True)
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
        assert handle.holder["protocol_sha256"] == yaml_sha(campaign)
        assert handle.holder["cuda_checked"] is True
        assert gpu_lock.probe_status(lock)["state"] == "HELD"
    finally:
        handle.release()
    assert gpu_lock.probe_status(lock)["state"] == "FREE"


def test_formal_acquire_refused_while_unauthorized(campaign, tmp_path, gpu_ok):
    """Authorization flags are false -> formal acquire MUST refuse (ORA-2/4b)."""
    write_chain(campaign, "G2-Move")  # gates pass, but no user sign-off
    lock = tmp_path / "gpu.lock"
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(lock, gate_id="G2-Move", campaign_root=campaign)
    assert "AUTHORIZATION_REQUIRED" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()
    assert gpu_lock.probe_status(lock)["state"] == "FREE"


def test_formal_acquire_failing_gate_never_touches_lock(campaign, tmp_path, gpu_ok):
    write_chain(campaign, "G2-Move")
    write_gate(campaign, "G2-Move", status="blocked")
    lock = tmp_path / "gpu.lock"
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(lock, lane="W-LOSS", gate_id="G2-Move", campaign_root=campaign)
    assert "GATE_NOT_PASSED" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()
    assert gpu_lock.probe_status(lock)["state"] == "FREE"


def test_formal_run_meta_without_gate_id_refused(campaign, tmp_path, gpu_ok):
    lock = tmp_path / "gpu.lock"
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(
            lock, run_meta={"formal": True, "run_id": "R-x"}, campaign_root=campaign
        )
    assert "GATE_REQUIRED" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()


def test_gate_id_inside_run_meta_is_enforced(campaign, tmp_path, gpu_ok):
    lock = tmp_path / "gpu.lock"
    # no gate file at all -> refuse even though the caller passed no gate_id arg
    with pytest.raises(GateNotPassedError) as excinfo:
        gpu_lock.acquire(
            lock, run_meta={"gate_id": "G0-Data", "campaign_root": str(campaign)}
        )
    assert "GATE_FILE_MISSING" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()


def test_formal_acquire_refused_on_precheck_failure(campaign, tmp_path, monkeypatch):
    write_chain(campaign, "G2-Move")
    write_task_state(campaign, authorized=True)
    monkeypatch.setattr(
        gpu_lock,
        "precheck_gpu",
        lambda *a, **k: {
            "ok": False,
            "external_memory_mib": None,
            "error": "nvidia-smi probe failed",
            "reason": "probe_failure",
        },
    )
    lock = tmp_path / "gpu.lock"
    with pytest.raises(gpu_lock.GpuPrecheckError) as excinfo:
        gpu_lock.acquire(lock, gate_id="G2-Move", campaign_root=campaign)
    assert "GPU_PRECHECK_REFUSED" in str(excinfo.value)
    assert not gpu_lock.holder_path(lock).exists()


def test_formal_acquire_refused_on_external_occupancy(campaign, tmp_path, monkeypatch):
    write_chain(campaign, "G2-Move")
    write_task_state(campaign, authorized=True)
    monkeypatch.setattr(
        gpu_lock,
        "precheck_gpu",
        lambda *a, **k: {
            "ok": False,
            "external_memory_mib": 4096,
            "error": None,
            "reason": "external_memory",
        },
    )
    with pytest.raises(gpu_lock.GpuPrecheckError) as excinfo:
        gpu_lock.acquire(tmp_path / "gpu2.lock", gate_id="G2-Move", campaign_root=campaign)
    assert "external_memory" in str(excinfo.value)


def test_non_formal_acquire_warns_but_proceeds_on_precheck_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gpu_lock,
        "precheck_gpu",
        lambda *a, **k: {"ok": False, "external_memory_mib": 3000, "reason": "external_memory", "error": None},
    )
    lock = tmp_path / "gpu.lock"
    handle = gpu_lock.acquire(lock, lane="fixture", run_id="R-fixture")
    assert handle is not None  # non-formal = warn-only
    try:
        assert handle.holder["cuda_checked"] is True
    finally:
        handle.release()


def test_precheck_gpu_probe_failure_is_fail_closed_not_raising(tmp_path):
    # nonexistent smi binary -> structured refusal result, no exception
    result = gpu_lock.precheck_gpu(smi=str(tmp_path / "definitely-not-nvidia-smi"))
    assert result["ok"] is False
    assert result["reason"] == "probe_failure"
    assert result["error"]


def test_precheck_gpu_parses_external_memory(tmp_path, monkeypatch):
    fake_smi = tmp_path / "fake-nvidia-smi"
    fake_smi.write_text(
        "#!/bin/sh\necho '999999999, 900'\necho '999999998, 2000'\n", encoding="utf-8"
    )
    fake_smi.chmod(0o755)
    result = gpu_lock.precheck_gpu(smi=str(fake_smi))
    # other pids summed: 900 + 2000 = 2900 > 1500 threshold
    assert result["ok"] is False
    assert result["reason"] == "external_memory"
    assert result["external_memory_mib"] == 2900

    quiet = tmp_path / "quiet-nvidia-smi"
    quiet.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    quiet.chmod(0o755)
    result = gpu_lock.precheck_gpu(smi=str(quiet))
    assert result["ok"] is True
    assert result["external_memory_mib"] == 0


def test_legacy_fixture_acquire_stays_backward_compatible(campaign, tmp_path):
    """No run_meta / no gate_id -> old behaviour (fixture-level CUDA work);
    the default cuda precheck only warns for non-formal acquisitions."""
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
