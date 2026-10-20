"""WP-L3 plan-pin tests (mission: ledger plan-sha+mode pinning, lane W-LOSS).

Covers E0 residual #1 (Oracle-probed): before ANY ledger append, the bound
``run_plan.jsonl`` is re-hashed and its header ``campaign_mode`` re-read and
compared against the E0-frozen pin (``task_state.e0_freeze.run_plan_pinned``).
Every Oracle tamper scenario is replayed:

* (a) a frozen-plan COPY with confirm flags flipped -> ``PLAN_SHA_MISMATCH``,
  no append;
* (b) a PD-4 partial plan re-headered as ``full`` -> ``PLAN_SHA_MISMATCH``
  (byte change) and, with a sha-matching-but-wrong-mode pin,
  ``PLAN_MODE_MISMATCH``;
* (c) ANY other byte change (trailing newline, single-char edit) ->
  ``PLAN_SHA_MISMATCH``;
* positive: the pristine pinned plan appends fine; a synthetic plan pinned
  with its own sha launches fine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agent_ops.ledger.run_ledger import (
    DEFAULT_PLAN_PIN,
    CAMPAIGN_PIN,
    LedgerError,
    PINNED_CAMPAIGN_MODE,
    PINNED_RUN_PLAN_SHA256,
    PlanNotFrozenError,
    RunLedger,
    load_campaign_pin,
    sha256_file,
    verify_plan_pin,
)
from scripts.agent_ops.runplan import build_r3_runplan as builder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "PES2TS_30H_R3_V52_V53"
EVIDENCE_SHA = "ab" * 32
#: read-only copy of the real E0-frozen plan (byte-identical -> same sha)
REAL_PLAN = PROJECT_ROOT / "outputs/pesn/30h_campaign/run_plan.jsonl"
REAL_TASK_STATE = PROJECT_ROOT / "docs/agent_runs/state/task_state.json"

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


def _ledger_cli(args, expect=0):
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.ledger.run_ledger"] + args,
        cwd=str(PROJECT_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == expect, proc.stderr
    return proc.stdout, proc.stderr


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


def launch(rec):
    out = dict(rec)
    out["status"] = "running"
    out["pid"] = os.getpid()
    return out


def read_lines(path: Path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture
def real_plan_copy(tmp_path) -> Path:
    """Byte-identical copy of the frozen E0 plan (read-only source)."""
    if not REAL_PLAN.is_file():
        pytest.skip("real frozen plan not present in this checkout")
    dst = tmp_path / "run_plan.jsonl"
    dst.write_bytes(REAL_PLAN.read_bytes())
    assert sha256_file(dst) == PINNED_RUN_PLAN_SHA256
    return dst


def flip_first_confirm_flag(src: Path, dst: Path) -> None:
    """Tamper (a): copy the plan and flip ONE explicit confirm flag."""
    lines = src.read_text(encoding="utf-8").splitlines()
    out = []
    flipped = False
    for line in lines:
        rec = json.loads(line)
        if rec.get("record_type") != "plan_header" and rec.get("confirm") is True and not flipped:
            rec["confirm"] = False
            flipped = True
        out.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
    assert flipped, "fixture expects at least one confirm:true line"
    dst.write_text("".join(ln + "\n" for ln in out), encoding="utf-8")


# ---------------------------------------------------------------------------
# pin primitives
# ---------------------------------------------------------------------------


def test_pinned_constants_match_task_state_and_protocol():
    """The compiled pin IS the E0-frozen pin (task_state / protocol source)."""
    assert PINNED_RUN_PLAN_SHA256 == (
        "d2a63599727804e9ff407b30c7e5e1bdfc555c38fa7cafd5857a6cf273073163"
    )
    assert PINNED_CAMPAIGN_MODE == "full"
    assert DEFAULT_PLAN_PIN == {
        "sha256": PINNED_RUN_PLAN_SHA256,
        "campaign_mode": PINNED_CAMPAIGN_MODE,
    }
    if REAL_TASK_STATE.is_file():
        assert load_campaign_pin(REAL_TASK_STATE) == DEFAULT_PLAN_PIN
    protocol = PROJECT_ROOT / "outputs/pesn/30h_campaign/protocol_30h.json"
    if protocol.is_file():
        frozen = json.loads(protocol.read_text(encoding="utf-8"))["frozen_artifacts"][
            "run_plan"
        ]
        assert frozen["sha256_full"] == PINNED_RUN_PLAN_SHA256
        assert frozen["campaign_mode"] == PINNED_CAMPAIGN_MODE


def test_verify_plan_pin_pristine(real_plan_copy):
    verified = verify_plan_pin(real_plan_copy, DEFAULT_PLAN_PIN)
    assert verified == DEFAULT_PLAN_PIN


def test_load_campaign_pin_corrupt_refused(tmp_path):
    missing = tmp_path / "no_state.json"
    with pytest.raises(PlanNotFrozenError) as excinfo:
        load_campaign_pin(missing)
    assert "PLAN_PIN_SOURCE_MISSING" in str(excinfo.value)

    corrupt = tmp_path / "state.json"
    corrupt.write_text("{broken", encoding="utf-8")
    with pytest.raises(PlanNotFrozenError) as excinfo:
        load_campaign_pin(corrupt)
    assert "PLAN_PIN_SOURCE_CORRUPT" in str(excinfo.value)

    no_pin = tmp_path / "state2.json"
    no_pin.write_text(json.dumps({"e0_freeze": {}}), encoding="utf-8")
    with pytest.raises(PlanNotFrozenError) as excinfo:
        load_campaign_pin(no_pin)
    assert "PLAN_PIN_SOURCE_CORRUPT" in str(excinfo.value)

    bad_pin = tmp_path / "state3.json"
    bad_pin.write_text(
        json.dumps(
            {"e0_freeze": {"run_plan_pinned": {"sha256": "zz", "campaign_mode": "full"}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(PlanNotFrozenError) as excinfo:
        load_campaign_pin(bad_pin)
    assert "PLAN_PIN_INVALID" in str(excinfo.value)


def test_verify_plan_pin_input_refusals(tmp_path):
    with pytest.raises(PlanNotFrozenError) as excinfo:
        verify_plan_pin(tmp_path / "absent.jsonl", DEFAULT_PLAN_PIN)
    assert "PLAN_PIN_CHECK_FAILED" in str(excinfo.value)

    not_a_pin = tmp_path / "p.jsonl"
    not_a_pin.write_text("x\n", encoding="utf-8")
    with pytest.raises(PlanNotFrozenError) as excinfo:
        verify_plan_pin(not_a_pin, {"sha256": "nope", "campaign_mode": "full"})
    assert "PLAN_PIN_INVALID" in str(excinfo.value)

    # header missing / not a plan_header
    no_header = tmp_path / "noheader.jsonl"
    no_header.write_text(
        json.dumps({"campaign_id": CAMPAIGN, "phase": "R3"}) + "\n", encoding="utf-8"
    )
    pin = {"sha256": sha256_file(no_header), "campaign_mode": "full"}
    with pytest.raises(PlanNotFrozenError) as excinfo:
        verify_plan_pin(no_header, pin)
    assert "PLAN_PIN_CHECK_FAILED" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Oracle-probed tamper scenarios (replayed)
# ---------------------------------------------------------------------------


def test_oracle_a_confirm_flag_flip_rejected(real_plan_copy, tmp_path):
    """Tampered COPY with confirm flags flipped: append must refuse."""
    tampered = tmp_path / "tampered_confirm.jsonl"
    flip_first_confirm_flag(real_plan_copy, tampered)
    assert sha256_file(tampered) != PINNED_RUN_PLAN_SHA256  # bytes differ

    led = RunLedger(
        tmp_path / "r.jsonl",
        plan_path=tampered,
        plan_pin={"sha256": PINNED_RUN_PLAN_SHA256, "campaign_mode": "full"},
    )
    led.init()
    rec = make_record()
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(rec)
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)
    # fail-closed: nothing was written
    assert read_lines(led.path) == []
    assert read_lines(led.events_path) == []
    # and launch is equally refused
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(launch(rec))
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)
    assert read_lines(led.path) == []


def test_oracle_b_partial_plan_reheadered_as_full_rejected(tmp_path):
    """Tamper (b): PD-4 partial plan re-headered as full -> refuse."""
    partial = tmp_path / "partial.jsonl"
    builder.write_plan(partial, mode="partial", shas=FULL_SHAS)
    # re-header: claim full mode (Oracle probe) -- bytes changed -> sha mismatch
    lines = partial.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["campaign_mode"] == "partial"
    header["campaign_mode"] = "full"
    reheadered = tmp_path / "reheadered.jsonl"
    reheadered.write_text(
        json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n"
        + "".join(ln + "\n" for ln in lines[1:]),
        encoding="utf-8",
    )
    # pinned to the ORIGINAL partial sha+modes: the re-headered file no longer matches
    partial_pin = {"sha256": sha256_file(partial), "campaign_mode": "partial"}
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=reheadered, plan_pin=partial_pin
    )
    led.init()
    rec = make_record(arm="B", fold=0, variant=None, phase="R3")
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(rec)
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)
    assert read_lines(led.path) == []

    # defense in depth: even a sha-FAILING-mode pin catches a header that
    # lies about its mode (sha matches the pin, mode does not)
    lying_pin = {"sha256": sha256_file(reheadered), "campaign_mode": "partial"}
    with pytest.raises(PlanNotFrozenError) as excinfo:
        verify_plan_pin(reheadered, lying_pin)
    assert "PLAN_MODE_MISMATCH" in str(excinfo.value)


def test_oracle_c_any_byte_change_rejected(real_plan_copy, tmp_path):
    """Tamper (c): a single trailing newline change is a different sha."""
    appended = tmp_path / "appended.jsonl"
    appended.write_bytes(real_plan_copy.read_bytes() + b"\n")
    assert sha256_file(appended) != PINNED_RUN_PLAN_SHA256
    with pytest.raises(PlanNotFrozenError) as excinfo:
        verify_plan_pin(appended, DEFAULT_PLAN_PIN)
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)

    # a one-character edit inside a decision_note
    edited = tmp_path / "edited.jsonl"
    text = real_plan_copy.read_text(encoding="utf-8")
    edited.write_text(text.replace("forward-only", "forward_only", 1), encoding="utf-8")
    assert sha256_file(edited) != PINNED_RUN_PLAN_SHA256
    with pytest.raises(PlanNotFrozenError) as excinfo:
        verify_plan_pin(edited, DEFAULT_PLAN_PIN)
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)


def test_oracle_tamper_between_appends_is_recaught(real_plan_copy, tmp_path):
    """Pin is re-verified on EVERY append, not just the first."""
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=real_plan_copy, plan_pin=DEFAULT_PLAN_PIN
    )
    led.init()
    rec = make_record()
    assert led.append(rec).action == "created"  # pristine -> ok
    # tamper AFTER registration: the next append must refuse
    tampered = tmp_path / "tampered.jsonl"
    flip_first_confirm_flag(real_plan_copy, tampered)
    real_plan_copy.write_bytes(tampered.read_bytes())
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(launch(rec))
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)
    assert len(read_lines(led.path)) == 1  # registration only


# ---------------------------------------------------------------------------
# positive paths
# ---------------------------------------------------------------------------


def test_pristine_pinned_plan_append_succeeds(real_plan_copy, tmp_path):
    """Positive: the pristine pinned plan appends (registration + noop)."""
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=real_plan_copy, plan_pin=DEFAULT_PLAN_PIN
    )
    led.init()
    rec = make_record()
    assert led.append(rec).action == "created"
    assert led.append(rec).action == "noop"
    assert len(read_lines(led.path)) == 1
    # launch is still refused -- but by the two-stage PENDING binding, NOT by
    # the pin (the frozen plan legitimately carries PENDING_E1/E2 tokens)
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(launch(rec))
    assert "PLAN_NOT_FROZEN" in str(excinfo.value)
    assert "PLAN_SHA_MISMATCH" not in str(excinfo.value)


def test_synthetic_pinned_plan_launch_succeeds(tmp_path):
    """Positive: a synthetic plan pinned with its own sha launches."""
    plan = tmp_path / "run_plan.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)
    pin = {"sha256": sha256_file(plan), "campaign_mode": "full"}
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan, plan_pin=pin)
    led.init()
    rec = make_record()
    led.append(rec)
    outcome = led.append(launch(rec))
    assert outcome.action == "transitioned"
    assert outcome.entry["status"] == "running"


def test_default_pin_binding_for_bound_plan(tmp_path):
    """Binding a plan WITHOUT an explicit pin defaults to the campaign pin."""
    plan = tmp_path / "fixture_plan.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)  # NOT the pinned bytes
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)  # CAMPAIGN_PIN default
    assert led.plan_pin == DEFAULT_PLAN_PIN
    led.init()
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(make_record())
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)

    # explicit pin=None is the documented fixture opt-out
    led2 = RunLedger(tmp_path / "r2.jsonl", plan_path=plan, plan_pin=None)
    assert led2.plan_pin is None
    led2.init()
    assert led2.append(make_record()).action == "created"

    # an explicit pin without a bound plan is incoherent -> refuse
    with pytest.raises(LedgerError) as excinfo:
        RunLedger(tmp_path / "r3.jsonl", plan_pin=DEFAULT_PLAN_PIN)
    assert "PLAN_PIN_WITHOUT_PLAN" in str(excinfo.value)
    # invalid explicit pins refuse at construction
    with pytest.raises(PlanNotFrozenError) as excinfo:
        RunLedger(
            tmp_path / "r4.jsonl", plan_path=plan, plan_pin={"sha256": "x" * 64}
        )
    assert "PLAN_PIN_INVALID" in str(excinfo.value)


def test_confirm_and_metric_unseal_verify_pin(real_plan_copy, tmp_path):
    """The unseal paths verify the pin too (they read the plan for batches)."""
    led = RunLedger(
        tmp_path / "r.jsonl",
        plan_path=real_plan_copy,
        plan_pin=DEFAULT_PLAN_PIN,
        unseal_path=tmp_path / "unseal.json",
        metric_unseal_path=tmp_path / "metric.json",
    )
    led.init()
    assert led.confirm_unseal(run_id="R-x", allowed_run_ids=["R-x"])["count"] == 1
    # tamper -> unseal refuses before touching the token
    tampered = tmp_path / "tampered.jsonl"
    flip_first_confirm_flag(real_plan_copy, tampered)
    real_plan_copy.write_bytes(tampered.read_bytes())
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.confirm_unseal(run_id="R-y", allowed_run_ids=["R-y"])
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.metric_unseal(run_id="R-z", allowed_run_ids=["R-z"])
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_default_pin_refuses_tampered_plan(real_plan_copy, tmp_path):
    ledger_path = tmp_path / "run_records.jsonl"
    _ledger_cli(["--ledger", str(ledger_path), "init"])
    tampered = tmp_path / "tampered.jsonl"
    flip_first_confirm_flag(real_plan_copy, tampered)
    out, err = _ledger_cli(
        [
            "--ledger",
            str(ledger_path),
            "--plan",
            str(tampered),
            "append",
            "--record-json",
            json.dumps(make_record()),
        ],
        expect=2,
    )
    assert out == ""
    assert "PLAN_SHA_MISMATCH" in err
    assert read_lines(ledger_path) == []


def test_cli_pristine_plan_append_and_explicit_pin_file(real_plan_copy, tmp_path):
    ledger_path = tmp_path / "run_records.jsonl"
    _ledger_cli(["--ledger", str(ledger_path), "init"])
    out, _ = _ledger_cli(
        [
            "--ledger",
            str(ledger_path),
            "--plan",
            str(real_plan_copy),
            "append",
            "--record-json",
            json.dumps(make_record()),
        ]
    )
    assert json.loads(out)["action"] == "created"

    # explicit --plan-pin-file overrides the task_state-derived pin
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps(DEFAULT_PLAN_PIN), encoding="utf-8")
    ledger2 = tmp_path / "r2.jsonl"
    _ledger_cli(["--ledger", str(ledger2), "init"])
    out, _ = _ledger_cli(
        [
            "--ledger",
            str(ledger2),
            "--plan",
            str(real_plan_copy),
            "--plan-pin-file",
            str(pin_file),
            "append",
            "--record-json",
            json.dumps(make_record()),
        ]
    )
    assert json.loads(out)["action"] == "created"

    # --plan-pin none = explicit fixture opt-out: tampered plan appends
    ledger3 = tmp_path / "r3.jsonl"
    _ledger_cli(["--ledger", str(ledger3), "init"])
    tampered = tmp_path / "tampered.jsonl"
    flip_first_confirm_flag(real_plan_copy, tampered)
    out, _ = _ledger_cli(
        [
            "--ledger",
            str(ledger3),
            "--plan",
            str(tampered),
            "--plan-pin",
            "none",
            "append",
            "--record-json",
            json.dumps(make_record()),
        ]
    )
    assert json.loads(out)["action"] == "created"

    # corrupt --plan-pin-file refuses
    bad_pin = tmp_path / "bad_pin.json"
    bad_pin.write_text("{broken", encoding="utf-8")
    out, err = _ledger_cli(
        [
            "--ledger",
            str(ledger3),
            "--plan",
            str(real_plan_copy),
            "--plan-pin-file",
            str(bad_pin),
            "append",
            "--record-json",
            json.dumps(make_record(seed=29)),
        ],
        expect=2,
    )
    assert "PLAN_PIN_SOURCE_CORRUPT" in err
