"""WP-L3 metric_unseal wiring tests (mission: B folds3/4 reveal consumption).

Covers E0 residual #2: the one-shot ``metric_unseal`` token for B-stage
folds3/4 sealed-baseline metrics (30h §4.2) is CONSUMED by the ``done``
transition and recorded on the run record -- previously the token existed but
nothing consumed it, so the sealed-metric contract was decorative.

Semantics under test (fail-closed, batch conventions identical to the confirm
token):
* first B folds3/4 ``done`` consumes the token for the plan's B folds3/4
  batch (count=1) and records ``metric_unseal`` on the run record;
* other members of the batch finalize idempotently (count stays 1);
* a ``done`` outside the registered batch -> ``UNSEAL_BATCH_MISMATCH`` /
  ``UNSEAL_OUTSIDE_BATCH`` per existing conventions;
* double consume (a *different* batch) -> refused;
* unconfigured token path -> ``METRIC_UNSEAL_NOT_CONFIGURED``, nothing written;
* non-B-folds3/4 runs finalize without touching the token;
* the bound plan is pin-verified (WP-L3 residual #1) before any of this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.agent_ops.ledger.run_ledger import (
    PlanNotFrozenError,
    RunLedger,
    UnsealExceeded,
    compute_run_id,
    sha256_file,
)
from scripts.agent_ops.runplan import build_r3_runplan as builder

CAMPAIGN = "PES2TS_30H_R3_V52_V53"
EVIDENCE_SHA = "ab" * 32

FULL_SHAS = {
    "config_sha256": "e5" * 32,
    "data_sha256": "a1" * 32,
    "teacher_sha256": "b2" * 32,
    "features_sha256": "c3" * 32,
    "split_sha256": "d4" * 32,
}


def make_record(**overrides):
    record = {
        "campaign_id": CAMPAIGN,
        "phase": "R3",
        "arm": "B",
        "fold": 3,
        "seed": 17,
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


def done(rec, metrics_path="outputs/pesn/30h_campaign/metrics/baseline.json"):
    out = dict(rec)
    out["status"] = "done"
    out["checkpoint_update"] = 3000
    out["checkpoint_sha256"] = "ab" * 32
    out["metrics_path"] = metrics_path
    return out


def read_lines(path: Path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture
def pinned_plan(tmp_path) -> Path:
    plan = tmp_path / "run_plan.jsonl"
    builder.write_plan(plan, shas=FULL_SHAS)
    return plan


@pytest.fixture
def pin(pinned_plan):
    return {"sha256": sha256_file(pinned_plan), "campaign_mode": "full"}


@pytest.fixture
def ledger(tmp_path, pinned_plan, pin) -> RunLedger:
    led = RunLedger(
        tmp_path / "r.jsonl",
        plan_path=pinned_plan,
        plan_pin=pin,
        metric_unseal_path=tmp_path / "metric_unseal.json",
    )
    led.init()
    return led


def b_fold(fold, seed):
    return make_record(arm="B", fold=fold, seed=seed)


# ---------------------------------------------------------------------------
# consumption wiring
# ---------------------------------------------------------------------------


def test_first_b_folds34_done_consumes_token_and_records_it(ledger, tmp_path):
    rec = b_fold(3, 17)
    led = ledger
    led.append(rec)
    led.append(launch(rec))
    token_path = tmp_path / "metric_unseal.json"
    assert not token_path.exists()  # launch does NOT burn the metric token

    outcome = led.append(done(rec))
    assert outcome.action == "transitioned"
    # consumption recorded on the run record
    receipt = outcome.entry["metric_unseal"]
    assert receipt["kind"] == "metric"
    assert receipt["batch_id"] == "plan-metric-batch"
    assert receipt["count"] == 1
    assert receipt["consumed_at"]
    # and on the token file: batch = the plan's B folds3/4 run set
    token = json.loads(token_path.read_text(encoding="utf-8"))
    assert token["kind"] == "metric"
    assert token["count"] == 1
    from scripts.agent_ops.ledger.run_ledger import read_run_plan

    plan = read_run_plan(led.plan_path)
    assert sorted(token["allowed_run_ids"]) == plan["metric_run_ids"]
    assert len(plan["metric_run_ids"]) == 6  # B folds3,4 x 3 seeds
    assert compute_run_id(rec) in token["allowed_run_ids"]


def test_batch_member_finalizes_idempotently(ledger, tmp_path):
    """Second/third batch member: token already consumed, count stays 1."""
    led = ledger
    r3 = b_fold(3, 17)
    led.append(r3)
    led.append(launch(r3))
    led.append(done(r3))

    r4 = b_fold(4, 29)
    led.append(r4)
    led.append(launch(r4))
    outcome = led.append(done(r4))
    assert outcome.action == "transitioned"
    assert outcome.entry["metric_unseal"]["count"] == 1  # no re-increment
    token = json.loads((tmp_path / "metric_unseal.json").read_text(encoding="utf-8"))
    assert token["count"] == 1


def test_non_b_folds34_done_never_touches_token(ledger, tmp_path):
    led = ledger
    # B fold1 (sealed-eval training but NOT folds3/4) -> no metric reveal
    rec = b_fold(1, 17)
    led.append(rec)
    led.append(launch(rec))
    outcome = led.append(done(rec))
    assert "metric_unseal" not in outcome.entry
    assert not (tmp_path / "metric_unseal.json").exists()

    # F-main fold0 run -> also untouched
    f0 = make_record(arm="F0", fold=0, seed=17, variant="forward")
    led.append(f0)
    led.append(launch(f0))
    outcome = led.append(done(f0))
    assert "metric_unseal" not in outcome.entry
    assert not (tmp_path / "metric_unseal.json").exists()


def test_unconfigured_token_refuses_done_and_writes_nothing(tmp_path, pinned_plan, pin):
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=pinned_plan, plan_pin=pin
        # metric_unseal_path NOT configured -> fail-closed
    )
    led.init()
    rec = b_fold(3, 17)
    led.append(rec)
    led.append(launch(rec))
    before = len(read_lines(led.path))
    with pytest.raises(UnsealExceeded) as excinfo:
        led.append(done(rec))
    assert "METRIC_UNSEAL_NOT_CONFIGURED" in str(excinfo.value)
    assert len(read_lines(led.path)) == before  # no done line written
    assert led.status(compute_run_id(rec))["status"] == "running"


def test_second_different_batch_refused(tmp_path, pinned_plan, pin):
    """Double consume: a DIFFERENT pre-registered batch is refused."""
    token_path = tmp_path / "metric_unseal.json"
    led = RunLedger(
        tmp_path / "r.jsonl",
        plan_path=pinned_plan,
        plan_pin=pin,
        metric_unseal_path=token_path,
    )
    led.init()
    # consume for some other batch first (out-of-band caller)
    from scripts.agent_ops.ledger.run_ledger import metric_unseal

    metric_unseal(
        token_path, run_id="intruder", batch_id="other-batch", allowed_run_ids=["intruder"]
    )
    rec = b_fold(3, 17)
    led.append(rec)
    led.append(launch(rec))
    before = len(read_lines(led.path))
    with pytest.raises(UnsealExceeded) as excinfo:
        led.append(done(rec))
    assert "UNSEAL_BATCH_MISMATCH" in str(excinfo.value)
    assert len(read_lines(led.path)) == before
    assert led.status(compute_run_id(rec))["status"] == "running"
    assert json.loads(token_path.read_text(encoding="utf-8"))["count"] == 1


def test_corrupt_token_refuses_done(tmp_path, pinned_plan, pin):
    token_path = tmp_path / "metric_unseal.json"
    token_path.write_text("{broken", encoding="utf-8")
    led = RunLedger(
        tmp_path / "r.jsonl",
        plan_path=pinned_plan,
        plan_pin=pin,
        metric_unseal_path=token_path,
    )
    led.init()
    rec = b_fold(3, 17)
    led.append(rec)
    led.append(launch(rec))
    with pytest.raises(UnsealExceeded) as excinfo:
        led.append(done(rec))
    assert "UNSEAL_CORRUPT" in str(excinfo.value)
    assert led.status(compute_run_id(rec))["status"] == "running"


def test_pin_violation_beats_metric_wiring(tmp_path, pinned_plan):
    """A tampered plan refuses the done transition before any token burn."""
    pin = {"sha256": sha256_file(pinned_plan), "campaign_mode": "full"}
    token_path = tmp_path / "metric_unseal.json"
    led = RunLedger(
        tmp_path / "r.jsonl",
        plan_path=pinned_plan,
        plan_pin=pin,
        metric_unseal_path=token_path,
    )
    led.init()
    rec = b_fold(3, 17)
    led.append(rec)
    led.append(launch(rec))
    # tamper the plan between launch and finalize
    pinned_plan.write_bytes(pinned_plan.read_bytes() + b"\n")
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(done(rec))
    assert "PLAN_SHA_MISMATCH" in str(excinfo.value)
    assert not token_path.exists()  # token untouched


def test_ledger_method_metric_unseal_uses_plan_batch(ledger, tmp_path):
    """The explicit ledger.metric_unseal() also defaults to the plan batch."""
    from scripts.agent_ops.ledger.run_ledger import read_run_plan

    plan = read_run_plan(ledger.plan_path)
    batch = plan["metric_run_ids"]
    assert len(batch) == 6  # B folds3,4 x 3 seeds
    first, second = batch[0], batch[1]
    out = ledger.metric_unseal(run_id=first)
    assert out["kind"] == "metric"
    assert out["batch_id"] == "plan-metric-batch"
    assert sorted(out["allowed_run_ids"]) == batch
    # idempotent inside the batch
    assert ledger.metric_unseal(run_id=second)["count"] == 1
    # a run outside the plan batch is refused (pre-registration)
    with pytest.raises(UnsealExceeded) as excinfo:
        ledger.metric_unseal(run_id="outsider", allowed_run_ids=["outsider"])
    assert "UNSEAL_BATCH_MISMATCH" in str(excinfo.value)
