"""WP-L2 ledger extension tests (mission: ledger extensions, lane W-LOSS).

Covers the formal-record schema (gate_id / admissibility / evidence_sha /
attempt / parent_run_id), the hashed run_id scheme (same name + different
config_sha => a DIFFERENT run, no silent reuse), budget caps R3=66 / V5.2=33
counting ALL launch attempts incl. retries with the run_plan.jsonl
cross-check, and the one-shot fold3/4 confirm unseal counter
(``state/gates/unseal.json``, count<=1).  Every guard refuses on
missing/invalid/corrupt input -- no warn-and-pass paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.agent_ops.ledger.run_ledger import (
    DEFAULT_CAPS,
    BudgetCapExceeded,
    IllegalTransition,
    LedgerError,
    PlaceholderHashError,
    RunIdConflict,
    RunLedger,
    UnsealExceeded,
    compute_run_id,
    confirm_unseal,
    is_confirm_run,
    read_run_plan,
    validate,
)

CAMPAIGN = "PES2TS_30H_R3_V52_V53"
EVIDENCE_SHA = "ab" * 32


def make_record(**overrides):
    """Legacy (pre-WP-L2) fully valid section 7.2 record."""
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


def make_formal(**overrides):
    """Formal WP-L2 record (all five formal keys present -> hashed run_id)."""
    record = make_record(phase="R3", arm="F0", fold=0, seed=17)
    record.update(
        {
            "gate_id": "G2-Move",
            "admissibility": "admissible",
            "evidence_sha": EVIDENCE_SHA,
            "attempt": 1,
            "parent_run_id": None,
        }
    )
    record.update(overrides)
    return record


def launch(rec):
    out = dict(rec)
    out["status"] = "running"
    out["pid"] = os.getpid()
    return out


def fail(rec):
    out = dict(rec)
    out["status"] = "failed"
    return out


@pytest.fixture
def ledger(tmp_path) -> RunLedger:
    led = RunLedger(tmp_path / "run_records.jsonl")
    led.init()
    return led


def read_lines(path: Path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# formal record schema (fail-closed validation)
# ---------------------------------------------------------------------------


def test_formal_record_roundtrip_and_fields(ledger):
    outcome = ledger.append(make_formal())
    assert outcome.action == "created"
    entry = outcome.entry
    assert entry["gate_id"] == "G2-Move"
    assert entry["admissibility"] == "admissible"
    assert entry["evidence_sha"] == EVIDENCE_SHA
    assert entry["attempt"] == 1
    assert entry["parent_run_id"] is None
    assert len(entry["run_id"]) == 16  # hashed scheme
    # transition keeps the formal metadata and advances status
    outcome = ledger.append(launch(make_formal()))
    assert outcome.action == "transitioned"
    assert outcome.entry["gate_id"] == "G2-Move"
    assert outcome.entry["status"] == "running"


def test_incomplete_formal_schema_refused(ledger):
    rec = make_formal()
    del rec["attempt"]
    with pytest.raises(LedgerError) as excinfo:
        ledger.append(rec)
    assert "INCOMPLETE_FORMAL_FIELDS" in str(excinfo.value)


def test_invalid_formal_field_values_refused():
    for overrides, marker in [
        ({"gate_id": ""}, "INVALID_GATE_ID"),
        ({"gate_id": None}, "INVALID_GATE_ID"),
        ({"admissibility": "maybe"}, "INVALID_ADMISSIBILITY"),
        ({"admissibility": True}, "INVALID_ADMISSIBILITY"),
        ({"attempt": 0}, "INVALID_ATTEMPT"),
        ({"attempt": True}, "INVALID_ATTEMPT"),
        ({"attempt": "1"}, "INVALID_ATTEMPT"),
        ({"parent_run_id": 7}, "INVALID_PARENT_RUN_ID"),
        ({"evidence_sha": "zz"}, "INVALID_HASH"),
        ({"evidence_sha": "REQUIRED_AT_RUNTIME"}, "INVALID_HASH"),
    ]:
        with pytest.raises(LedgerError) as excinfo:
            validate(make_formal(**overrides))
        assert marker in str(excinfo.value), overrides


def test_admissible_requires_gate_report_evidence():
    with pytest.raises(PlaceholderHashError) as excinfo:
        validate(make_formal(admissibility="admissible", evidence_sha=None))
    assert "MISSING_HASH" in str(excinfo.value)
    # not-admissible-until-gate-pass may legitimately have no evidence yet
    validate(make_formal(admissibility="not_admissible_until_gate_pass", evidence_sha=None))


def test_admissibility_same_status_drift_is_conflict(ledger):
    ledger.append(make_formal())
    with pytest.raises(RunIdConflict) as excinfo:
        ledger.append(make_formal(admissibility="not_admissible_until_gate_pass"))
    assert "RUN_ID_CONFLICT" in str(excinfo.value)


def test_legacy_and_formal_mixing_fails_loudly(ledger):
    """Legacy planned record then a formal (hashed-id) launch: never silent."""
    ledger.append(make_record(phase="R3", arm="F0", fold=0, seed=17))
    with pytest.raises(IllegalTransition) as excinfo:
        ledger.append(launch(make_formal()))
    assert "first record" in str(excinfo.value)


# ---------------------------------------------------------------------------
# run_id hash sensitivity
# ---------------------------------------------------------------------------


def test_run_id_hash_sensitivity_same_name_different_config():
    base = make_formal()
    changed = make_formal(config_sha256="ff" * 32)
    id_base = compute_run_id(base)
    id_changed = compute_run_id(changed)
    assert id_base != id_changed  # changed config => DIFFERENT run (no reuse)
    assert len(id_base) == len(id_changed) == 16
    assert all(c in "0123456789abcdef" for c in id_base)
    # deterministic for identical input
    assert compute_run_id(base) == id_base
    # every identity component is in the hash: flip each field -> new id
    for field in (
        "data_sha256",
        "teacher_sha256",
        "features_sha256",
        "split_sha256",
        "config_sha256",
        "arm",
        "seed",
        "fold",
        "phase",
        "variant",
    ):
        if field.endswith("_sha256"):
            overrides = {field: "99" * 32}
        elif field in ("seed", "fold"):
            overrides = {field: base[field] + 1}
        else:
            overrides = {field: "other_{}".format(base.get(field))}
        other = make_formal(**overrides)
        assert compute_run_id(other) != id_base, field


def test_legacy_run_id_scheme_unchanged():
    rec = make_record()
    assert compute_run_id(rec) == "{}:V5.2:QT:f0:s17".format(CAMPAIGN)
    assert compute_run_id(rec, scheme="legacy") == compute_run_id(rec)
    # explicit hashed scheme also works on legacy records
    hashed = compute_run_id(rec, scheme="hashed")
    assert len(hashed) == 16 and hashed != compute_run_id(rec)


def test_ledger_auto_scheme_creates_two_runs_for_changed_config(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl")
    led.init()
    first = led.append(make_formal())
    second = led.append(make_formal(config_sha256="ff" * 32))
    assert first.action == "created" and second.action == "created"
    assert first.entry["run_id"] != second.entry["run_id"]  # no silent reuse/conflict
    assert len(read_lines(led.path)) == 2

    # explicit scheme + invalid scheme
    forced = RunLedger(tmp_path / "forced.jsonl", run_id_scheme="hashed")
    forced.init()
    out = forced.append(make_record())
    assert len(out.entry["run_id"]) == 16
    with pytest.raises(LedgerError) as excinfo:
        RunLedger(tmp_path / "x.jsonl", run_id_scheme="bogus")
    assert "INVALID_SCHEME" in str(excinfo.value)


# ---------------------------------------------------------------------------
# budget caps: ALL launch attempts incl. retries, run_plan cross-check
# ---------------------------------------------------------------------------


def test_default_caps_match_task_state():
    assert DEFAULT_CAPS == {"R3": 66, "V5.2": 33}


def test_budget_cap_exceeded_across_runs(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl", caps={"R3": 1})
    led.init()
    a = make_formal(arm="F0", seed=17)
    b = make_formal(arm="F1", seed=17)
    led.append(a)
    led.append(launch(a))  # attempt 1 <= cap 1
    led.append(b)
    before = len(read_lines(led.path))
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(b))  # attempt 2 > cap 1
    assert "BUDGET_CAP_EXCEEDED" in str(excinfo.value)
    assert len(read_lines(led.path)) == before  # refused launch wrote nothing


def test_budget_cap_counts_retries_as_attempts(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl", caps={"R3": 1})
    led.init()
    a = make_formal(arm="F0", seed=17)
    led.append(a)
    led.append(launch(a))  # attempt 1
    led.append(fail(a))
    before = len(read_lines(led.path))
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(a))  # retry = attempt 2 > cap 1 (retries are counted)
    assert "BUDGET_CAP_EXCEEDED" in str(excinfo.value)
    assert len(read_lines(led.path)) == before

    # with a bigger cap the same retry fits and is tracked via retry_count
    led2 = RunLedger(tmp_path / "r2.jsonl", caps={"R3": 2})
    led2.init()
    led2.append(a)
    led2.append(launch(a))
    led2.append(fail(a))
    out = led2.append(launch(a))
    assert out.entry["retry_count"] == 1
    led2.append(fail(a))
    with pytest.raises(BudgetCapExceeded):
        led2.append(launch(a))  # 3rd attempt > cap 2


def test_uncapped_phase_is_not_budget_governed(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl")
    led.init()
    for i in range(3):  # E0 smoke-style phase: no cap, no plan binding
        rec = make_formal(phase="E0-smoke", arm="smoke", seed=17 + i)
        led.append(rec)
        led.append(launch(rec))
    assert len(read_lines(led.path)) == 6


def write_plan_file(path: Path, rows) -> Path:
    header = {"record_type": "plan_header", "campaign_id": CAMPAIGN, "pd3_note": "x"}
    lines = [json.dumps(header)]
    lines.extend(json.dumps(r) for r in rows)
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return path


def synthetic_plan_row(**overrides):
    row = {
        "run_id": "0000000000000000",
        "campaign_id": CAMPAIGN,
        "phase": "R3",
        "substage": "F-main",
        "arm": "F0",
        "fold": 0,
        "seed": 17,
        "variant": "forward",
        "reuses": [],
        "status": "planned",
    }
    row.update(overrides)
    return row


def test_plan_missing_or_corrupt_refused_at_launch(tmp_path):
    missing = tmp_path / "plan.jsonl"
    led = RunLedger(tmp_path / "r.jsonl", plan_path=missing)
    led.init()
    rec = make_formal()
    led.append(rec)  # registration does not consume budget
    with pytest.raises(LedgerError) as excinfo:
        led.append(launch(rec))
    assert "PLAN_MISSING" in str(excinfo.value)

    corrupt = tmp_path / "plan2.jsonl"
    corrupt.write_text("{broken\n", encoding="utf-8")
    led2 = RunLedger(tmp_path / "r2.jsonl", plan_path=corrupt)
    led2.init()
    led2.append(rec)
    with pytest.raises(LedgerError) as excinfo:
        led2.append(launch(rec))
    assert "PLAN_CORRUPT" in str(excinfo.value)

    empty = tmp_path / "plan3.jsonl"
    empty.write_text("\n", encoding="utf-8")
    led3 = RunLedger(tmp_path / "r3.jsonl", plan_path=empty)
    led3.init()
    led3.append(rec)
    with pytest.raises(LedgerError) as excinfo:
        led3.append(launch(rec))
    assert "PLAN_CORRUPT" in str(excinfo.value)


def test_plan_not_in_plan_and_not_planned_refused(tmp_path):
    plan = write_plan_file(tmp_path / "plan.jsonl", [synthetic_plan_row()])
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()

    # in-plan run launches fine
    ok_rec = make_formal(arm="F0", fold=0, seed=17)
    led.append(ok_rec)
    assert led.append(launch(ok_rec)).action == "transitioned"

    # same phase, run absent from the plan -> refuse
    outsider = make_formal(arm="F99", fold=0, seed=17)
    led.append(outsider)
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(outsider))
    assert "NOT_IN_PLAN" in str(excinfo.value)

    # phase carrying a default cap but zero planned runs -> refuse
    unplanned_phase = make_formal(phase="V5.2", arm="QT", fold=0, seed=17)
    led.append(unplanned_phase)
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(unplanned_phase))
    assert "NOT_PLANNED" in str(excinfo.value)


def test_plan_count_is_the_effective_cap_including_retries(tmp_path):
    plan = write_plan_file(tmp_path / "plan.jsonl", [synthetic_plan_row()])
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()
    rec = make_formal()
    led.append(rec)
    led.append(launch(rec))  # 1 planned run -> cap 1: first attempt ok
    led.append(fail(rec))
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(rec))  # retry would make 2 attempts > plan count 1
    assert "BUDGET_CAP_EXCEEDED" in str(excinfo.value)


def test_real_run_plan_binds_f_main_launches(tmp_path):
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "run_plan.jsonl"
    write_plan(plan)
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=plan, caps={"R3": 66, "V5.2": 33}
    )
    led.init()
    # three in-plan F-main launches (real 66-run plan rows)
    for seed in (17, 29, 43):
        rec = make_formal(arm="F0", fold=0, seed=seed)
        led.append(rec)
        assert led.append(launch(rec)).action == "transitioned"
    # V5.2 QT row from the plan also binds
    qt = make_record(phase="V5.2", arm="QT", fold=0, seed=17)
    qt.update(
        {
            "gate_id": "G3-V52Supervision",
            "admissibility": "admissible",
            "evidence_sha": EVIDENCE_SHA,
            "attempt": 1,
            "parent_run_id": None,
        }
    )
    led.append(qt)
    assert led.append(launch(qt)).action == "transitioned"
    # seed outside the legal plan seeds -> NOT_IN_PLAN
    bad = make_formal(arm="F0", fold=0, seed=99)
    led.append(bad)
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(bad))
    assert "NOT_IN_PLAN" in str(excinfo.value)


# ---------------------------------------------------------------------------
# one-shot fold3/4 confirm unseal counter
# ---------------------------------------------------------------------------


def test_is_confirm_run_detection():
    assert is_confirm_run(make_formal(arm="C", fold=3)) is True
    assert is_confirm_run(make_formal(arm="C", fold=4)) is True
    assert is_confirm_run(make_formal(arm="QT", variant="confirm", fold=4)) is True
    assert is_confirm_run(make_formal(arm="QT", variant="confirm", fold=2)) is False
    assert is_confirm_run(make_formal(arm="B", fold=3)) is False  # sealed eval != confirm
    assert is_confirm_run(make_formal(arm="C", fold=0)) is False


def test_confirm_unseal_one_shot(tmp_path):
    unseal = tmp_path / "unseal.json"
    led = RunLedger(tmp_path / "r.jsonl", unseal_path=unseal)
    led.init()

    first = make_formal(arm="C", fold=3, seed=17)
    led.append(first)
    assert led.append(launch(first)).action == "transitioned"
    token = json.loads(unseal.read_text(encoding="utf-8"))
    assert token["count"] == 1
    assert token["run_id"] == compute_run_id(first)
    assert token["used_at"]

    # same-run retry: token already ours -> idempotent, never re-incremented
    led.append(fail(first))
    led.append(launch(first))
    assert json.loads(unseal.read_text(encoding="utf-8"))["count"] == 1

    # a DIFFERENT confirm run must be refused (one-shot unseal)
    second = make_formal(arm="C", fold=4, seed=29)
    led.append(second)
    before = len(read_lines(led.path))
    with pytest.raises(UnsealExceeded) as excinfo:
        led.append(launch(second))
    assert "UNSEAL_ALREADY_USED" in str(excinfo.value)
    assert len(read_lines(led.path)) == before
    assert json.loads(unseal.read_text(encoding="utf-8"))["count"] == 1


def test_confirm_run_without_unseal_config_refused(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl")  # unseal_path=None (fail-closed default)
    led.init()
    rec = make_formal(arm="C", fold=3, seed=17)
    led.append(rec)
    with pytest.raises(UnsealExceeded) as excinfo:
        led.append(launch(rec))
    assert "UNSEAL_NOT_CONFIGURED" in str(excinfo.value)


def test_non_confirm_fold34_run_needs_no_unseal(tmp_path):
    unseal = tmp_path / "unseal.json"
    led = RunLedger(tmp_path / "r.jsonl", unseal_path=unseal)
    led.init()
    rec = make_formal(arm="B", fold=3, seed=29)  # sealed-eval training, not confirm
    led.append(rec)
    assert led.append(launch(rec)).action == "transitioned"
    assert not unseal.exists()  # token untouched


def test_corrupt_or_tampered_unseal_refused(tmp_path):
    unseal = tmp_path / "unseal.json"
    unseal.write_text("{broken", encoding="utf-8")
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="R-x")
    assert "UNSEAL_CORRUPT" in str(excinfo.value)

    unseal.write_text(json.dumps({"count": "many"}), encoding="utf-8")
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="R-x")
    assert "UNSEAL_CORRUPT" in str(excinfo.value)

    # tampered counter already consumed by someone else -> refuse
    unseal.write_text(json.dumps({"count": 7, "run_id": "intruder"}), encoding="utf-8")
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="R-x")
    assert "UNSEAL_ALREADY_USED" in str(excinfo.value)

    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(None)
    assert "UNSEAL_NOT_CONFIGURED" in str(excinfo.value)


def test_ledger_confirm_unseal_method(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl", unseal_path=tmp_path / "unseal.json")
    led.init()
    assert led.confirm_unseal(run_id="R-1")["count"] == 1
    with pytest.raises(UnsealExceeded):
        led.confirm_unseal(run_id="R-2")

    plain = RunLedger(tmp_path / "r2.jsonl")
    plain.init()
    with pytest.raises(UnsealExceeded) as excinfo:
        plain.confirm_unseal()
    assert "UNSEAL_NOT_CONFIGURED" in str(excinfo.value)


def test_plan_reader_rejects_bad_rows(tmp_path):
    bad = tmp_path / "p.jsonl"
    bad.write_text('{"campaign_id": "c", "phase": "R3"}\n', encoding="utf-8")  # no arm
    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(bad)
    assert "PLAN_CORRUPT" in str(excinfo.value)

    bad.write_text('{"record_type": "plan_header"}\n', encoding="utf-8")
    assert read_run_plan(bad) == {"counts": {}, "keys": set()}  # header-only is valid

    bad.write_text('{"campaign_id": "c", "phase": "R3", "arm": "F0", "fold": "x", "seed": 17}\n', encoding="utf-8")
    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(bad)
    assert "PLAN_CORRUPT" in str(excinfo.value)

    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(tmp_path / "absent.jsonl")
    assert "PLAN_MISSING" in str(excinfo.value)
