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
    ExploratoryOnlyError,
    IllegalTransition,
    LedgerError,
    PlanNotFrozenError,
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
    # PD-4 partial value is valid schema too (still no evidence claim)
    validate(make_formal(admissibility="not_admissible_exploratory", evidence_sha=None))


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
    header = {
        "record_type": "plan_header",
        "campaign_id": CAMPAIGN,
        "pd3_note": "x",
        "campaign_mode": "full",
    }
    lines = [json.dumps(header)]
    lines.extend(json.dumps(r) for r in rows)
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return path


def synthetic_plan_row(**overrides):
    """A frozen (real-sha) planned row whose run_id matches make_formal()."""
    row = {
        "campaign_id": CAMPAIGN,
        "phase": "R3",
        "substage": "F-main",
        "arm": "F0",
        "fold": 0,
        "seed": 17,
        "variant": None,
        "reuses": [],
        "status": "planned",
        "data_sha256": "a1" * 32,
        "teacher_sha256": "b2" * 32,
        "features_sha256": "c3" * 32,
        "split_sha256": "d4" * 32,
        "config_sha256": "e5" * 32,
    }
    row.update(overrides)
    if "run_id" not in overrides:
        row["run_id"] = compute_run_id(row, scheme="hashed")
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


#: identity shas that make plan run_ids match make_formal()/make_record()
FULL_SHAS = {
    "config_sha256": "e5" * 32,
    "data_sha256": "a1" * 32,
    "teacher_sha256": "b2" * 32,
    "features_sha256": "c3" * 32,
    "split_sha256": "d4" * 32,
}


def test_real_run_plan_binds_f_main_launches(tmp_path):
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "run_plan.jsonl"
    write_plan(plan, shas=FULL_SHAS)  # stage g0/launch: all five shas frozen
    led = RunLedger(
        tmp_path / "r.jsonl", plan_path=plan, caps={"R3": 66, "V5.2": 33}
    )
    led.init()
    # three in-plan F-main launches (real 66-run plan rows; variant must match)
    for seed in (17, 29, 43):
        rec = make_formal(arm="F0", fold=0, seed=seed, variant="forward")
        led.append(rec)
        assert led.append(launch(rec)).action == "transitioned"
    # V5.2 QT row from the plan also binds (variant None on both sides)
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
    bad = make_formal(arm="F0", fold=0, seed=99, variant="forward")
    led.append(bad)
    with pytest.raises(BudgetCapExceeded) as excinfo:
        led.append(launch(bad))
    assert "NOT_IN_PLAN" in str(excinfo.value)


def test_plan_not_frozen_rejects_pending_plan_launch(tmp_path):
    """Default (stage-e0) plan still carries PENDING shas -> launch refused."""
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "run_plan.jsonl"
    write_plan(plan)  # config real, data/teacher/features/split = PENDING_*
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()
    rec = make_formal(arm="F0", fold=0, seed=17, variant="forward")
    led.append(rec)  # registration is fine
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(launch(rec))
    assert "PLAN_NOT_FROZEN" in str(excinfo.value)
    assert "PENDING" in str(excinfo.value)


def test_run_id_mismatch_same_key_different_shas_refused(tmp_path):
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "run_plan.jsonl"
    write_plan(plan, shas=FULL_SHAS)
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()
    # same (campaign, phase, arm, fold, seed, variant) but a different config
    # sha -> different run_id; the plan is frozen so this is RUN_ID_MISMATCH
    rec = make_formal(arm="F0", fold=0, seed=17, variant="forward", config_sha256="ff" * 32)
    led.append(rec)
    with pytest.raises(PlanNotFrozenError) as excinfo:
        led.append(launch(rec))
    assert "RUN_ID_MISMATCH" in str(excinfo.value)
    assert "PLAN_NOT_FROZEN" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# one-shot unseal tokens: BATCH semantics (ORA-2 blocker 3)
# ---------------------------------------------------------------------------


def test_is_confirm_run_detection_is_explicit_flag_only():
    # the explicit confirm flag is the ONLY trigger (no arm/variant heuristics)
    assert is_confirm_run(make_formal(confirm=True, fold=3)) is True
    assert is_confirm_run(make_formal(confirm=True, fold=4)) is True
    assert is_confirm_run(make_formal(arm="C", fold=3)) is False
    assert is_confirm_run(make_formal(arm="QT", variant="confirm", fold=4)) is False
    assert is_confirm_run(make_formal(arm="C", fold=3, confirm=False)) is False
    assert is_confirm_run(make_formal(arm="B", fold=3)) is False  # sealed eval != confirm
    with pytest.raises(LedgerError) as excinfo:
        validate(make_formal(confirm="yes"))
    assert "INVALID_CONFIRM" in str(excinfo.value)


def test_confirm_unseal_solo_batch_one_shot(tmp_path):
    unseal = tmp_path / "unseal.json"
    led = RunLedger(tmp_path / "r.jsonl", unseal_path=unseal)
    led.init()

    first = make_formal(arm="C", fold=3, seed=17, confirm=True)
    led.append(first)
    assert led.append(launch(first)).action == "transitioned"
    token = json.loads(unseal.read_text(encoding="utf-8"))
    first_id = compute_run_id(first)
    assert token["kind"] == "confirm"
    assert token["batch_id"] == "solo-confirm-{}".format(first_id)
    assert token["allowed_run_ids"] == [first_id]
    assert token["count"] == 1
    assert token["consumed_at"]
    assert token["run_id"] == first_id

    # same-run retry: token already ours -> idempotent, never re-incremented
    led.append(fail(first))
    led.append(launch(first))
    assert json.loads(unseal.read_text(encoding="utf-8"))["count"] == 1

    # a DIFFERENT confirm run = a second, different batch -> refused
    second = make_formal(arm="C", fold=4, seed=29, confirm=True)
    led.append(second)
    before = len(read_lines(led.path))
    with pytest.raises(UnsealExceeded) as excinfo:
        led.append(launch(second))
    assert "UNSEAL_BATCH_MISMATCH" in str(excinfo.value)
    assert len(read_lines(led.path)) == before
    assert json.loads(unseal.read_text(encoding="utf-8"))["count"] == 1


def test_confirm_unseal_plan_batch_covers_all_registered_runs(tmp_path):
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "run_plan.jsonl"
    write_plan(plan, shas=FULL_SHAS)
    unseal = tmp_path / "unseal.json"
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan, unseal_path=unseal)
    led.init()

    # bound plan registers C folds3/4 (6) + QT-confirm (6) = ONE batch of 12
    c1 = make_formal(arm="C", fold=3, seed=17, confirm=True)
    led.append(c1)
    assert led.append(launch(c1)).action == "transitioned"
    token = json.loads(unseal.read_text(encoding="utf-8"))
    assert token["batch_id"] == "plan-confirm-batch"
    assert token["count"] == 1
    assert len(token["allowed_run_ids"]) == 12
    assert compute_run_id(c1) in token["allowed_run_ids"]

    # every other pre-registered batch member is admissible (no re-burn)
    c2 = make_formal(arm="C", fold=4, seed=29, confirm=True)
    led.append(c2)
    assert led.append(launch(c2)).action == "transitioned"
    qt = make_formal(
        phase="V5.2", arm="QT", variant="confirm", fold=3, seed=17, confirm=True
    )
    led.append(qt)
    assert led.append(launch(qt)).action == "transitioned"
    token = json.loads(unseal.read_text(encoding="utf-8"))
    assert token["count"] == 1  # one consumption covers the whole batch

    # record/plan confirm-flag disagreement refuses fail-closed
    clash = make_formal(arm="C", fold=3, seed=29, confirm=False)
    led.append(clash)
    with pytest.raises(LedgerError) as excinfo:
        led.append(launch(clash))
    assert "CONFIRM_FLAG_MISMATCH" in str(excinfo.value)


def test_confirm_run_without_unseal_config_refused(tmp_path):
    led = RunLedger(tmp_path / "r.jsonl")  # unseal_path=None (fail-closed default)
    led.init()
    rec = make_formal(arm="C", fold=3, seed=17, confirm=True)
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

    unseal.write_text(
        json.dumps({"count": "many", "batch_id": "b", "allowed_run_ids": ["R-x"]}),
        encoding="utf-8",
    )
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="R-x")
    assert "UNSEAL_CORRUPT" in str(excinfo.value)

    # consumed by a DIFFERENT pre-registered batch -> refuse (second batch)
    unseal.write_text(
        json.dumps({"count": 7, "batch_id": "intruder", "allowed_run_ids": ["someone-else"]}),
        encoding="utf-8",
    )
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="R-x")
    assert "UNSEAL_BATCH_MISMATCH" in str(excinfo.value)

    # same registered batch but the asking run is outside it -> refuse
    unseal.write_text(
        json.dumps({"count": 1, "batch_id": "b", "allowed_run_ids": ["R-x"]}),
        encoding="utf-8",
    )
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="intruder", allowed_run_ids=["R-x"])
    assert "UNSEAL_OUTSIDE_BATCH" in str(excinfo.value)

    # first use cannot OPEN a batch it is not registered for (pre-registration)
    unseal.unlink()
    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(unseal, run_id="R-x", allowed_run_ids=["R-y"])
    assert "UNSEAL_OUTSIDE_BATCH" in str(excinfo.value)

    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(None, run_id="R-x")
    assert "UNSEAL_NOT_CONFIGURED" in str(excinfo.value)

    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(tmp_path / "fresh.json", run_id=None, allowed_run_ids=[])
    assert "UNSEAL_CORRUPT" in str(excinfo.value)

    with pytest.raises(UnsealExceeded) as excinfo:
        confirm_unseal(tmp_path / "fresh.json")
    assert "UNSEAL_CORRUPT" in str(excinfo.value)


def test_metric_unseal_is_a_separate_one_shot_token(tmp_path):
    unseal = tmp_path / "unseal.json"
    metric = tmp_path / "metric_unseal.json"
    led = RunLedger(
        tmp_path / "r.jsonl", unseal_path=unseal, metric_unseal_path=metric
    )
    led.init()
    out = led.metric_unseal(
        run_id="R-b-fold3", allowed_run_ids=["R-b-fold3"], batch_id="B-metrics"
    )
    assert out["kind"] == "metric"
    assert out["count"] == 1
    assert out["batch_id"] == "B-metrics"
    token = json.loads(metric.read_text(encoding="utf-8"))
    assert token["allowed_run_ids"] == ["R-b-fold3"]
    assert not unseal.exists()  # the confirm token is untouched

    # second, different metric batch -> refused (B-stage reveal is one-shot)
    with pytest.raises(UnsealExceeded) as excinfo:
        led.metric_unseal(run_id="R-other", allowed_run_ids=["R-other"], batch_id="B2")
    assert "UNSEAL_BATCH_MISMATCH" in str(excinfo.value)

    # same registered batch member stays admissible
    assert led.metric_unseal(
        run_id="R-b-fold3", allowed_run_ids=["R-b-fold3"], batch_id="B-metrics"
    )["count"] == 1

    plain = RunLedger(tmp_path / "r2.jsonl")
    plain.init()
    with pytest.raises(UnsealExceeded) as excinfo:
        plain.metric_unseal(run_id="x", allowed_run_ids=["x"])
    assert "METRIC_UNSEAL_NOT_CONFIGURED" in str(excinfo.value)


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

    # missing run_id / duplicate run_id are corrupt (identity binding, ORA-2/2)
    bad.write_text(
        '{"campaign_id": "c", "phase": "R3", "arm": "F0", "fold": 0, "seed": 17}\n',
        encoding="utf-8",
    )
    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(bad)
    assert "PLAN_CORRUPT" in str(excinfo.value)
    row = synthetic_plan_row()
    bad.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(bad)
    assert "duplicate run_id" in str(excinfo.value)

    # header-only is a valid (empty) full plan
    bad.write_text('{"record_type": "plan_header", "campaign_mode": "full"}\n', encoding="utf-8")
    result = read_run_plan(bad)
    assert result["counts"] == {} and result["keys"] == set()
    assert result["mode"] == "full" and result["confirm_run_ids"] == []

    bad.write_text('{"campaign_id": "c", "phase": "R3", "arm": "F0", "fold": "x", "seed": 17}\n', encoding="utf-8")
    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(bad)
    assert "PLAN_CORRUPT" in str(excinfo.value)

    # invalid campaign_mode refuses (fail-closed)
    bad.write_text(
        '{"record_type": "plan_header", "campaign_mode": "halfway"}\n', encoding="utf-8"
    )
    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(bad)
    assert "PLAN_CORRUPT" in str(excinfo.value)

    with pytest.raises(LedgerError) as excinfo:
        read_run_plan(tmp_path / "absent.jsonl")
    assert "PLAN_MISSING" in str(excinfo.value)


# ---------------------------------------------------------------------------
# PD-4 partial (21-run exploratory) campaign enforcement
# ---------------------------------------------------------------------------


def test_partial_plan_forces_exploratory_admissibility(tmp_path):
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "partial.jsonl"
    write_plan(plan, mode="partial", shas=FULL_SHAS)
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()

    rec = make_formal(arm="B", fold=0, seed=17, admissibility="admissible")
    led.append(rec)  # registration alone is not a claim of admissibility
    with pytest.raises(ExploratoryOnlyError) as excinfo:
        led.append(launch(rec))
    assert "NOT_ADMISSIBLE_EXPLORATORY" in str(excinfo.value)

    # corrected record: the plan FORCES not_admissible_exploratory
    fixed = make_formal(
        arm="B", fold=0, seed=17, admissibility="not_admissible_exploratory"
    )
    assert led.append(launch(fixed)).action == "transitioned"
    entry = led.status(compute_run_id(fixed))
    assert entry["admissibility"] == "not_admissible_exploratory"
    assert len(read_lines(led.path)) == 2  # planned + running only


def test_partial_plan_finalize_refuses_forbidden_keywords(tmp_path):
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "partial.jsonl"
    write_plan(plan, mode="partial", shas=FULL_SHAS)
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()

    def done_record(metrics_path):
        return make_formal(
            arm="B",
            fold=0,
            seed=17,
            admissibility="not_admissible_exploratory",
            status="done",
            checkpoint_update=3000,
            checkpoint_sha256="ab" * 32,
            metrics_path=metrics_path,
        )

    rec = make_formal(arm="B", fold=0, seed=17, admissibility="not_admissible_exploratory")
    led.append(rec)
    led.append(launch(rec))

    for keyword in ("scale_controlled", "V5.1b"):
        bad = done_record(
            "outputs/pesn/30h_campaign/metrics/{}.json".format(keyword)
        )
        with pytest.raises(ExploratoryOnlyError) as excinfo:
            led.append(bad)
        assert "PARTIAL_FINALIZE_FORBIDDEN" in str(excinfo.value)
        assert keyword.lower() in str(excinfo.value).lower()

    clean = done_record("outputs/pesn/30h_campaign/metrics/baseline.json")
    assert led.append(clean).action == "transitioned"
    assert led.status(compute_run_id(clean))["status"] == "done"


def test_full_plan_finalize_still_allows_outcome_keywords(tmp_path):
    """scale_controlled remains a legitimate R3 full-campaign outcome (§4.3)."""
    from scripts.agent_ops.runplan.build_r3_runplan import write_plan

    plan = tmp_path / "run_plan.jsonl"
    write_plan(plan, shas=FULL_SHAS)
    led = RunLedger(tmp_path / "r.jsonl", plan_path=plan)
    led.init()
    rec = make_formal(arm="F0", fold=0, seed=17, variant="forward")
    led.append(rec)
    led.append(launch(rec))
    done = make_formal(
        arm="F0",
        fold=0,
        seed=17,
        variant="forward",
        status="done",
        checkpoint_update=3000,
        checkpoint_sha256="ab" * 32,
        metrics_path="outputs/pesn/30h_campaign/metrics/scale_controlled.json",
    )
    assert led.append(done).action == "transitioned"
