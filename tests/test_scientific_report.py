"""Offline tests for todo 30 — stratified comparison + version lock + report.

Locks the todo-30 acceptance surface (design §13.4/§12, 裁决记录 §7.5):

* the assembled ScientificReport carries the full §13.4 per-cohort field set
  (N total / review-pending refusals / compilable / executed / numerically
  usable / target-path compatible / strict TS / core-hours / unit-success
  cost / failure-code histogram) plus the five comparison arms under one
  fixed shared budget;
* population guard: full-population (183,460 scan_ready) work is refused at
  L1–L3 and at L4 without a VersionLock; allowed at L4 with the lock
  (guard logic only — the population is never executed here);
* valid-tuning guard: a parameter change after a valid-touching run raises;
  no change (or no valid touch) passes;
* demo24 scope label: the 24-reaction cohort must carry
  ``scope="development_mechanism_coverage"``, never ``population_estimate``;
* candidate vs reaction denominators stay separate (candidate success never
  masquerades as reaction success);
* truth_assisted_p1 declaration: every input provenance block states
  ``mapping_provenance`` explicitly;
* the version lock freezes strategy versions + config — the digest changes
  when either changes;
* determinism: identical inputs produce byte-identical report JSON and an
  identical lock digest.

Every fixture is synthetic/in-memory; no test reads the real ``data/`` tree.
"""

from __future__ import annotations

from typing import Any

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.scientific_report import (
    ARM_COUPLED_1D,
    ARM_GRAPH_SINGLE_B,
    ARM_OLD_SINGLE_B,
    ARM_PATH_NEB,
    ARM_TIMING_DIRECTION,
    COHORT_DEMO24,
    COHORT_FULL_POPULATION,
    COHORT_TRAIN16,
    COHORT_VALID8,
    COMPARISON_ARMS,
    DENOMINATOR_NOTE,
    N_DEMO_TOTAL,
    N_TRAIN_DEMO,
    N_VALID_DEMO,
    OBJECT_SCIENTIFIC_REPORT,
    OBJECT_VERSION_LOCK,
    POPULATION_SCAN_READY_N,
    SCOPE_DEVELOPMENT_MECHANISM_COVERAGE,
    SCOPE_POPULATION_ESTIMATE,
    SCHEMA_SCIENTIFIC_REPORT,
    SCHEMA_VERSION_LOCK,
    STAGE_L1,
    STAGE_L2,
    STAGE_L3,
    STAGE_L4,
    ArmResult,
    CohortResult,
    InputProvenance,
    RunLogEntry,
    ScientificReportError,
    SharedBudget,
    VersionLock,
    assert_population_unlocked,
    assert_valid_not_tuned,
    build_scientific_report,
    default_strategy_versions,
    dumps_scientific_report,
    lock_version,
)
from pes2ts_core.utils.hashing import stable_json_dumps

FORBIDDEN = (
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
    | {"endpoint_match", "orientation", "irc_evidence"}
)

_SHA = "a" * 64


# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------
def _budget(**overrides: Any) -> SharedBudget:
    base: dict[str, Any] = {
        "max_attempts": 2,
        "max_cpu_hours": 240.0,
        "max_wall_seconds": 7200,
        "max_total_candidates_scan": 6,
        "max_total_candidates_path_neb": 1,
    }
    base.update(overrides)
    return SharedBudget(**base)


def _demo24_cohort(**overrides: Any) -> CohortResult:
    base: dict[str, Any] = {
        "cohort_id": COHORT_DEMO24,
        "scope": SCOPE_DEVELOPMENT_MECHANISM_COVERAGE,
        "n_total": N_DEMO_TOTAL,
        "n_review_pending_refusals": 10,
        "n_compilable": 8,
        "n_executed": 6,
        "n_numerically_usable": 5,
        "n_target_path_compatible": 3,
        "n_strict_ts": 1,
        "n_candidates_total": 12,
        "n_candidates_target_path_compatible": 4,
        "core_hours": 12.5,
        "failure_code_histogram": {"G2_ENDPOINT_NOT_REACHED": 2, "BACKEND_SMOKE_REQUIRED": 1},
    }
    base.update(overrides)
    return CohortResult(**base)


def _train16_cohort(**overrides: Any) -> CohortResult:
    base: dict[str, Any] = {
        "cohort_id": COHORT_TRAIN16,
        "scope": "train_development",
        "n_total": N_TRAIN_DEMO,
        "n_review_pending_refusals": 2,
        "n_compilable": 12,
        "n_executed": 10,
        "n_numerically_usable": 8,
        "n_target_path_compatible": 4,
        "n_strict_ts": 1,
        "n_candidates_total": 20,
        "n_candidates_target_path_compatible": 5,
        "core_hours": 40.0,
        "failure_code_histogram": {"SCF_NOT_CONVERGED": 2},
    }
    base.update(overrides)
    return CohortResult(**base)


def _valid8_cohort(**overrides: Any) -> CohortResult:
    base: dict[str, Any] = {
        "cohort_id": COHORT_VALID8,
        "scope": "valid_locked",
        "n_total": N_VALID_DEMO,
        "n_review_pending_refusals": 0,
        "n_compilable": 6,
        "n_executed": 5,
        "n_numerically_usable": 4,
        "n_target_path_compatible": 2,
        "n_strict_ts": 0,
        "n_candidates_total": 8,
        "n_candidates_target_path_compatible": 2,
        "core_hours": 20.0,
        "failure_code_histogram": {},
        "consumes_valid": True,
        "valid_used_at": "2026-10-01T00:00:00+00:00",
        "valid_input_set_sha256": _SHA,
    }
    base.update(overrides)
    return CohortResult(**base)


def _population_cohort(**overrides: Any) -> CohortResult:
    base: dict[str, Any] = {
        "cohort_id": COHORT_FULL_POPULATION,
        "scope": SCOPE_POPULATION_ESTIMATE,
        "n_total": POPULATION_SCAN_READY_N,
        "n_review_pending_refusals": 0,
        "n_compilable": POPULATION_SCAN_READY_N,
        "n_executed": 0,
        "n_numerically_usable": 0,
        "n_target_path_compatible": 0,
        "n_strict_ts": 0,
        "core_hours": 0.0,
        "failure_code_histogram": {},
    }
    base.update(overrides)
    return CohortResult(**base)


def _arm(
    arm_id: str,
    cohorts: tuple[CohortResult, ...],
    *,
    budget: SharedBudget | None = None,
    core_hours: float = 10.0,
) -> ArmResult:
    return ArmResult(
        arm_id=arm_id,
        budget=budget if budget is not None else _budget(),
        cohorts=cohorts,
        core_hours=core_hours,
    )


def _config() -> dict[str, Any]:
    return {"scan_strategy": {"context_radius": 2, "schedule_budget": 3}}


def _two_arm_inputs() -> tuple[list[ArmResult], dict[str, Any]]:
    arms = [
        _arm(ARM_OLD_SINGLE_B, (_demo24_cohort(),), core_hours=12.5),
        _arm(ARM_GRAPH_SINGLE_B, (_demo24_cohort(),), core_hours=18.0),
    ]
    return arms, _config()


def _forbidden_hits(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN:
                hits.append(f"{path}.{key}")
            hits.extend(_forbidden_hits(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_forbidden_hits(item, f"{path}[{index}]"))
    return hits


# ---------------------------------------------------------------------------
# Full-field report (§13.4).
# ---------------------------------------------------------------------------
def test_report_carries_full_13_4_field_set() -> None:
    """Given two arms under one budget, when the report is built, then every
    §13.4 per-cohort field and both denominator families are present."""
    arms, config = _two_arm_inputs()
    doc = build_scientific_report(arms, config, report_id="r-full", stage=STAGE_L1)

    assert doc["schema_name"] == OBJECT_SCIENTIFIC_REPORT
    assert doc["schema_version"] == SCHEMA_SCIENTIFIC_REPORT
    assert doc["object_id"] == "r-full"
    assert doc["stage"] == STAGE_L1
    assert doc["status"] == "assembled"

    # Five comparison arms are registered as data; the report carries the
    # arm results actually assembled plus the full comparison vocabulary.
    assert doc["comparison_arms"][0]["arm_id"] == ARM_OLD_SINGLE_B
    assert {row["arm_id"] for row in doc["comparison_arms"]} == set(COMPARISON_ARMS)
    assert [row["arm_id"] for row in doc["arms"]] == [ARM_OLD_SINGLE_B, ARM_GRAPH_SINGLE_B]

    required_cohort_fields = {
        "cohort_id",
        "scope",
        "n_total",
        "n_review_pending_refusals",
        "n_compilable",
        "n_executed",
        "n_numerically_usable",
        "n_target_path_compatible",
        "n_strict_ts",
        "core_hours",
        "unit_success_cost",
        "failure_code_histogram",
        "n_candidates_total",
        "n_candidates_target_path_compatible",
        "reaction_target_path_rate",
        "candidate_target_path_rate",
        "consumes_valid",
        "valid_used_at",
        "valid_input_set_sha256",
    }
    for arm in doc["arms"]:
        assert set(arm["cohorts"][0]) >= required_cohort_fields
        cohort = arm["cohorts"][0]
        assert cohort["n_total"] == N_DEMO_TOTAL
        assert cohort["n_review_pending_refusals"] == 10
        assert cohort["n_compilable"] == 8
        assert cohort["n_executed"] == 6
        assert cohort["n_numerically_usable"] == 5
        assert cohort["n_target_path_compatible"] == 3
        assert cohort["n_strict_ts"] == 1
        assert cohort["core_hours"] == 12.5
        assert cohort["unit_success_cost"] == pytest.approx(12.5 / 3)
        assert cohort["failure_code_histogram"] == {
            "G2_ENDPOINT_NOT_REACHED": 2,
            "BACKEND_SMOKE_REQUIRED": 1,
        }
        # Reaction rate uses the reaction denominator; candidate rate uses
        # the candidate denominator — never mixed.
        assert cohort["reaction_target_path_rate"] == pytest.approx(3 / 24)
        assert cohort["candidate_target_path_rate"] == pytest.approx(4 / 12)

        # Arm-level: new target-path rate + cost under the fixed budget.
        assert arm["new_target_path_rate"] == pytest.approx(3 / 24)
        assert arm["core_hours"] in (12.5, 18.0)
        assert arm["budget"]["max_attempts"] == 2
        assert arm["shared_budget_key"] == doc["shared_budget_key"]
        assert arm["denominator_note"] == DENOMINATOR_NOTE

    # Shared budget block + provenance + valid discipline + population note.
    assert doc["shared_budget"]["max_total_candidates_scan"] == 6
    assert doc["input_provenance"][0]["mapping_provenance"] == "truth_assisted_p1"
    assert doc["valid_discipline"]["valid_used"] is False
    assert doc["population"]["scan_ready_n"] == POPULATION_SCAN_READY_N
    assert doc["population"]["full_population_unlocked"] is False
    # Demo24 block carries the coverage declaration (special_domain source).
    assert doc["demo24"]["scope"] == SCOPE_DEVELOPMENT_MECHANISM_COVERAGE
    assert doc["demo24"]["n_total"] == N_DEMO_TOTAL
    assert "C" in doc["demo24"]["coverage"]["domains_with_evidence"]
    # Digest present and stable under seal semantics.
    assert isinstance(doc["content_sha256"], str) and len(doc["content_sha256"]) == 64


def test_report_dumps_roundtrip_and_purity() -> None:
    """Given an assembled report, when it is serialized, then the digest
    checks and no forbidden truth-derived key appears anywhere."""
    arms, config = _two_arm_inputs()
    doc = build_scientific_report(arms, config, report_id="r-purity")
    payload = dumps_scientific_report(doc)
    assert stable_json_dumps(doc) == payload
    assert _forbidden_hits(doc) == []


# ---------------------------------------------------------------------------
# Population guard (L1–L3 locked; L4 requires the version lock).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", [STAGE_L1, STAGE_L2, STAGE_L3])
def test_population_guard_refuses_before_l4(stage: str) -> None:
    """Given stage L1/L2/L3, when full-population work is attempted, then a
    typed refusal POPULATION_LOCKED_BEFORE_L4 is raised."""
    with pytest.raises(ScientificReportError) as excinfo:
        assert_population_unlocked(stage)
    assert excinfo.value.code == "POPULATION_LOCKED_BEFORE_L4"


def test_population_guard_refuses_l4_without_lock() -> None:
    """Given stage L4 with no lock record, when full-population work is
    attempted, then a typed refusal POPULATION_LOCK_REQUIRED is raised."""
    with pytest.raises(ScientificReportError) as excinfo:
        assert_population_unlocked(STAGE_L4)
    assert excinfo.value.code == "POPULATION_LOCK_REQUIRED"


def test_population_guard_allows_l4_with_lock() -> None:
    """Given stage L4 and a VersionLock, when the guard runs, then it passes
    and returns the lock (guard logic only; population never executed)."""
    lock = lock_version("r-pop", strategy_versions={"registry_version": "strategy_registry_v1"})
    returned = assert_population_unlocked(STAGE_L4, lock=lock)
    assert isinstance(returned, VersionLock)
    assert returned.report_id == "r-pop"


def test_population_guard_rejects_unknown_stage() -> None:
    """Given an unknown stage token, when the guard runs, then STAGE_UNKNOWN."""
    with pytest.raises(ScientificReportError) as excinfo:
        assert_population_unlocked("L9")
    assert excinfo.value.code == "STAGE_UNKNOWN"


def test_report_population_estimate_requires_l4_lock() -> None:
    """Given a population-estimate cohort at L1, when the report is built,
    then the refusal propagates; at L4 with a lock the report unlocks."""
    population_arm = _arm(ARM_GRAPH_SINGLE_B, (_population_cohort(),))
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report([population_arm], _config(), stage=STAGE_L1)
    assert excinfo.value.code == "POPULATION_LOCKED_BEFORE_L4"

    lock = lock_version("r-pop-report")
    doc = build_scientific_report(
        [population_arm], _config(), stage=STAGE_L4, population_lock=lock
    )
    assert doc["population"]["claims_population_estimate"] is True
    assert doc["population"]["full_population_unlocked"] is True
    assert doc["population"]["population_lock"]["content_sha256"] == lock.content_sha256


# ---------------------------------------------------------------------------
# Valid-tuning guard (16 train / 8 valid discipline).
# ---------------------------------------------------------------------------
def _run_log(*, tuned: bool) -> list[RunLogEntry]:
    entries = [
        RunLogEntry(
            run_id="run-train-1",
            timestamp="2026-10-01T01:00:00+00:00",
            split="train",
            parameters={"context_radius": 2, "schedule_budget": 3},
        ),
        RunLogEntry(
            run_id="run-valid-1",
            timestamp="2026-10-01T02:00:00+00:00",
            split="valid",
            parameters={"context_radius": 2, "schedule_budget": 3},
            input_set_sha256=_SHA,
        ),
    ]
    if tuned:
        entries.append(
            RunLogEntry(
                run_id="run-train-2",
                timestamp="2026-10-01T03:00:00+00:00",
                split="train",
                parameters={"context_radius": 3, "schedule_budget": 3},
            )
        )
    else:
        entries.append(
            RunLogEntry(
                run_id="run-train-2",
                timestamp="2026-10-01T03:00:00+00:00",
                split="train",
                parameters={"context_radius": 2, "schedule_budget": 3},
            )
        )
    return entries


def test_valid_tuning_guard_errors_on_parameter_change_after_valid() -> None:
    """Given a parameter change after a valid-touching run, when the guard
    scans the run log, then VALID_TUNED_AFTER_VALID_RUN is raised."""
    with pytest.raises(ScientificReportError) as excinfo:
        assert_valid_not_tuned(_run_log(tuned=True))
    assert excinfo.value.code == "VALID_TUNED_AFTER_VALID_RUN"


def test_valid_tuning_guard_passes_without_parameter_change() -> None:
    """Given unchanged parameters after valid, when the guard scans the run
    log, then it passes."""
    assert_valid_not_tuned(_run_log(tuned=False))


def test_valid_tuning_guard_passes_when_valid_never_touched() -> None:
    """Given a train-only run log, when the guard scans, then it passes."""
    assert_valid_not_tuned(
        [
            RunLogEntry(
                run_id="run-train-1",
                timestamp="2026-10-01T01:00:00+00:00",
                split="train",
                parameters={"context_radius": 2},
            )
        ]
    )


def test_report_valid_use_carries_timestamp_and_input_hash() -> None:
    """Given a cohort consuming valid rows with timestamp + input hash, when
    the report is built, then valid_used=true and both fields are carried."""
    arms = [_arm(ARM_COUPLED_1D, (_valid8_cohort(),), core_hours=20.0)]
    doc = build_scientific_report(arms, _config(), report_id="r-valid")
    assert doc["valid_discipline"]["valid_used"] is True
    assert doc["valid_discipline"]["valid_used_at"] == "2026-10-01T00:00:00+00:00"
    assert doc["valid_discipline"]["valid_input_set_sha256"] == _SHA
    cohort = doc["arms"][0]["cohorts"][0]
    assert cohort["consumes_valid"] is True
    assert cohort["valid_used_at"] == "2026-10-01T00:00:00+00:00"


def test_report_valid_use_without_declaration_is_refused() -> None:
    """Given a valid-consuming cohort missing timestamp/hash, when the report
    is built, then VALID_USE_UNDECLARED is raised."""
    broken = _valid8_cohort(valid_used_at=None, valid_input_set_sha256=None)
    arms = [_arm(ARM_COUPLED_1D, (broken,))]
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report(arms, _config())
    assert excinfo.value.code == "VALID_USE_UNDECLARED"


# ---------------------------------------------------------------------------
# Demo24 scope label (mechanism-coverage DEV set).
# ---------------------------------------------------------------------------
def test_demo24_scope_population_estimate_is_refused() -> None:
    """Given a demo24 cohort labelled population_estimate, when the report is
    built, then DEMO24_SCOPE_MISMATCH is raised."""
    bad = _demo24_cohort(scope=SCOPE_POPULATION_ESTIMATE)
    arms = [_arm(ARM_OLD_SINGLE_B, (bad,))]
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report(arms, _config())
    assert excinfo.value.code == "DEMO24_SCOPE_MISMATCH"


def test_demo24_scope_development_mechanism_coverage_accepted() -> None:
    """Given a demo24 cohort labelled development_mechanism_coverage, when
    the report is built, then the scope label is carried verbatim."""
    arms = [_arm(ARM_OLD_SINGLE_B, (_demo24_cohort(),))]
    doc = build_scientific_report(arms, _config(), report_id="r-demo-scope")
    cohort = doc["arms"][0]["cohorts"][0]
    assert cohort["scope"] == SCOPE_DEVELOPMENT_MECHANISM_COVERAGE
    assert doc["demo24"]["scope"] == SCOPE_DEVELOPMENT_MECHANISM_COVERAGE


def test_full_population_cohort_exact_n_enforced() -> None:
    """Given a full_population cohort with the wrong N, when the report is
    built, then COHORT_INVALID is raised (exact population count discipline)."""
    bad = _population_cohort(n_total=1000)
    arms = [_arm(ARM_GRAPH_SINGLE_B, (bad,))]
    lock = lock_version("r-bad-pop")
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report([arms[0]], _config(), stage=STAGE_L4, population_lock=lock)
    assert excinfo.value.code == "COHORT_INVALID"


# ---------------------------------------------------------------------------
# Candidate vs reaction denominators stay separate.
# ---------------------------------------------------------------------------
def test_candidate_success_never_masquerades_as_reaction_success() -> None:
    """Given arm data where candidate successes differ from reaction
    successes, when the report is built, then the two denominators and the
    two rates stay separate and the arm rate uses reactions only."""
    cohort = _demo24_cohort(
        n_target_path_compatible=3,
        n_candidates_target_path_compatible=9,
        n_candidates_total=12,
    )
    arms = [_arm(ARM_TIMING_DIRECTION, (cohort,), core_hours=12.0)]
    doc = build_scientific_report(arms, _config(), report_id="r-denom")
    arm = doc["arms"][0]
    assert arm["n_reactions_target_path_compatible"] == 3
    assert arm["n_candidates_target_path_compatible"] == 9
    assert arm["new_target_path_rate"] == pytest.approx(3 / 24)
    assert arm["candidate_target_path_rate"] == pytest.approx(9 / 12)
    assert arm["new_target_path_rate"] != arm["candidate_target_path_rate"]
    # Unit-success cost is reaction-based: core-hours per reaction success.
    assert arm["unit_success_cost"] == pytest.approx(12.0 / 3)
    assert "never substitute" in arm["denominator_note"]


def test_reaction_funnel_violation_is_refused() -> None:
    """Given n_strict_ts above n_target_path_compatible, when the report is
    built, then DENOMINATOR_INVALID is raised."""
    bad = _demo24_cohort(n_strict_ts=9)
    arms = [_arm(ARM_OLD_SINGLE_B, (bad,))]
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report(arms, _config())
    assert excinfo.value.code == "DENOMINATOR_INVALID"


# ---------------------------------------------------------------------------
# truth_assisted_p1 declaration.
# ---------------------------------------------------------------------------
def test_truth_assisted_p1_declaration_carried_by_default() -> None:
    """Given no explicit provenance input, when the report is built, then the
    default block declares mapping_provenance=truth_assisted_p1."""
    arms, config = _two_arm_inputs()
    doc = build_scientific_report(arms, config, report_id="r-truth-default")
    assert doc["input_provenance"][0]["mapping_provenance"] == "truth_assisted_p1"


def test_truth_assisted_p1_declaration_carried_from_input() -> None:
    """Given an explicit provenance block, when the report is built, then the
    declared mapping_provenance is carried verbatim."""
    arms, config = _two_arm_inputs()
    provenance = [
        InputProvenance(
            source="demo24_bundle_snapshot",
            mapping_provenance="truth_assisted_p1",
            input_set_sha256=_SHA,
            note="declared from bundle docs",
        )
    ]
    doc = build_scientific_report(
        arms, config, report_id="r-truth-input", input_provenance=provenance
    )
    block = doc["input_provenance"][0]
    assert block["mapping_provenance"] == "truth_assisted_p1"
    assert block["source"] == "demo24_bundle_snapshot"
    assert block["input_set_sha256"] == _SHA


def test_missing_mapping_provenance_is_refused() -> None:
    """Given a provenance block without mapping_provenance, when the report
    is built, then PROVENANCE_MAPPING_UNDECLARED is raised."""
    arms, config = _two_arm_inputs()
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report(
            arms,
            config,
            input_provenance=[{"source": "some_bundle"}],
        )
    assert excinfo.value.code == "PROVENANCE_MAPPING_UNDECLARED"


# ---------------------------------------------------------------------------
# Version lock freezes versions + config.
# ---------------------------------------------------------------------------
def test_lock_version_digest_changes_when_versions_change() -> None:
    """Given the same config but different strategy versions, when the lock
    is built, then the digests differ."""
    config = _config()
    lock_a = lock_version(
        "r-lock", strategy_versions={"registry_version": "strategy_registry_v1"}, config=config
    )
    lock_b = lock_version(
        "r-lock", strategy_versions={"registry_version": "strategy_registry_v2"}, config=config
    )
    assert lock_a.content_sha256 != lock_b.content_sha256


def test_lock_version_digest_changes_when_config_changes() -> None:
    """Given the same versions but different config, when the lock is built,
    then the digests differ."""
    versions = default_strategy_versions()
    lock_a = lock_version("r-lock", strategy_versions=versions, config={"a": 1})
    lock_b = lock_version("r-lock", strategy_versions=versions, config={"a": 2})
    assert lock_a.content_sha256 != lock_b.content_sha256


def test_lock_version_deterministic_for_identical_inputs() -> None:
    """Given identical versions + config, when the lock is built twice, then
    the digests are equal."""
    versions = default_strategy_versions()
    config = _config()
    lock_a = lock_version("r-lock", strategy_versions=versions, config=config)
    lock_b = lock_version("r-lock", strategy_versions=versions, config=config)
    assert lock_a.content_sha256 == lock_b.content_sha256
    assert lock_a.to_doc()["schema_version"] == SCHEMA_VERSION_LOCK
    assert lock_a.to_doc()["schema_name"] == OBJECT_VERSION_LOCK
    # Default versions come from the live scan-strategy modules.
    assert lock_a.strategy_versions["registry_version"] == "strategy_registry_v1"
    assert lock_a.strategy_versions["mapping_provenance"] == "truth_assisted_p1"


def test_lock_version_rejects_empty_versions() -> None:
    """Given empty strategy versions, when the lock is built, then
    LOCK_INPUT_INVALID is raised."""
    with pytest.raises(ScientificReportError) as excinfo:
        lock_version("r-lock", strategy_versions={})
    assert excinfo.value.code == "LOCK_INPUT_INVALID"


# ---------------------------------------------------------------------------
# Determinism + shared-budget discipline + arm vocabulary.
# ---------------------------------------------------------------------------
def test_report_determinism_byte_identical() -> None:
    """Given identical inputs, when the report is built twice with a pinned
    created_at, then the serialized JSON is byte-identical."""
    arms, config = _two_arm_inputs()
    doc_a = build_scientific_report(
        arms, config, report_id="r-det", created_at="2026-10-01T00:00:00+00:00"
    )
    doc_b = build_scientific_report(
        arms, config, report_id="r-det", created_at="2026-10-01T00:00:00+00:00"
    )
    assert stable_json_dumps(doc_a) == stable_json_dumps(doc_b)
    assert doc_a["content_sha256"] == doc_b["content_sha256"]


def test_report_budget_mismatch_is_refused() -> None:
    """Given two arms with different budgets, when the report is built, then
    BUDGET_MISMATCH is raised (fixed shared budget discipline)."""
    arms = [
        _arm(ARM_OLD_SINGLE_B, (_demo24_cohort(),), budget=_budget(max_attempts=2)),
        _arm(ARM_GRAPH_SINGLE_B, (_demo24_cohort(),), budget=_budget(max_attempts=5)),
    ]
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report(arms, _config())
    assert excinfo.value.code == "BUDGET_MISMATCH"


def test_unknown_arm_id_is_refused() -> None:
    """Given an unknown arm id, when the report is built, then ARM_UNKNOWN."""
    bad = _arm("made_up_arm", (_demo24_cohort(),))
    with pytest.raises(ScientificReportError) as excinfo:
        build_scientific_report([bad], _config())
    assert excinfo.value.code == "ARM_UNKNOWN"


def test_all_five_comparison_arms_accepted_under_shared_budget() -> None:
    """Given one cohort per comparison arm under one budget, when the report
    is built, then all five arms appear with rate + cost fields."""
    budget = _budget()
    arms = [
        _arm(ARM_OLD_SINGLE_B, (_demo24_cohort(),), budget=budget, core_hours=10.0),
        _arm(ARM_GRAPH_SINGLE_B, (_demo24_cohort(),), budget=budget, core_hours=11.0),
        _arm(ARM_COUPLED_1D, (_demo24_cohort(),), budget=budget, core_hours=12.0),
        _arm(ARM_TIMING_DIRECTION, (_demo24_cohort(),), budget=budget, core_hours=13.0),
        _arm(ARM_PATH_NEB, (_demo24_cohort(),), budget=budget, core_hours=14.0),
    ]
    doc = build_scientific_report(arms, _config(), report_id="r-arms", stage=STAGE_L1)
    assert [row["arm_id"] for row in doc["arms"]] == list(COMPARISON_ARMS)
    for row in doc["arms"]:
        assert "new_target_path_rate" in row
        assert "core_hours" in row
        assert "unit_success_cost" in row
        assert row["shared_budget_key"] == doc["shared_budget_key"]
    neb_row = doc["arms"][-1]
    assert neb_row["arm_id"] == ARM_PATH_NEB
    assert neb_row["method_channel"] == "NEB"
    assert "native scan branch" in neb_row["channel_note"]
