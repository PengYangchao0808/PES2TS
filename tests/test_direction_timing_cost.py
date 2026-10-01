"""Offline tests for todo 28 — direction/timing comparison and cost accounting.

Locks the todo-28 acceptance surface (design §7/§8.3/§13.4):

* the comparison ledger records the four todo-15 fields per candidate
  (``start_endpoint/direction/assembly_id/anchor_reason``) and emits
  forward-vs-reverse / early-vs-late / per-assembly comparisons;
* a reverse candidate without a verified independent start-optimization +
  assembly verification record yields an **incomplete** comparison with
  ``splice_permitted=False`` — curves are never spliced without
  verification (typed refusal via :func:`assert_splice_permitted`);
* NEB and Scan cost/success stay in **separate channels** (todo-26
  vocabulary imported, not duplicated); entering NEB exits the native scan
  branch and never counts as a scan success;
* per-point optimization retries + endpoint preparations are counted into
  the todo-23 ``BudgetLedger`` (additive extension);
* the cost report separates **counted operations** from **measured time**:
  ``Σ constrained_optimizations`` is never actual core-hours
  (``equals_actual_core_hours=false``), and CPU/wall time stays
  ``n_cpu_hours=0.0`` + ``unknown`` until an actual receipt is recorded —
  never estimated silently;
* determinism: identical inputs produce byte-identical ledger/report JSON.

Every fixture is synthetic/in-memory; no test reads the real ``data/`` tree.
"""

from __future__ import annotations

from typing import Any

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.direction_assembly import (
    AnchorReason,
    SPLICE_POLICY,
)
from pes2ts_core.scan_strategy.direction_timing_cost import (
    CHANNEL_NEB,
    CHANNEL_SCAN,
    CODE_DIRECTION_MISMATCH,
    CODE_ENDPOINT_INVALID,
    CODE_METHOD_KIND_INVALID,
    CODE_MEASUREMENT_REQUIRES_RECEIPT,
    CODE_SPLICE_BLOCKED,
    COMPARISON_EARLY_VS_LATE,
    COMPARISON_FORWARD_VS_REVERSE,
    COMPARISON_KIND_ORDER,
    COMPARISON_PER_ASSEMBLY,
    COST_NOTE_COUNTED_VS_MEASURED,
    OBJECT_COST_REPORT,
    OBJECT_DIRECTION_TIMING_LEDGER,
    REASON_NO_REVERSE_PARTICIPANT,
    REASON_REVERSE_ASSEMBLY_UNVERIFIED,
    REASON_REVERSE_START_OPTIMIZATION_FAILED,
    REASON_REVERSE_VERIFICATION_REQUIRED,
    REASON_REVERSE_VERIFIED,
    REASON_TIMING_COMPARISON_NO_SPLICE,
    SCHEDULE_KIND_EARLY,
    SCHEDULE_KIND_LATE,
    SCHEMA_COST_REPORT,
    SCHEMA_DIRECTION_TIMING_LEDGER,
    STATUS_COMPLETE,
    STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION,
    SUM_IS_NOT_CORE_HOURS,
    ComparisonCandidate,
    CostAccounting,
    DirectionTimingCostError,
    ReverseStartVerification,
    assert_splice_permitted,
    build_comparison_ledger,
    cost_accounting,
    enter_neb_snapshot,
    zeroed_channels,
)
from pes2ts_core.scan_strategy.path_request import (
    BRANCH_NATIVE_SCAN_EXITED,
    METHOD_CHANNELS,
    enter_neb_channels,
)
from pes2ts_core.scan_strategy.plan_freeze import BudgetLedger
from pes2ts_core.utils.hashing import stable_json_dumps

FORBIDDEN: frozenset[str] = frozenset(
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
)

ANCHOR = (
    AnchorReason(
        code="LAYER2_DRIVER_BONDED_AT_START",
        layer="L2",
        endpoint="R",
        detail="chosen driver bonded at R start",
    ),
)


def _candidate(
    candidate_id: str,
    *,
    start_endpoint: str = "R",
    assembly_id: str | None = "asm-1",
    schedule_kind: str | None = None,
    schedule_id: str | None = None,
    anchor_reason: tuple[AnchorReason, ...] = ANCHOR,
    method_kind: str = CHANNEL_SCAN,
) -> ComparisonCandidate:
    direction = "R_to_P" if start_endpoint == "R" else "P_to_R"
    return ComparisonCandidate(
        candidate_id=candidate_id,
        start_endpoint=start_endpoint,
        direction=direction,
        assembly_id=assembly_id,
        anchor_reason=anchor_reason,
        schedule_id=schedule_id,
        schedule_kind=schedule_kind,
        method_kind=method_kind,
    )


def _verified(candidate_id: str) -> ReverseStartVerification:
    return ReverseStartVerification(
        candidate_id=candidate_id,
        verification_id=f"rv-{candidate_id}",
        start_optimization_ok=True,
        assembly_verified=True,
    )


def _unverified_start(candidate_id: str) -> ReverseStartVerification:
    return ReverseStartVerification(
        candidate_id=candidate_id,
        verification_id=f"rv-{candidate_id}",
        start_optimization_ok=False,
        assembly_verified=True,
    )


def _unverified_assembly(candidate_id: str) -> ReverseStartVerification:
    return ReverseStartVerification(
        candidate_id=candidate_id,
        verification_id=f"rv-{candidate_id}",
        start_optimization_ok=True,
        assembly_verified=False,
    )


# ---------------------------------------------------------------------------
# Ledger rows carry the four todo-15 fields per candidate.
# ---------------------------------------------------------------------------
def test_ledger_records_four_fields_per_candidate() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reaction_id="RXN_0000000001",
        reverse_verifications=[_verified("cand-rev")],
    )
    assert ledger.reaction_id == "RXN_0000000001"
    assert len(ledger.candidates) == 2
    for candidate, doc in zip(ledger.candidates, ledger.to_doc()["candidates"], strict=True):
        # The four fields exist both on the dataclass and in the projection.
        assert candidate.start_endpoint in {"R", "P"}
        assert candidate.direction in {"R_to_P", "P_to_R"}
        assert candidate.assembly_id == "asm-1"
        assert candidate.anchor_reason == ANCHOR
        for field in ("start_endpoint", "direction", "assembly_id", "anchor_reason"):
            assert field in doc
        assert doc["anchor_reason"][0]["code"] == "LAYER2_DRIVER_BONDED_AT_START"
    doc = ledger.to_doc()
    assert doc["schema_name"] == OBJECT_DIRECTION_TIMING_LEDGER
    assert doc["schema_version"] == SCHEMA_DIRECTION_TIMING_LEDGER
    assert doc["splice_policy"] == SPLICE_POLICY


def test_ledger_accepts_mapping_candidates_and_normalizes_anchor_reason() -> None:
    ledger = build_comparison_ledger(
        [
            {
                "candidate_id": "cand-fwd",
                "start_endpoint": "R",
                "direction": "R_to_P",
                "assembly_id": "asm-1",
                "anchor_reason": [{"code": "L3_CONFIGURATION_OK", "layer": "L3"}],
            }
        ]
    )
    candidate = ledger.candidates[0]
    assert candidate.anchor_reason[0].code == "L3_CONFIGURATION_OK"
    assert candidate.anchor_reason[0].layer == "L3"


def test_forward_only_ledger_emits_no_forward_vs_reverse_comparison() -> None:
    ledger = build_comparison_ledger(
        [_candidate("cand-fwd-1"), _candidate("cand-fwd-2")]
    )
    kinds = {record.comparison_kind for record in ledger.comparisons}
    assert COMPARISON_FORWARD_VS_REVERSE not in kinds
    # per-assembly still aggregates the forward group (no reverse → complete).
    per_assembly = [
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_PER_ASSEMBLY
    ]
    assert len(per_assembly) == 1
    assert per_assembly[0].status == STATUS_COMPLETE
    assert per_assembly[0].splice_permitted is False
    assert REASON_NO_REVERSE_PARTICIPANT in per_assembly[0].reasons


# ---------------------------------------------------------------------------
# Reverse without verification → incomplete comparison + no splice (typed).
# ---------------------------------------------------------------------------
def test_reverse_without_verification_incomplete_and_no_splice() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reaction_id="RXN_0000000002",
        # NO reverse verification record — structural gate must fire.
    )
    forward = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_FORWARD_VS_REVERSE
    )
    assert forward.status == STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION
    assert forward.splice_permitted is False
    assert REASON_REVERSE_VERIFICATION_REQUIRED in forward.reasons
    # Curves are NOT spliced: typed refusal, not a silent skip.
    with pytest.raises(DirectionTimingCostError) as excinfo:
        assert_splice_permitted(forward)
    assert excinfo.value.code == CODE_SPLICE_BLOCKED
    # per-assembly aggregate carries the same gate.
    per_assembly = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_PER_ASSEMBLY
    )
    assert per_assembly.status == STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION
    assert per_assembly.splice_permitted is False


def test_reverse_with_failed_start_optimization_typed_reason() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reverse_verifications=[_unverified_start("cand-rev")],
    )
    forward = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_FORWARD_VS_REVERSE
    )
    assert forward.status == STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION
    assert REASON_REVERSE_START_OPTIMIZATION_FAILED in forward.reasons
    assert forward.splice_permitted is False


def test_reverse_with_unverified_assembly_typed_reason() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reverse_verifications=[_unverified_assembly("cand-rev")],
    )
    forward = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_FORWARD_VS_REVERSE
    )
    assert REASON_REVERSE_ASSEMBLY_UNVERIFIED in forward.reasons
    assert forward.splice_permitted is False


def test_reverse_with_verified_record_completes_and_permits_splice() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reverse_verifications=[_verified("cand-rev")],
    )
    forward = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_FORWARD_VS_REVERSE
    )
    assert forward.status == STATUS_COMPLETE
    assert forward.splice_permitted is True
    assert REASON_REVERSE_VERIFIED in forward.reasons
    assert forward.reverse_verification_ids == ("rv-cand-rev",)
    assert_splice_permitted(forward)  # must not raise


def test_verification_for_unknown_candidate_rejected() -> None:
    with pytest.raises(DirectionTimingCostError):
        build_comparison_ledger(
            [_candidate("cand-fwd")],
            reverse_verifications=[_verified("ghost")],
        )


def test_direction_mismatch_and_bad_method_kind_rejected() -> None:
    with pytest.raises(DirectionTimingCostError) as excinfo:
        ComparisonCandidate(
            candidate_id="bad",
            start_endpoint="P",
            direction="R_to_P",  # must be P_to_R for a P start
            assembly_id=None,
        )
    assert excinfo.value.code == CODE_DIRECTION_MISMATCH
    with pytest.raises(DirectionTimingCostError) as excinfo:
        ComparisonCandidate(
            candidate_id="bad",
            start_endpoint="R",
            direction="R_to_P",
            assembly_id=None,
            method_kind="grid",  # not a METHOD_CHANNELS member
        )
    assert excinfo.value.code == CODE_METHOD_KIND_INVALID


def test_endpoint_and_direction_vocabulary_from_contracts() -> None:
    # The ledger vocabulary stays aligned with contracts_v2 (single source).
    from pes2ts_core.scan_strategy.contracts_v2 import DIRECTIONS, ENDPOINTS

    assert set(METHOD_CHANNELS) == {CHANNEL_NEB, CHANNEL_SCAN}
    assert DIRECTIONS == ("R_to_P", "P_to_R")
    assert ENDPOINTS == ("R", "P")
    assert COMPARISON_KIND_ORDER == (
        COMPARISON_FORWARD_VS_REVERSE,
        COMPARISON_EARLY_VS_LATE,
        COMPARISON_PER_ASSEMBLY,
    )


# ---------------------------------------------------------------------------
# Early vs late timing comparison + per-assembly grouping.
# ---------------------------------------------------------------------------
def test_early_vs_late_comparison_emitted_and_never_splices() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate(
                "cand-early",
                schedule_kind=SCHEDULE_KIND_EARLY,
                schedule_id="sched-1",
            ),
            _candidate(
                "cand-late",
                schedule_kind=SCHEDULE_KIND_LATE,
                schedule_id="sched-1",
            ),
        ]
    )
    timing = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_EARLY_VS_LATE
    )
    assert timing.status == STATUS_COMPLETE
    assert timing.splice_permitted is False
    assert REASON_TIMING_COMPARISON_NO_SPLICE in timing.reasons
    assert timing.candidate_ids == ("cand-early", "cand-late")


def test_early_vs_late_with_unverified_reverse_candidate_incomplete() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate(
                "cand-early",
                start_endpoint="P",
                schedule_kind=SCHEDULE_KIND_EARLY,
                schedule_id="sched-1",
            ),
            _candidate(
                "cand-late",
                start_endpoint="P",
                schedule_kind=SCHEDULE_KIND_LATE,
                schedule_id="sched-1",
            ),
        ]
        # no verification → incomplete even for the timing comparison
    )
    timing = next(
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_EARLY_VS_LATE
    )
    assert timing.status == STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION
    assert timing.splice_permitted is False


def test_per_assembly_comparison_groups_candidates_of_one_assembly() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("a-fwd", assembly_id="asm-a"),
            _candidate("a-rev", start_endpoint="P", assembly_id="asm-a"),
            _candidate("b-fwd", assembly_id="asm-b"),
        ],
        reverse_verifications=[_verified("a-rev")],
    )
    per_assembly = [
        record
        for record in ledger.comparisons
        if record.comparison_kind == COMPARISON_PER_ASSEMBLY
    ]
    assert len(per_assembly) == 2  # asm-a and asm-b
    asm_a = next(
        record for record in per_assembly if "a-fwd" in record.candidate_ids
    )
    assert asm_a.candidate_ids == ("a-fwd", "a-rev")
    asm_b = next(
        record for record in per_assembly if "b-fwd" in record.candidate_ids
    )
    assert asm_b.candidate_ids == ("b-fwd",)
    assert asm_b.status == STATUS_COMPLETE


def test_forward_vs_reverse_requires_shared_assembly_and_schedule_key() -> None:
    # Different assembly ids → no forward_vs_reverse pairing.
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd", assembly_id="asm-1"),
            _candidate("cand-rev", start_endpoint="P", assembly_id="asm-2"),
        ],
        reverse_verifications=[_verified("cand-rev")],
    )
    kinds = {record.comparison_kind for record in ledger.comparisons}
    assert COMPARISON_FORWARD_VS_REVERSE not in kinds


# ---------------------------------------------------------------------------
# NEB / Scan channels stay separate (todo-26 vocabulary, imported).
# ---------------------------------------------------------------------------
def test_zeroed_channels_align_with_todo26_enter_neb_snapshot() -> None:
    accounting = cost_accounting("RXN_0000000003")
    entered = accounting.enter_method(CHANNEL_NEB)
    assert entered.channels == enter_neb_channels()
    assert enter_neb_snapshot() == enter_neb_channels()


def test_neb_outcome_never_increments_scan_channel() -> None:
    accounting = cost_accounting("RXN_0000000004").enter_method(CHANNEL_NEB)
    neb_before = accounting.channel_of(CHANNEL_NEB)
    scan_before = accounting.channel_of(CHANNEL_SCAN)
    assert neb_before.n_entered == 1
    assert neb_before.branch == BRANCH_NATIVE_SCAN_EXITED
    assert scan_before.n_entered == 0
    succeeded = accounting.record_outcome(CHANNEL_NEB, succeeded=True)
    assert succeeded.channel_of(CHANNEL_NEB).n_succeeded == 1
    # Test lock: a NEB success never increments scan.
    assert succeeded.channel_of(CHANNEL_SCAN).n_succeeded == 0
    assert succeeded.channel_of(CHANNEL_SCAN).n_entered == 0
    failed = succeeded.record_outcome(CHANNEL_NEB, succeeded=False)
    assert failed.channel_of(CHANNEL_NEB).n_failed == 1
    assert failed.channel_of(CHANNEL_SCAN).n_failed == 0


def test_scan_entry_does_not_touch_neb_channel() -> None:
    accounting = cost_accounting("RXN_0000000005").enter_method(CHANNEL_SCAN)
    neb = accounting.channel_of(CHANNEL_NEB)
    scan = accounting.channel_of(CHANNEL_SCAN)
    assert scan.n_entered == 1
    assert neb.n_entered == 0
    ok = accounting.record_outcome(CHANNEL_SCAN, succeeded=True)
    assert ok.channel_of(CHANNEL_SCAN).n_succeeded == 1
    assert ok.channel_of(CHANNEL_NEB).n_succeeded == 0


def test_unknown_method_kind_rejected() -> None:
    accounting = cost_accounting("RXN_0000000006")
    with pytest.raises(DirectionTimingCostError) as excinfo:
        accounting.enter_method("grid")
    assert excinfo.value.code == CODE_METHOD_KIND_INVALID
    with pytest.raises(DirectionTimingCostError):
        accounting.record_outcome("grid", succeeded=True)


# ---------------------------------------------------------------------------
# Budget extension: retries + endpoint preparations counted (todo 23).
# ---------------------------------------------------------------------------
def test_point_retries_and_endpoint_preparations_counted_into_budget() -> None:
    accounting = cost_accounting("RXN_0000000007", max_attempts=10)
    extended = accounting.record_constrained_optimizations(
        n=9, point_retries=2, endpoint_preparations=1
    )
    assert extended.budget.spent_retries == 2
    assert extended.budget.spent_endpoint_preparations == 1
    assert extended.budget.spent_attempts == 0
    assert extended.counted.n_constrained_optimizations == 9
    assert extended.counted.n_point_retries == 2
    assert extended.counted.n_endpoint_preparations == 1
    # Additive: original ledger untouched.
    assert accounting.budget.spent_retries == 0
    assert accounting.counted.n_constrained_optimizations == 0


def test_directions_assemblies_schedules_counted_into_budget() -> None:
    accounting = cost_accounting("RXN_0000000008", max_attempts=10)
    extended = accounting.record_directions(
        n_directions=2, n_assemblies=2, n_schedules=3
    )
    assert extended.budget.spent_directions == 2
    assert extended.budget.spent_assemblies == 2
    assert extended.budget.spent_schedules == 3
    assert extended.counted.n_directions == 2
    assert extended.counted.n_assemblies == 2
    assert extended.counted.n_schedules == 3


def test_budget_ledger_is_plan_freeze_budget() -> None:
    accounting = cost_accounting("RXN_0000000009")
    assert isinstance(accounting.budget, BudgetLedger)
    frozen = BudgetLedger(max_attempts=3)
    with_frozen = cost_accounting("RXN_0000000009", budget=frozen)
    assert with_frozen.budget is frozen
    extended = with_frozen.record_constrained_optimizations(n=1, point_retries=1)
    assert extended.budget.max_attempts == 3
    assert extended.budget.exhausted() is False


def test_negative_counts_rejected() -> None:
    accounting = cost_accounting("RXN_0000000010")
    with pytest.raises(DirectionTimingCostError):
        accounting.record_constrained_optimizations(n=-1)
    with pytest.raises(DirectionTimingCostError):
        accounting.record_constrained_optimizations(n=1, point_retries=-2)
    with pytest.raises(DirectionTimingCostError):
        accounting.record_directions(n_assemblies=-1)


# ---------------------------------------------------------------------------
# Cost report: counted vs measured (design §13.4; never estimate CPU).
# ---------------------------------------------------------------------------
def test_cost_report_distinguishes_counted_from_measured() -> None:
    accounting = cost_accounting("RXN_0000000011", max_attempts=10)
    report = accounting.record_constrained_optimizations(
        n=45, point_retries=3, endpoint_preparations=2
    ).cost_report()
    assert report["schema_name"] == OBJECT_COST_REPORT
    assert report["schema_version"] == SCHEMA_COST_REPORT
    counted = report["counted_operations"]
    assert counted["n_constrained_optimizations"] == 45
    assert counted["sum_constrained_optimizations"] == 45
    assert counted["equals_actual_core_hours"] is SUM_IS_NOT_CORE_HOURS
    assert counted["equals_actual_core_hours"] is False
    assert counted["note"] == COST_NOTE_COUNTED_VS_MEASURED
    measured = report["measured_time"]
    assert measured["n_cpu_hours"] == 0.0
    assert measured["cpu_time_status"] == "unknown"
    assert measured["receipt_ids"] == []
    # Budget side shows the counted retries/prep landed in the ledger.
    assert report["budget"]["spent"]["retries"] == 3
    assert report["budget"]["spent"]["endpoint_preparations"] == 2


def test_cpu_time_stays_unknown_until_receipt_never_estimated() -> None:
    accounting = cost_accounting("RXN_0000000012")
    # Counting 1000 operations must NOT invent CPU hours.
    heavy = accounting.record_constrained_optimizations(n=1000, point_retries=500)
    report = heavy.cost_report()
    assert report["measured_time"]["n_cpu_hours"] == 0.0
    assert report["measured_time"]["cpu_time_status"] == "unknown"
    assert (
        report["counted_operations"]["n_constrained_optimizations"] == 1000
    )
    # Measured time requires an explicit receipt — silent estimation refused.
    with pytest.raises(DirectionTimingCostError) as excinfo:
        heavy.record_measured_time(receipt_id="   ", n_cpu_hours=12.5)
    assert excinfo.value.code == CODE_MEASUREMENT_REQUIRES_RECEIPT
    still_unknown = heavy.cost_report()["measured_time"]
    assert still_unknown["n_cpu_hours"] == 0.0
    assert still_unknown["cpu_time_status"] == "unknown"


def test_measured_time_from_receipt_updates_status() -> None:
    accounting = cost_accounting("RXN_0000000013")
    measured = accounting.record_measured_time(
        receipt_id="receipt-acp-001", n_cpu_hours=12.5, n_wall_seconds=3600
    )
    doc = measured.cost_report()["measured_time"]
    assert doc["n_cpu_hours"] == 12.5
    assert doc["cpu_time_status"] == "measured"
    assert doc["n_wall_seconds"] == 3600.0
    assert doc["wall_time_status"] == "measured"
    assert doc["receipt_ids"] == ["receipt-acp-001"]
    # Counted operations remain a pure count — measured time never folds in.
    assert (
        measured.cost_report()["counted_operations"]["n_constrained_optimizations"]
        == 0
    )


def test_cost_report_includes_separate_method_channels() -> None:
    accounting = cost_accounting("RXN_0000000014").enter_method(CHANNEL_NEB)
    report = accounting.cost_report()
    channels = report["method_channels"]
    assert set(channels) == set(METHOD_CHANNELS)
    assert channels[CHANNEL_NEB]["n_entered"] == 1
    assert channels[CHANNEL_SCAN]["n_entered"] == 0
    assert channels[CHANNEL_SCAN]["branch"] == BRANCH_NATIVE_SCAN_EXITED


# ---------------------------------------------------------------------------
# Determinism + purity.
# ---------------------------------------------------------------------------
def test_ledger_and_cost_report_are_deterministic() -> None:
    candidates = [
        _candidate("cand-fwd"),
        _candidate("cand-rev", start_endpoint="P"),
        _candidate(
            "cand-early",
            schedule_kind=SCHEDULE_KIND_EARLY,
            schedule_id="sched-1",
        ),
        _candidate(
            "cand-late",
            schedule_kind=SCHEDULE_KIND_LATE,
            schedule_id="sched-1",
        ),
    ]
    verifications = [_verified("cand-rev")]
    first = build_comparison_ledger(
        candidates, reaction_id="RXN_0000000015", reverse_verifications=verifications
    )
    second = build_comparison_ledger(
        candidates, reaction_id="RXN_0000000015", reverse_verifications=verifications
    )
    assert first.to_json() == second.to_json()
    assert stable_json_dumps(first.to_doc()) == stable_json_dumps(second.to_doc())
    accounting_a = cost_accounting("RXN_0000000015", max_attempts=5).record_constrained_optimizations(
        n=9, point_retries=1
    )
    accounting_b = cost_accounting("RXN_0000000015", max_attempts=5).record_constrained_optimizations(
        n=9, point_retries=1
    )
    assert accounting_a.cost_report_json() == accounting_b.cost_report_json()


def test_ledger_and_cost_report_carry_no_forbidden_truth_keys() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reverse_verifications=[_verified("cand-rev")],
    )
    report = (
        cost_accounting("RXN_0000000016")
        .enter_method(CHANNEL_NEB)
        .record_outcome(CHANNEL_NEB, succeeded=True)
        .record_constrained_optimizations(n=3)
        .cost_report()
    )
    for doc in (ledger.to_doc(), report):
        stack: list[Any] = [doc]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    assert str(key).lower() not in FORBIDDEN
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)


def test_comparison_lookup_and_missing_id_typed() -> None:
    ledger = build_comparison_ledger(
        [
            _candidate("cand-fwd"),
            _candidate("cand-rev", start_endpoint="P"),
        ],
        reverse_verifications=[_verified("cand-rev")],
    )
    record = ledger.comparison("cmp-0001")
    assert record.comparison_kind == COMPARISON_FORWARD_VS_REVERSE
    with pytest.raises(DirectionTimingCostError):
        ledger.comparison("cmp-9999")


def test_candidate_to_doc_always_carries_four_fields_even_when_none() -> None:
    candidate = _candidate("cand-bare", assembly_id=None, anchor_reason=())
    doc = candidate.to_doc()
    assert doc["candidate_id"] == "cand-bare"
    assert doc["start_endpoint"] == "R"
    assert doc["direction"] == "R_to_P"
    assert doc["assembly_id"] is None
    assert doc["anchor_reason"] == []
