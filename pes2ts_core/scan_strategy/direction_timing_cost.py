"""Direction/timing comparison ledger and cost accounting (todo 28, design §7/§8.3/§13.4).

Implements the comparison-ledger and cost-report surface of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``:

- **Comparison ledger** (§7): every direction/timing comparison candidate of
  one reaction records the four todo-15 fields
  ``start_endpoint / direction / assembly_id / anchor_reason``.  Comparisons
  are emitted across candidates of one reaction — forward vs reverse, early
  vs late, per assembly.
- **Reverse-verification gate** (§7): a reverse candidate (``P_to_R``)
  requires an independent start-optimization + assembly verification record
  (:class:`ReverseStartVerification`).  Structurally, the ledger marks any
  comparison that involves an unverified reverse candidate
  ``incomplete_missing_reverse_verification`` with ``splice_permitted=False``
  — reverse high points are **never** spliced onto forward curves without
  verification (``SPLICE_POLICY`` from todo 15 is re-exported, not restated).
- **Separate NEB / Scan channels** (§8.3/§13.4, todo 26): cost and success
  are accounted per ``method_kind`` channel by **importing** the todo-26
  vocabulary (``MethodChannelAccounting``, ``enter_neb_channels``,
  ``record_channel_outcome``, ``CHANNEL_SCAN``/``CHANNEL_NEB``); this module
  never duplicates the channel accounting logic.
- **Budget extension** (§8.3, todo 23): per-point optimization retries and
  endpoint preparations are counted into a todo-23 ``BudgetLedger`` —
  additive usage only (``BudgetLedger.record``), no fork of the ledger.
- **Counted vs measured cost** (§13.4): ``Σ constrained_optimizations`` is a
  count of operations, **not** actual core-hours.  :meth:`CostAccounting.
  cost_report` separates ``counted_operations`` from ``measured_time``;
  measured CPU/wall time is recorded only from actual receipts
  (``n_cpu_hours=0.0`` + ``cpu_time_status="unknown"`` until a receipt is
  supplied — never estimated silently).
"""

# allow: SIZE_OK — plan-named todo-28 single module (comparison ledger +
# reverse-verification splice gate + channel/budget cost accounting +
# counted-vs-measured cost report); precedent path_request.py / plan_freeze.py.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from pes2ts_core.scan_strategy.contracts_v2 import DIRECTIONS, ENDPOINTS
from pes2ts_core.scan_strategy.direction_assembly import SPLICE_POLICY, AnchorReason
from pes2ts_core.scan_strategy.path_request import (
    BRANCH_NATIVE_SCAN,
    BRANCH_NATIVE_SCAN_EXITED,
    CHANNEL_NEB,
    CHANNEL_SCAN,
    METHOD_CHANNELS,
    NEB_ENTER_EXITS_SCAN_BRANCH,
    SCAN_CHANNEL_COST_NOTE,
    MethodChannelAccounting,
    enter_neb_channels,
    record_channel_outcome,
)
from pes2ts_core.scan_strategy.plan_freeze import BudgetLedger
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps

# ---------------------------------------------------------------------------
# Schema vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_DIRECTION_TIMING_LEDGER: Final[str] = "g1_direction_timing_ledger_v1"
SCHEMA_COST_REPORT: Final[str] = "g1_direction_timing_cost_v1"
OBJECT_DIRECTION_TIMING_LEDGER: Final[str] = "DirectionTimingLedger"
OBJECT_COST_REPORT: Final[str] = "CostReport"

#: Typed refusal codes (stable machine tokens; never free text).
CODE_CANDIDATE_INVALID: Final[str] = "CANDIDATE_INVALID"
CODE_ENDPOINT_INVALID: Final[str] = "ENDPOINT_INVALID"
CODE_DIRECTION_MISMATCH: Final[str] = "DIRECTION_MISMATCH"
CODE_METHOD_KIND_INVALID: Final[str] = "METHOD_KIND_INVALID"
CODE_REVERSE_VERIFICATION_INVALID: Final[str] = "REVERSE_VERIFICATION_INVALID"
CODE_SPLICE_BLOCKED: Final[str] = "SPLICE_BLOCKED"
CODE_MEASUREMENT_REQUIRES_RECEIPT: Final[str] = "MEASUREMENT_REQUIRES_RECEIPT"
CODE_MEASUREMENT_INVALID: Final[str] = "MEASUREMENT_INVALID"
CODE_COUNT_INVALID: Final[str] = "COUNT_INVALID"

# Comparison vocabulary (design §7: forward vs reverse, early vs late, per
# assembly).
COMPARISON_FORWARD_VS_REVERSE: Final[str] = "forward_vs_reverse"
COMPARISON_EARLY_VS_LATE: Final[str] = "early_vs_late"
COMPARISON_PER_ASSEMBLY: Final[str] = "per_assembly"
COMPARISON_KIND_ORDER: Final[tuple[str, ...]] = (
    COMPARISON_FORWARD_VS_REVERSE,
    COMPARISON_EARLY_VS_LATE,
    COMPARISON_PER_ASSEMBLY,
)

STATUS_COMPLETE: Final[str] = "complete"
STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION: Final[str] = (
    "incomplete_missing_reverse_verification"
)

# Typed comparison reasons.
REASON_REVERSE_VERIFICATION_REQUIRED: Final[str] = "REVERSE_START_VERIFICATION_REQUIRED"
REASON_REVERSE_START_OPTIMIZATION_FAILED: Final[str] = (
    "REVERSE_START_OPTIMIZATION_FAILED"
)
REASON_REVERSE_ASSEMBLY_UNVERIFIED: Final[str] = "REVERSE_ASSEMBLY_UNVERIFIED"
REASON_NO_REVERSE_PARTICIPANT: Final[str] = "NO_REVERSE_PARTICIPANT"
REASON_TIMING_COMPARISON_NO_SPLICE: Final[str] = "TIMING_COMPARISON_NO_CURVE_SPLICE"
REASON_REVERSE_VERIFIED: Final[str] = "REVERSE_START_VERIFIED"

#: Timing schedule kinds participating in early-vs-late comparisons
#: (contracts_v2.SCHEDULE_KINDS names; todo 14 owns generation).
SCHEDULE_KIND_EARLY: Final[str] = "event_A_early"
SCHEDULE_KIND_LATE: Final[str] = "event_A_late"
TIMING_SCHEDULE_KINDS: Final[frozenset[str]] = frozenset(
    {SCHEDULE_KIND_EARLY, SCHEDULE_KIND_LATE}
)

#: Machine-checkable cost-report invariants (design §8.3/§13.4).
COST_NOTE_COUNTED_VS_MEASURED: Final[str] = (
    "Sigma constrained_optimizations counts operations, not actual "
    "core-hours; SCF/iteration cost must be measured from backend receipts"
)
MEASURED_ONLY_FROM_RECEIPTS_NOTE: Final[str] = (
    "measured CPU/wall time recorded only from actual receipts; "
    "never estimated silently"
)
#: Lock: counted operations NEVER equal actual core-hours.
SUM_IS_NOT_CORE_HOURS: Final[bool] = False

_MEASURED_STATUS_UNKNOWN: Final[str] = "unknown"
_MEASURED_STATUS_MEASURED: Final[str] = "measured"


class DirectionTimingCostError(ValueError):
    """Typed ledger/cost refusal; ``code`` is a stable machine token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Candidate ledger rows (four todo-15 fields + timing/method dimensions).
# ---------------------------------------------------------------------------
def _normalize_anchor_reason(raw: Any) -> AnchorReason:
    """Parse one anchor-reason entry at the single trust boundary."""
    if isinstance(raw, AnchorReason):
        return raw
    if isinstance(raw, str):
        return AnchorReason(code=raw, layer="ledger", endpoint=None, detail="")
    if isinstance(raw, Mapping):
        code = raw.get("code")
        if not isinstance(code, str) or not code:
            raise DirectionTimingCostError(
                CODE_CANDIDATE_INVALID, "anchor_reason entry needs a non-empty code"
            )
        layer = raw.get("layer")
        endpoint = raw.get("endpoint")
        detail = raw.get("detail")
        return AnchorReason(
            code=code,
            layer=str(layer) if layer is not None else "ledger",
            endpoint=None if endpoint is None else str(endpoint),
            detail=str(detail) if detail is not None else "",
        )
    raise DirectionTimingCostError(
        CODE_CANDIDATE_INVALID, f"anchor_reason entry must be str/mapping/AnchorReason, got {type(raw).__name__}"
    )


@dataclass(frozen=True, slots=True)
class ComparisonCandidate:
    """One direction/timing comparison candidate row (design §7 fields)."""

    candidate_id: str
    start_endpoint: str
    direction: str
    assembly_id: str | None
    anchor_reason: tuple[AnchorReason, ...] = ()
    schedule_id: str | None = None
    schedule_kind: str | None = None
    method_kind: str = CHANNEL_SCAN

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise DirectionTimingCostError(
                CODE_CANDIDATE_INVALID, "candidate_id must be a non-empty string"
            )
        if self.start_endpoint not in ENDPOINTS:
            raise DirectionTimingCostError(
                CODE_ENDPOINT_INVALID,
                f"start_endpoint={self.start_endpoint!r}; expected one of {list(ENDPOINTS)}",
            )
        if self.direction not in DIRECTIONS:
            raise DirectionTimingCostError(
                CODE_DIRECTION_MISMATCH,
                f"direction={self.direction!r}; expected one of {list(DIRECTIONS)}",
            )
        expected = "R_to_P" if self.start_endpoint == "R" else "P_to_R"
        if self.direction != expected:
            raise DirectionTimingCostError(
                CODE_DIRECTION_MISMATCH,
                f"direction={self.direction!r} must start from start_endpoint="
                f"{self.start_endpoint!r} (expected {expected!r})",
            )
        if self.method_kind not in METHOD_CHANNELS:
            raise DirectionTimingCostError(
                CODE_METHOD_KIND_INVALID,
                f"method_kind={self.method_kind!r}; expected one of {list(METHOD_CHANNELS)}",
            )

    @property
    def is_reverse(self) -> bool:
        """True for a P-start (reverse) candidate — verification-gated."""
        return self.direction == "P_to_R"

    @property
    def is_timing_pair(self) -> bool:
        return self.schedule_kind in TIMING_SCHEDULE_KINDS

    def to_doc(self) -> dict[str, JSONValue]:
        """JSON-safe row; always carries the four todo-15 comparison fields."""
        return {
            "candidate_id": self.candidate_id,
            "start_endpoint": self.start_endpoint,
            "direction": self.direction,
            "assembly_id": self.assembly_id,
            "anchor_reason": [reason.to_doc() for reason in self.anchor_reason],
            "schedule_id": self.schedule_id,
            "schedule_kind": self.schedule_kind,
            "method_kind": self.method_kind,
        }


def _parse_candidate(raw: ComparisonCandidate | Mapping[str, Any]) -> ComparisonCandidate:
    """Boundary parse for one candidate row (dataclass passthrough or mapping)."""
    if isinstance(raw, ComparisonCandidate):
        return raw
    if not isinstance(raw, Mapping):
        raise DirectionTimingCostError(
            CODE_CANDIDATE_INVALID, "candidate must be a ComparisonCandidate or mapping"
        )
    candidate_id = raw.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise DirectionTimingCostError(
            CODE_CANDIDATE_INVALID, "candidate.candidate_id must be a non-empty string"
        )
    assembly_raw = raw.get("assembly_id")
    assembly_id = None if assembly_raw is None else str(assembly_raw)
    schedule_id = raw.get("schedule_id")
    schedule_kind = raw.get("schedule_kind")
    anchor_raw = raw.get("anchor_reason") or ()
    if not isinstance(anchor_raw, (list, tuple)):
        raise DirectionTimingCostError(
            CODE_CANDIDATE_INVALID, "candidate.anchor_reason must be an array"
        )
    anchors = tuple(_normalize_anchor_reason(entry) for entry in anchor_raw)
    return ComparisonCandidate(
        candidate_id=candidate_id,
        start_endpoint=str(raw.get("start_endpoint") or ""),
        direction=str(raw.get("direction") or ""),
        assembly_id=assembly_id,
        anchor_reason=anchors,
        schedule_id=None if schedule_id is None else str(schedule_id),
        schedule_kind=None if schedule_kind is None else str(schedule_kind),
        method_kind=str(raw.get("method_kind") or CHANNEL_SCAN),
    )


# ---------------------------------------------------------------------------
# Reverse-start verification (design §7 structural gate).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReverseStartVerification:
    """Independent reverse-start optimization + assembly verification record.

    A reverse candidate needs BOTH a completed start optimization and an
    assembly verification before any forward-vs-reverse comparison may be
    marked complete or splice-permitted.  This module records the typed
    verdict; it never performs the optimization/assembly itself.
    """

    candidate_id: str
    verification_id: str
    start_optimization_ok: bool
    assembly_verified: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise DirectionTimingCostError(
                CODE_REVERSE_VERIFICATION_INVALID,
                "candidate_id must be a non-empty string",
            )
        if not isinstance(self.verification_id, str) or not self.verification_id:
            raise DirectionTimingCostError(
                CODE_REVERSE_VERIFICATION_INVALID,
                "verification_id must be a non-empty string",
            )

    @property
    def verified(self) -> bool:
        return self.start_optimization_ok and self.assembly_verified

    @property
    def failure_reason(self) -> str | None:
        """Typed reason code when unverified; ``None`` when verified."""
        if self.verified:
            return None
        if not self.start_optimization_ok:
            return REASON_REVERSE_START_OPTIMIZATION_FAILED
        return REASON_REVERSE_ASSEMBLY_UNVERIFIED

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "verification_id": self.verification_id,
            "start_optimization_ok": self.start_optimization_ok,
            "assembly_verified": self.assembly_verified,
            "verified": self.verified,
            "detail": self.detail,
        }


def _parse_reverse_verification(
    raw: ReverseStartVerification | Mapping[str, Any],
) -> ReverseStartVerification:
    if isinstance(raw, ReverseStartVerification):
        return raw
    if not isinstance(raw, Mapping):
        raise DirectionTimingCostError(
            CODE_REVERSE_VERIFICATION_INVALID,
            "reverse verification must be a ReverseStartVerification or mapping",
        )
    candidate_id = raw.get("candidate_id")
    verification_id = raw.get("verification_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise DirectionTimingCostError(
            CODE_REVERSE_VERIFICATION_INVALID,
            "reverse verification.candidate_id must be a non-empty string",
        )
    if not isinstance(verification_id, str) or not verification_id:
        raise DirectionTimingCostError(
            CODE_REVERSE_VERIFICATION_INVALID,
            "reverse verification.verification_id must be a non-empty string",
        )
    start_ok = raw.get("start_optimization_ok")
    assembly_ok = raw.get("assembly_verified")
    if not isinstance(start_ok, bool) or not isinstance(assembly_ok, bool):
        raise DirectionTimingCostError(
            CODE_REVERSE_VERIFICATION_INVALID,
            "reverse verification flags must be booleans",
        )
    detail = raw.get("detail")
    return ReverseStartVerification(
        candidate_id=candidate_id,
        verification_id=verification_id,
        start_optimization_ok=start_ok,
        assembly_verified=assembly_ok,
        detail="" if detail is None else str(detail),
    )


# ---------------------------------------------------------------------------
# Comparison records + the per-reaction ledger.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """One emitted comparison with its typed splice gate."""

    comparison_id: str
    comparison_kind: str
    candidate_ids: tuple[str, ...]
    status: str
    splice_permitted: bool
    reasons: tuple[str, ...]
    reverse_verification_ids: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.status == STATUS_COMPLETE

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "comparison_id": self.comparison_id,
            "comparison_kind": self.comparison_kind,
            "candidate_ids": list(self.candidate_ids),
            "status": self.status,
            "splice_permitted": self.splice_permitted,
            "reasons": list(self.reasons),
            "reverse_verification_ids": list(self.reverse_verification_ids),
        }


@dataclass(frozen=True, slots=True)
class DirectionTimingLedger:
    """Per-reaction direction/timing comparison ledger (design §7)."""

    reaction_id: str | None
    candidates: tuple[ComparisonCandidate, ...]
    comparisons: tuple[ComparisonRecord, ...]
    reverse_verifications: tuple[ReverseStartVerification, ...]

    def comparison(self, comparison_id: str) -> ComparisonRecord:
        """Look up one comparison; typed refusal when absent."""
        for record in self.comparisons:
            if record.comparison_id == comparison_id:
                return record
        raise DirectionTimingCostError(
            CODE_CANDIDATE_INVALID, f"comparison_id={comparison_id!r} not in ledger"
        )

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "schema_name": OBJECT_DIRECTION_TIMING_LEDGER,
            "schema_version": SCHEMA_DIRECTION_TIMING_LEDGER,
            "reaction_id": self.reaction_id,
            "splice_policy": SPLICE_POLICY,
            "candidates": [candidate.to_doc() for candidate in self.candidates],
            "comparisons": [record.to_doc() for record in self.comparisons],
            "reverse_verifications": [
                record.to_doc() for record in self.reverse_verifications
            ],
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


def _reverse_gate(
    group: Sequence[ComparisonCandidate],
    verification_by_id: Mapping[str, ReverseStartVerification],
) -> tuple[str, bool, tuple[str, ...], tuple[str, ...]]:
    """Apply the structural reverse-verification gate to one candidate group.

    Returns ``(status, splice_permitted, reasons, verification_ids)``.
    No reverse participant → complete (nothing to verify), splice not
    applicable.  Any reverse participant without a verified record →
    ``incomplete_missing_reverse_verification`` + ``splice_permitted=False``.
    """
    reverses = sorted(
        (c for c in group if c.is_reverse), key=lambda c: c.candidate_id
    )
    if not reverses:
        return (
            STATUS_COMPLETE,
            False,
            (REASON_NO_REVERSE_PARTICIPANT,),
            (),
        )
    reasons: list[str] = []
    verification_ids: list[str] = []
    missing = False
    for candidate in reverses:
        verification = verification_by_id.get(candidate.candidate_id)
        if verification is None or not verification.verified:
            missing = True
            if verification is None:
                reasons.append(REASON_REVERSE_VERIFICATION_REQUIRED)
            else:
                reasons.append(verification.failure_reason or REASON_REVERSE_VERIFICATION_REQUIRED)
        else:
            verification_ids.append(verification.verification_id)
            reasons.append(REASON_REVERSE_VERIFIED)
    if missing:
        return (
            STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION,
            False,
            tuple(reasons),
            tuple(verification_ids),
        )
    return STATUS_COMPLETE, True, tuple(reasons), tuple(verification_ids)


def _pair_key(candidate: ComparisonCandidate) -> tuple[str, str]:
    """Forward/reverse pairing key: same assembly + same schedule identity."""
    return (
        candidate.assembly_id or "",
        candidate.schedule_id or candidate.schedule_kind or "",
    )


def _emit_comparisons(
    candidates: Sequence[ComparisonCandidate],
    verification_by_id: Mapping[str, ReverseStartVerification],
) -> tuple[ComparisonRecord, ...]:
    """Deterministically emit forward-vs-reverse, early-vs-late, per-assembly."""
    ordered = sorted(candidates, key=lambda c: c.candidate_id)
    drafts: list[tuple[int, tuple[str, ...], ComparisonRecord]] = []

    # --- forward vs reverse (per assembly+schedule pairing key) -----------
    groups: dict[tuple[str, str], list[ComparisonCandidate]] = {}
    for candidate in ordered:
        groups.setdefault(_pair_key(candidate), []).append(candidate)
    for key in sorted(groups):
        group = groups[key]
        forwards = [c for c in group if not c.is_reverse]
        reverses = [c for c in group if c.is_reverse]
        if not forwards or not reverses:
            continue
        status, splice, reasons, v_ids = _reverse_gate(group, verification_by_id)
        ids = tuple(sorted(c.candidate_id for c in group))
        drafts.append(
            (
                COMPARISON_KIND_ORDER.index(COMPARISON_FORWARD_VS_REVERSE),
                ids,
                ComparisonRecord(
                    comparison_id="",  # assigned after deterministic sort
                    comparison_kind=COMPARISON_FORWARD_VS_REVERSE,
                    candidate_ids=ids,
                    status=status,
                    splice_permitted=splice,
                    reasons=reasons,
                    reverse_verification_ids=v_ids,
                ),
            )
        )

    # --- early vs late (same direction + assembly; timing kinds) ----------
    timing_groups: dict[tuple[str, str, str], list[ComparisonCandidate]] = {}
    for candidate in ordered:
        if not candidate.is_timing_pair:
            continue
        timing_groups.setdefault(
            (candidate.direction, candidate.assembly_id or "", candidate.schedule_id or ""),
            [],
        ).append(candidate)
    for key in sorted(timing_groups):
        group = timing_groups[key]
        kinds = {c.schedule_kind for c in group}
        if not (TIMING_SCHEDULE_KINDS <= kinds):
            continue
        status, splice, reasons, v_ids = _reverse_gate(group, verification_by_id)
        reasons = tuple(dict.fromkeys((*reasons, REASON_TIMING_COMPARISON_NO_SPLICE)))
        ids = tuple(sorted(c.candidate_id for c in group))
        drafts.append(
            (
                COMPARISON_KIND_ORDER.index(COMPARISON_EARLY_VS_LATE),
                ids,
                ComparisonRecord(
                    comparison_id="",
                    comparison_kind=COMPARISON_EARLY_VS_LATE,
                    candidate_ids=ids,
                    status=status,
                    splice_permitted=False,  # timing comparison never splices curves
                    reasons=reasons,
                    reverse_verification_ids=v_ids,
                ),
            )
        )

    # --- per assembly (aggregate gate over every candidate of an assembly) -
    assembly_groups: dict[str, list[ComparisonCandidate]] = {}
    for candidate in ordered:
        if candidate.assembly_id is None:
            continue
        assembly_groups.setdefault(candidate.assembly_id, []).append(candidate)
    for assembly_id in sorted(assembly_groups):
        group = assembly_groups[assembly_id]
        status, splice, reasons, v_ids = _reverse_gate(group, verification_by_id)
        ids = tuple(sorted(c.candidate_id for c in group))
        drafts.append(
            (
                COMPARISON_KIND_ORDER.index(COMPARISON_PER_ASSEMBLY),
                ids,
                ComparisonRecord(
                    comparison_id="",
                    comparison_kind=COMPARISON_PER_ASSEMBLY,
                    candidate_ids=ids,
                    status=status,
                    splice_permitted=splice,
                    reasons=reasons,
                    reverse_verification_ids=v_ids,
                ),
            )
        )

    drafts.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        replace(record, comparison_id=f"cmp-{index:04d}")
        for index, (_, _, record) in enumerate(drafts, start=1)
    )


def build_comparison_ledger(
    candidates: Sequence[ComparisonCandidate | Mapping[str, Any]],
    *,
    reaction_id: str | None = None,
    reverse_verifications: Sequence[
        ReverseStartVerification | Mapping[str, Any]
    ] = (),
) -> DirectionTimingLedger:
    """Build the per-reaction direction/timing comparison ledger.

    Every candidate row carries ``start_endpoint/direction/assembly_id/
    anchor_reason``; comparisons are emitted for forward-vs-reverse, early-
    vs-late and per-assembly groups.  A reverse candidate without a verified
    :class:`ReverseStartVerification` yields
    ``incomplete_missing_reverse_verification`` + ``splice_permitted=False``
    (curves are never spliced without verification).
    """
    parsed_candidates = tuple(_parse_candidate(raw) for raw in candidates)
    seen_ids: set[str] = set()
    for candidate in parsed_candidates:
        if candidate.candidate_id in seen_ids:
            raise DirectionTimingCostError(
                CODE_CANDIDATE_INVALID,
                f"duplicate candidate_id={candidate.candidate_id!r}",
            )
        seen_ids.add(candidate.candidate_id)
    parsed_verifications = tuple(
        _parse_reverse_verification(raw) for raw in reverse_verifications
    )
    verification_by_id: dict[str, ReverseStartVerification] = {}
    for verification in parsed_verifications:
        if verification.candidate_id not in seen_ids:
            raise DirectionTimingCostError(
                CODE_REVERSE_VERIFICATION_INVALID,
                f"verification {verification.verification_id!r} references unknown "
                f"candidate {verification.candidate_id!r}",
            )
        verification_by_id[verification.candidate_id] = verification
    comparisons = _emit_comparisons(parsed_candidates, verification_by_id)
    return DirectionTimingLedger(
        reaction_id=reaction_id,
        candidates=parsed_candidates,
        comparisons=comparisons,
        reverse_verifications=parsed_verifications,
    )


def assert_splice_permitted(comparison: ComparisonRecord) -> None:
    """Refuse splicing when the ledger gate has not permitted it (typed)."""
    if not comparison.splice_permitted:
        raise DirectionTimingCostError(
            CODE_SPLICE_BLOCKED,
            f"{comparison.comparison_id}: status={comparison.status}; "
            f"reasons={list(comparison.reasons)}",
        )


# ---------------------------------------------------------------------------
# Cost accounting: budget extension + separate channels + counted/measured.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CountedOperations:
    """Counted operation ledger — counts only, never core-hours (§13.4)."""

    n_constrained_optimizations: int = 0
    n_point_retries: int = 0
    n_endpoint_preparations: int = 0
    n_directions: int = 0
    n_assemblies: int = 0
    n_schedules: int = 0

    @property
    def sum_constrained_optimizations(self) -> int:
        return self.n_constrained_optimizations

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "n_constrained_optimizations": self.n_constrained_optimizations,
            "n_point_retries": self.n_point_retries,
            "n_endpoint_preparations": self.n_endpoint_preparations,
            "n_directions": self.n_directions,
            "n_assemblies": self.n_assemblies,
            "n_schedules": self.n_schedules,
            "sum_constrained_optimizations": self.sum_constrained_optimizations,
            "equals_actual_core_hours": SUM_IS_NOT_CORE_HOURS,
            "note": COST_NOTE_COUNTED_VS_MEASURED,
        }


@dataclass(frozen=True, slots=True)
class MeasuredTime:
    """Measured CPU/wall time — receipt-gated; unknown until measured."""

    n_cpu_hours: float = 0.0
    cpu_time_status: str = _MEASURED_STATUS_UNKNOWN
    n_wall_seconds: float = 0.0
    wall_time_status: str = _MEASURED_STATUS_UNKNOWN
    receipt_ids: tuple[str, ...] = ()

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "n_cpu_hours": self.n_cpu_hours,
            "cpu_time_status": self.cpu_time_status,
            "n_wall_seconds": self.n_wall_seconds,
            "wall_time_status": self.wall_time_status,
            "receipt_ids": list(self.receipt_ids),
            "note": MEASURED_ONLY_FROM_RECEIPTS_NOTE,
        }


def zeroed_channels() -> tuple[MethodChannelAccounting, ...]:
    """Zeroed NEB + scan channels aligned with todo-26 branch vocabulary.

    Starting from these zeroed channels,
    ``CostAccounting.enter_method(CHANNEL_NEB)`` reproduces
    ``path_request.enter_neb_channels()`` exactly (semantic alignment test).
    """
    return (
        MethodChannelAccounting(
            method_kind=CHANNEL_NEB,
            n_entered=0,
            n_succeeded=0,
            n_failed=0,
            branch=BRANCH_NATIVE_SCAN_EXITED,
            cost_note=NEB_ENTER_EXITS_SCAN_BRANCH,
        ),
        MethodChannelAccounting(
            method_kind=CHANNEL_SCAN,
            n_entered=0,
            n_succeeded=0,
            n_failed=0,
            branch=BRANCH_NATIVE_SCAN,
            cost_note=SCAN_CHANNEL_COST_NOTE,
        ),
    )


def _nonnegative_int(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise DirectionTimingCostError(
            CODE_COUNT_INVALID, f"{field} must be a non-negative integer"
        )
    return raw


def _finite_nonnegative(raw: Any, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DirectionTimingCostError(
            CODE_MEASUREMENT_INVALID, f"{field} must be a number"
        )
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise DirectionTimingCostError(
            CODE_MEASUREMENT_INVALID, f"{field} must be finite and >= 0"
        )
    return value


@dataclass(frozen=True, slots=True)
class CostAccounting:
    """Per-reaction cost accounting: budget + channels + counted/measured.

    ``budget`` is the todo-23 :class:`BudgetLedger` extended additively
    (per-point optimization retries and endpoint preparations feed
    ``BudgetLedger.record``); ``channels`` are the todo-26 method channels
    (NEB and Scan never merged); ``counted`` vs ``measured`` keep
    ``Σ constrained_optimizations`` strictly separate from actual core-hours.
    """

    reaction_id: str | None
    budget: BudgetLedger
    channels: tuple[MethodChannelAccounting, ...]
    counted: CountedOperations
    measured: MeasuredTime

    # -- counted operations + budget extension (design §8.3) ---------------
    def record_constrained_optimizations(
        self,
        *,
        n: int,
        point_retries: int = 0,
        endpoint_preparations: int = 0,
    ) -> CostAccounting:
        """Count constrained optimizations; feed retries/prep into the budget."""
        n_ops = _nonnegative_int(n, "n")
        n_retries = _nonnegative_int(point_retries, "point_retries")
        n_prep = _nonnegative_int(endpoint_preparations, "endpoint_preparations")
        budget = self.budget.record(
            retries=n_retries, endpoint_preparations=n_prep
        )
        counted = replace(
            self.counted,
            n_constrained_optimizations=self.counted.n_constrained_optimizations
            + n_ops,
            n_point_retries=self.counted.n_point_retries + n_retries,
            n_endpoint_preparations=self.counted.n_endpoint_preparations + n_prep,
        )
        return replace(self, budget=budget, counted=counted)

    def record_directions(
        self,
        *,
        n_directions: int = 0,
        n_assemblies: int = 0,
        n_schedules: int = 0,
    ) -> CostAccounting:
        """Count direction/assembly/schedule exploration into budget + counts."""
        n_dir = _nonnegative_int(n_directions, "n_directions")
        n_asm = _nonnegative_int(n_assemblies, "n_assemblies")
        n_sch = _nonnegative_int(n_schedules, "n_schedules")
        budget = self.budget.record(
            directions=n_dir, assemblies=n_asm, schedules=n_sch
        )
        counted = replace(
            self.counted,
            n_directions=self.counted.n_directions + n_dir,
            n_assemblies=self.counted.n_assemblies + n_asm,
            n_schedules=self.counted.n_schedules + n_sch,
        )
        return replace(self, budget=budget, counted=counted)

    # -- separate method channels (design §8.3/§13.4; todo 26 vocabulary) --
    @property
    def channel(self) -> MethodChannelAccounting:
        """Zero-value scan channel accessor for convenience/tests."""
        for entry in self.channels:
            if entry.method_kind == CHANNEL_SCAN:
                return entry
        raise DirectionTimingCostError(
            CODE_METHOD_KIND_INVALID, "scan channel missing from accounting"
        )

    def channel_of(self, method_kind: str) -> MethodChannelAccounting:
        if method_kind not in METHOD_CHANNELS:
            raise DirectionTimingCostError(
                CODE_METHOD_KIND_INVALID,
                f"method_kind={method_kind!r}; expected one of {list(METHOD_CHANNELS)}",
            )
        for entry in self.channels:
            if entry.method_kind == method_kind:
                return entry
        raise DirectionTimingCostError(
            CODE_METHOD_KIND_INVALID, f"channel {method_kind!r} missing"
        )

    def enter_method(self, method_kind: str) -> CostAccounting:
        """Record a method-channel entry; NEB entry exits the native scan branch.

        Additive over the existing channels: only the addressed channel's
        ``n_entered`` grows; a NEB entry also stamps the scan channel with
        the branch-exit marker (entering NEB is never a scan success).
        """
        if method_kind not in METHOD_CHANNELS:
            raise DirectionTimingCostError(
                CODE_METHOD_KIND_INVALID,
                f"method_kind={method_kind!r}; expected one of {list(METHOD_CHANNELS)}",
            )
        out: list[MethodChannelAccounting] = []
        for entry in self.channels:
            if entry.method_kind != method_kind:
                if method_kind == CHANNEL_NEB and entry.method_kind == CHANNEL_SCAN:
                    out.append(
                        MethodChannelAccounting(
                            method_kind=entry.method_kind,
                            n_entered=entry.n_entered,
                            n_succeeded=entry.n_succeeded,
                            n_failed=entry.n_failed,
                            branch=BRANCH_NATIVE_SCAN_EXITED,
                            cost_note=SCAN_CHANNEL_COST_NOTE,
                            n_cpu_hours=entry.n_cpu_hours,
                        )
                    )
                else:
                    out.append(entry)
                continue
            if method_kind == CHANNEL_NEB:
                branch, note = BRANCH_NATIVE_SCAN_EXITED, NEB_ENTER_EXITS_SCAN_BRANCH
            else:
                branch, note = BRANCH_NATIVE_SCAN, SCAN_CHANNEL_COST_NOTE
            out.append(
                MethodChannelAccounting(
                    method_kind=entry.method_kind,
                    n_entered=entry.n_entered + 1,
                    n_succeeded=entry.n_succeeded,
                    n_failed=entry.n_failed,
                    branch=branch,
                    cost_note=note,
                    n_cpu_hours=entry.n_cpu_hours,
                )
            )
        return replace(self, channels=tuple(out))

    def record_outcome(self, method_kind: str, *, succeeded: bool) -> CostAccounting:
        """Record one outcome on ``method_kind`` only (todo-26 single-source)."""
        if method_kind not in METHOD_CHANNELS:
            raise DirectionTimingCostError(
                CODE_METHOD_KIND_INVALID,
                f"method_kind={method_kind!r}; expected one of {list(METHOD_CHANNELS)}",
            )
        return replace(
            self,
            channels=record_channel_outcome(
                self.channels, method_kind, succeeded=succeeded
            ),
        )

    # -- measured time: receipt-gated, never estimated (design §13.4) ------
    def record_measured_time(
        self,
        *,
        receipt_id: str,
        n_cpu_hours: float | None = None,
        n_wall_seconds: float | None = None,
    ) -> CostAccounting:
        """Record measured CPU/wall time only from an actual receipt."""
        if not isinstance(receipt_id, str) or not receipt_id.strip():
            raise DirectionTimingCostError(
                CODE_MEASUREMENT_REQUIRES_RECEIPT,
                "receipt_id required; measured time is never estimated silently",
            )
        cpu = (
            self.measured.n_cpu_hours
            if n_cpu_hours is None
            else _finite_nonnegative(n_cpu_hours, "n_cpu_hours")
        )
        wall = (
            self.measured.n_wall_seconds
            if n_wall_seconds is None
            else _finite_nonnegative(n_wall_seconds, "n_wall_seconds")
        )
        cpu_status = (
            _MEASURED_STATUS_MEASURED
            if (n_cpu_hours is not None or self.measured.cpu_time_status == _MEASURED_STATUS_MEASURED)
            else _MEASURED_STATUS_UNKNOWN
        )
        wall_status = (
            _MEASURED_STATUS_MEASURED
            if (n_wall_seconds is not None or self.measured.wall_time_status == _MEASURED_STATUS_MEASURED)
            else _MEASURED_STATUS_UNKNOWN
        )
        measured = MeasuredTime(
            n_cpu_hours=cpu,
            cpu_time_status=cpu_status,
            n_wall_seconds=wall,
            wall_time_status=wall_status,
            receipt_ids=(*self.measured.receipt_ids, receipt_id),
        )
        return replace(self, measured=measured)

    # -- cost report: counted vs measured (design §13.4) -------------------
    def cost_report(self) -> dict[str, JSONValue]:
        """Distinguish counted operations from measured time.

        ``Σ constrained_optimizations`` lives under ``counted_operations``
        with ``equals_actual_core_hours=false``; measured CPU/wall time lives
        under ``measured_time`` with explicit ``*_time_status`` (``unknown``
        until a receipt is recorded) — never estimated silently.
        """
        return {
            "schema_name": OBJECT_COST_REPORT,
            "schema_version": SCHEMA_COST_REPORT,
            "reaction_id": self.reaction_id,
            "counted_operations": self.counted.to_doc(),
            "measured_time": self.measured.to_doc(),
            "method_channels": {
                entry.method_kind: entry.to_doc() for entry in self.channels
            },
            "budget": {
                "spent": self.budget.spent_summary(),
                "max": self.budget.max_summary(),
                "exhausted": self.budget.exhausted(),
            },
            "splice_policy": SPLICE_POLICY,
        }

    def cost_report_json(self) -> str:
        return stable_json_dumps(self.cost_report())


def cost_accounting(
    reaction_id: str | None = None,
    *,
    max_attempts: int = 0,
    budget: BudgetLedger | None = None,
    channels: Sequence[MethodChannelAccounting] | None = None,
) -> CostAccounting:
    """Fresh cost accounting; zeroed todo-26 channels + todo-23 budget."""
    active_budget = (
        budget
        if budget is not None
        else BudgetLedger(max_attempts=_nonnegative_int(max_attempts, "max_attempts"))
    )
    active_channels = (
        tuple(channels) if channels is not None else zeroed_channels()
    )
    return CostAccounting(
        reaction_id=reaction_id,
        budget=active_budget,
        channels=active_channels,
        counted=CountedOperations(),
        measured=MeasuredTime(),
    )


def enter_neb_snapshot() -> tuple[MethodChannelAccounting, ...]:
    """Todo-26 snapshot for a reaction entering NEB (imported, not copied)."""
    return enter_neb_channels()


__all__ = [
    "BRANCH_NATIVE_SCAN",
    "BRANCH_NATIVE_SCAN_EXITED",
    "CHANNEL_NEB",
    "CHANNEL_SCAN",
    "CODE_CANDIDATE_INVALID",
    "CODE_COUNT_INVALID",
    "CODE_DIRECTION_MISMATCH",
    "CODE_ENDPOINT_INVALID",
    "CODE_MEASUREMENT_INVALID",
    "CODE_MEASUREMENT_REQUIRES_RECEIPT",
    "CODE_METHOD_KIND_INVALID",
    "CODE_REVERSE_VERIFICATION_INVALID",
    "CODE_SPLICE_BLOCKED",
    "COMPARISON_EARLY_VS_LATE",
    "COMPARISON_FORWARD_VS_REVERSE",
    "COMPARISON_KIND_ORDER",
    "COMPARISON_PER_ASSEMBLY",
    "COST_NOTE_COUNTED_VS_MEASURED",
    "MEASURED_ONLY_FROM_RECEIPTS_NOTE",
    "OBJECT_COST_REPORT",
    "OBJECT_DIRECTION_TIMING_LEDGER",
    "REASON_NO_REVERSE_PARTICIPANT",
    "REASON_REVERSE_ASSEMBLY_UNVERIFIED",
    "REASON_REVERSE_START_OPTIMIZATION_FAILED",
    "REASON_REVERSE_VERIFICATION_REQUIRED",
    "REASON_REVERSE_VERIFIED",
    "REASON_TIMING_COMPARISON_NO_SPLICE",
    "SCHEDULE_KIND_EARLY",
    "SCHEDULE_KIND_LATE",
    "SCHEMA_COST_REPORT",
    "SCHEMA_DIRECTION_TIMING_LEDGER",
    "SPLICE_POLICY",
    "STATUS_COMPLETE",
    "STATUS_INCOMPLETE_MISSING_REVERSE_VERIFICATION",
    "SUM_IS_NOT_CORE_HOURS",
    "TIMING_SCHEDULE_KINDS",
    "ComparisonCandidate",
    "ComparisonRecord",
    "CostAccounting",
    "CountedOperations",
    "DirectionTimingCostError",
    "DirectionTimingLedger",
    "MeasuredTime",
    "ReverseStartVerification",
    "assert_splice_permitted",
    "build_comparison_ledger",
    "cost_accounting",
    "enter_neb_snapshot",
    "zeroed_channels",
]
