"""Stratified comparison, version lock, and the scientific report (todo 30).

Implements design §13.4 / §12 and 裁决记录 §7.5:

- **Stratification discipline.** L1 is limited to Demo24 + subsets; the full
  ``scan_ready`` population (183,460) may only be consumed AFTER the L4
  version lock.  :func:`assert_population_unlocked` refuses full-population
  runs at L1–L3 (and at L4 before a :class:`VersionLock` exists) with typed
  codes — the 183k population is never executed here, only guarded.
- **Comparison arms under a fixed shared budget** (§13.4): old single-B (v1
  baseline), graph-theory single-B, coupled-1D, limited timing/direction
  variants, and NEB.  Arms are *data* (:data:`COMPARISON_ARMS`); report
  assembly is pure.  Every arm reports its new target-path rate and cost
  under the SAME shared budget (:class:`SharedBudget`); a budget mismatch
  across arms is a typed refusal.
- **§13.4 report fields, per cohort:** N total, N review-pending refusals,
  N compilable, N executed, N numerically usable, N target-path compatible,
  N strict TS, core-hours, unit-success cost, and the failure-code
  histogram.  Reaction counts are the statistical unit; candidate counts
  live in SEPARATE denominator fields and never masquerade as reaction
  successes.
- **16 train / 8 valid discipline:** train drives development; the 8 valid
  rows are locked after input re-check and must not be used for tuning.
  Any report consuming valid-split rows carries ``valid_used=true`` plus a
  timestamp and the input-set hash; :func:`assert_valid_not_tuned` refuses
  parameter changes after a valid-touching run.
- **Demo24 is a mechanism-coverage development set** (§12/§13.4): reports on
  it must carry ``scope="development_mechanism_coverage"``, never
  ``population_estimate``.  Whole-dataset success rates must never be
  estimated from these 24 reactions.
- **truth_assisted_p1 declaration.** Every input provenance block states
  ``mapping_provenance`` explicitly (single source:
  ``pes2ts_core.g1.v2_schema.MAPPING_PROVENANCE``); a missing declaration is
  a typed refusal.
- :func:`build_scientific_report` assembles a sealed ``ScientificReport``
  document from arm results + config (pure, deterministic).
  :func:`lock_version` freezes strategy versions + config into a
  :class:`VersionLock` record whose digest changes when either changes.

# allow: SIZE_OK — plan-named todo-30 single module (stratification guards +
# comparison arms + report assembly + version lock); precedent
# path_request.py / plan_freeze.py / direction_timing_cost.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from pes2ts_core.contracts import seal_document
from pes2ts_core.g1.v2_schema import MAPPING_PROVENANCE
from pes2ts_core.scan_strategy.direction_timing_cost import COST_NOTE_COUNTED_VS_MEASURED
from pes2ts_core.scan_strategy.path_request import (
    CHANNEL_NEB,
    NEB_ENTER_EXITS_SCAN_BRANCH,
    PATH_REQUEST_VERSION,
)
from pes2ts_core.scan_strategy.plan_freeze import (
    FAILURE_TREE_VERSION,
    PLAN_FREEZE_VERSION,
    REGISTRY_VERSION,
    SCHEDULE_VERSION,
)
from pes2ts_core.scan_strategy.special_domain import DEMO24_COVERAGE
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, stable_json_dumps

# ---------------------------------------------------------------------------
# Vocabulary: stages, arms, scopes, cohorts, schema, population.
# ---------------------------------------------------------------------------
SCHEMA_SCIENTIFIC_REPORT: Final[str] = "g1_scientific_report_v1"
OBJECT_SCIENTIFIC_REPORT: Final[str] = "ScientificReport"
SCHEMA_VERSION_LOCK: Final[str] = "g1_scan_strategy_version_lock_v1"
OBJECT_VERSION_LOCK: Final[str] = "VersionLock"

STAGE_L1: Final[str] = "L1"
STAGE_L2: Final[str] = "L2"
STAGE_L3: Final[str] = "L3"
STAGE_L4: Final[str] = "L4"
STAGES: Final[tuple[str, ...]] = (STAGE_L1, STAGE_L2, STAGE_L3, STAGE_L4)

#: v2 ``scan_ready`` population (裁决记录 §3; full run only after L4 lock).
POPULATION_SCAN_READY_N: Final[int] = 183_460

#: Demo24 cohort constants (16 train / 8 valid discipline, design §13.4).
N_DEMO_TOTAL: Final[int] = 24
N_TRAIN_DEMO: Final[int] = 16
N_VALID_DEMO: Final[int] = 8

COHORT_DEMO24: Final[str] = "demo24"
COHORT_TRAIN16: Final[str] = "train16"
COHORT_VALID8: Final[str] = "valid8"
COHORT_FULL_POPULATION: Final[str] = "full_population"

#: Named cohorts carry exact reaction counts (machine-checkable discipline).
_COHORT_EXACT_N: Final[dict[str, int]] = {
    COHORT_DEMO24: N_DEMO_TOTAL,
    COHORT_TRAIN16: N_TRAIN_DEMO,
    COHORT_VALID8: N_VALID_DEMO,
    COHORT_FULL_POPULATION: POPULATION_SCAN_READY_N,
}

SCOPE_DEVELOPMENT_MECHANISM_COVERAGE: Final[str] = "development_mechanism_coverage"
SCOPE_POPULATION_ESTIMATE: Final[str] = "population_estimate"

#: Comparison arms (design §13.4) — data, never code branches.
ARM_OLD_SINGLE_B: Final[str] = "old_single_b_v1"
ARM_GRAPH_SINGLE_B: Final[str] = "graph_theory_single_b"
ARM_COUPLED_1D: Final[str] = "coupled_1d"
ARM_TIMING_DIRECTION: Final[str] = "timing_direction_variants"
ARM_PATH_NEB: Final[str] = "path_neb"
COMPARISON_ARMS: Final[tuple[str, ...]] = (
    ARM_OLD_SINGLE_B,
    ARM_GRAPH_SINGLE_B,
    ARM_COUPLED_1D,
    ARM_TIMING_DIRECTION,
    ARM_PATH_NEB,
)
ARM_LABELS: Final[dict[str, str]] = {
    ARM_OLD_SINGLE_B: "old single-B (v1 baseline ScanPlan)",
    ARM_GRAPH_SINGLE_B: "graph-theory single-B",
    ARM_COUPLED_1D: "coupled-1D multi-coordinate",
    ARM_TIMING_DIRECTION: "limited timing/direction variants",
    ARM_PATH_NEB: "NEB path (separate cost/success channel)",
}

#: Machine-checkable notes carried into every assembled report.
DENOMINATOR_NOTE: Final[str] = (
    "reaction counts are the statistical unit; candidate counts are a "
    "separate denominator and never substitute for reaction successes "
    "(design §13.4)"
)
VALID_LOCK_NOTE: Final[str] = (
    "16 train rows drive development; the 8 valid rows are locked after "
    "input re-check and must not be used to tune strategy parameters "
    "(design §13.4)"
)
DEMO24_SCOPE_NOTE: Final[str] = (
    "Demo24 is a mechanism-coverage development set; whole-dataset success "
    "rates must never be estimated from these 24 reactions (design §12/§13.4)"
)
NEB_ARM_NOTE: Final[str] = (
    f"{NEB_ENTER_EXITS_SCAN_BRANCH}; the NEB arm reports its own cost and "
    "success channel"
)

# ---------------------------------------------------------------------------
# Typed refusal codes (closed vocabulary).
# ---------------------------------------------------------------------------
CODE_STAGE_UNKNOWN: Final[str] = "STAGE_UNKNOWN"
CODE_POPULATION_LOCKED_BEFORE_L4: Final[str] = "POPULATION_LOCKED_BEFORE_L4"
CODE_POPULATION_LOCK_REQUIRED: Final[str] = "POPULATION_LOCK_REQUIRED"
CODE_DEMO24_SCOPE_MISMATCH: Final[str] = "DEMO24_SCOPE_MISMATCH"
CODE_VALID_USE_UNDECLARED: Final[str] = "VALID_USE_UNDECLARED"
CODE_VALID_TUNED_AFTER_VALID_RUN: Final[str] = "VALID_TUNED_AFTER_VALID_RUN"
CODE_PROVENANCE_MAPPING_UNDECLARED: Final[str] = "PROVENANCE_MAPPING_UNDECLARED"
CODE_ARM_UNKNOWN: Final[str] = "ARM_UNKNOWN"
CODE_BUDGET_MISMATCH: Final[str] = "BUDGET_MISMATCH"
CODE_DENOMINATOR_INVALID: Final[str] = "DENOMINATOR_INVALID"
CODE_COHORT_INVALID: Final[str] = "COHORT_INVALID"
CODE_LOCK_INPUT_INVALID: Final[str] = "LOCK_INPUT_INVALID"
CODE_REPORT_INPUT_INVALID: Final[str] = "REPORT_INPUT_INVALID"


class ScientificReportError(ValueError):
    """Typed stratification/report refusal; ``code`` is a stable machine token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Frozen input types (parse-don't-validate at the trust boundary).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SharedBudget:
    """Fixed shared budget every comparison arm must run under (§13.4)."""

    max_attempts: int
    max_cpu_hours: float
    max_wall_seconds: int
    max_total_candidates_scan: int
    max_total_candidates_path_neb: int

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "max_attempts": self.max_attempts,
            "max_cpu_hours": self.max_cpu_hours,
            "max_wall_seconds": self.max_wall_seconds,
            "max_total_candidates_scan": self.max_total_candidates_scan,
            "max_total_candidates_path_neb": self.max_total_candidates_path_neb,
        }


@dataclass(frozen=True, slots=True)
class CohortResult:
    """Raw per-cohort outcome counts for one comparison arm."""

    cohort_id: str
    scope: str
    n_total: int
    n_review_pending_refusals: int
    n_compilable: int
    n_executed: int
    n_numerically_usable: int
    n_target_path_compatible: int
    n_strict_ts: int
    n_candidates_total: int = 0
    n_candidates_target_path_compatible: int = 0
    core_hours: float = 0.0
    failure_code_histogram: Mapping[str, int] | None = None
    consumes_valid: bool = False
    valid_used_at: str | None = None
    valid_input_set_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One comparison arm's data under the shared budget."""

    arm_id: str
    budget: SharedBudget
    cohorts: tuple[CohortResult, ...]
    core_hours: float


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Input provenance block; ``mapping_provenance`` must be explicit."""

    source: str
    mapping_provenance: str
    input_set_sha256: str | None = None
    note: str = ""

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "source": self.source,
            "mapping_provenance": self.mapping_provenance,
            "input_set_sha256": self.input_set_sha256,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class VersionLock:
    """Frozen strategy-versions + config record (L4 lock evidence)."""

    report_id: str
    strategy_versions: Mapping[str, str]
    config: Mapping[str, Any]
    content_sha256: str
    created_at: str

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "schema_name": OBJECT_VERSION_LOCK,
            "schema_version": SCHEMA_VERSION_LOCK,
            "report_id": self.report_id,
            "strategy_versions": dict(self.strategy_versions),
            "config": dict(self.config),
            "content_sha256": self.content_sha256,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class RunLogEntry:
    """One execution-layer run record for the valid-tuning guard."""

    run_id: str
    timestamp: str
    split: str
    parameters: Mapping[str, Any]
    input_set_sha256: str | None = None

    @property
    def touched_valid(self) -> bool:
        return self.split == "valid"


# ---------------------------------------------------------------------------
# Boundary coercion helpers.
# ---------------------------------------------------------------------------
def _require_int(raw: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ScientificReportError(
            CODE_REPORT_INPUT_INVALID, f"{field} must be an int >= {minimum}"
        )
    return raw


def _require_float(raw: Any, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, f"{field} must be a number")
    value = float(raw)
    if value < 0:
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, f"{field} must be >= 0")
    return value


def _require_str(raw: Any, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, f"{field} must be a non-empty string")
    return raw


def _params_key(parameters: Mapping[str, Any]) -> str:
    """Canonical key for one parameter snapshot (raises on non-JSON values)."""
    if not isinstance(parameters, Mapping):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "parameters must be a mapping")
    try:
        return stable_json_dumps(dict(parameters))
    except (TypeError, ValueError) as exc:
        raise ScientificReportError(
            CODE_REPORT_INPUT_INVALID, f"parameters must be JSON-serializable: {exc}"
        ) from exc


def _coerce_budget(raw: SharedBudget | Mapping[str, Any]) -> SharedBudget:
    if isinstance(raw, SharedBudget):
        return raw
    if not isinstance(raw, Mapping):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "budget must be a mapping")
    return SharedBudget(
        max_attempts=_require_int(raw.get("max_attempts"), "budget.max_attempts"),
        max_cpu_hours=_require_float(raw.get("max_cpu_hours"), "budget.max_cpu_hours"),
        max_wall_seconds=_require_int(raw.get("max_wall_seconds"), "budget.max_wall_seconds"),
        max_total_candidates_scan=_require_int(
            raw.get("max_total_candidates_scan"), "budget.max_total_candidates_scan"
        ),
        max_total_candidates_path_neb=_require_int(
            raw.get("max_total_candidates_path_neb"), "budget.max_total_candidates_path_neb"
        ),
    )


def _coerce_histogram(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ScientificReportError(CODE_COHORT_INVALID, "failure_code_histogram must be a mapping")
    histogram: dict[str, int] = {}
    for code, count in raw.items():
        if not isinstance(code, str) or not code.strip():
            raise ScientificReportError(
                CODE_COHORT_INVALID, "failure_code_histogram keys must be non-empty strings"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ScientificReportError(
                CODE_COHORT_INVALID, f"failure_code_histogram[{code!r}] must be an int >= 0"
            )
        histogram[code] = count
    return histogram


def _coerce_cohort(raw: CohortResult | Mapping[str, Any]) -> CohortResult:
    if isinstance(raw, CohortResult):
        result = raw
    elif isinstance(raw, Mapping):
        result = CohortResult(
            cohort_id=_require_str(raw.get("cohort_id"), "cohort.cohort_id"),
            scope=_require_str(raw.get("scope"), "cohort.scope"),
            n_total=_require_int(raw.get("n_total"), "cohort.n_total"),
            n_review_pending_refusals=_require_int(
                raw.get("n_review_pending_refusals"), "cohort.n_review_pending_refusals"
            ),
            n_compilable=_require_int(raw.get("n_compilable"), "cohort.n_compilable"),
            n_executed=_require_int(raw.get("n_executed"), "cohort.n_executed"),
            n_numerically_usable=_require_int(
                raw.get("n_numerically_usable"), "cohort.n_numerically_usable"
            ),
            n_target_path_compatible=_require_int(
                raw.get("n_target_path_compatible"), "cohort.n_target_path_compatible"
            ),
            n_strict_ts=_require_int(raw.get("n_strict_ts"), "cohort.n_strict_ts"),
            n_candidates_total=_require_int(
                raw.get("n_candidates_total", 0), "cohort.n_candidates_total"
            ),
            n_candidates_target_path_compatible=_require_int(
                raw.get("n_candidates_target_path_compatible", 0),
                "cohort.n_candidates_target_path_compatible",
            ),
            core_hours=_require_float(raw.get("core_hours", 0.0), "cohort.core_hours"),
            failure_code_histogram=raw.get("failure_code_histogram"),
            consumes_valid=bool(raw.get("consumes_valid", False)),
            valid_used_at=raw.get("valid_used_at"),
            valid_input_set_sha256=raw.get("valid_input_set_sha256"),
        )
    else:
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "cohort result must be a mapping")

    exact_n = _COHORT_EXACT_N.get(result.cohort_id)
    if exact_n is not None and result.n_total != exact_n:
        raise ScientificReportError(
            CODE_COHORT_INVALID,
            f"cohort {result.cohort_id!r} must carry exactly {exact_n} reactions, got {result.n_total}",
        )
    # §13.4 funnel: total >= review-refusals + compilable; each tier nests.
    if result.n_compilable > result.n_total:
        raise ScientificReportError(
            CODE_DENOMINATOR_INVALID, "n_compilable must not exceed n_total"
        )
    if result.n_executed > result.n_compilable:
        raise ScientificReportError(
            CODE_DENOMINATOR_INVALID, "n_executed must not exceed n_compilable"
        )
    if result.n_numerically_usable > result.n_executed:
        raise ScientificReportError(
            CODE_DENOMINATOR_INVALID, "n_numerically_usable must not exceed n_executed"
        )
    if result.n_target_path_compatible > result.n_numerically_usable:
        raise ScientificReportError(
            CODE_DENOMINATOR_INVALID,
            "n_target_path_compatible must not exceed n_numerically_usable",
        )
    if result.n_strict_ts > result.n_target_path_compatible:
        raise ScientificReportError(
            CODE_DENOMINATOR_INVALID, "n_strict_ts must not exceed n_target_path_compatible"
        )
    if result.n_candidates_target_path_compatible > result.n_candidates_total:
        raise ScientificReportError(
            CODE_DENOMINATOR_INVALID,
            "n_candidates_target_path_compatible must not exceed n_candidates_total",
        )
    return result


def _coerce_arm(raw: ArmResult | Mapping[str, Any]) -> ArmResult:
    if isinstance(raw, ArmResult):
        arm = raw
    elif isinstance(raw, Mapping):
        cohorts_raw = raw.get("cohorts")
        if not isinstance(cohorts_raw, Sequence) or isinstance(cohorts_raw, (str, bytes)):
            raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "arm.cohorts must be a sequence")
        arm = ArmResult(
            arm_id=_require_str(raw.get("arm_id"), "arm.arm_id"),
            budget=_coerce_budget(raw.get("budget")),
            cohorts=tuple(_coerce_cohort(entry) for entry in cohorts_raw),
            core_hours=_require_float(raw.get("core_hours", 0.0), "arm.core_hours"),
        )
    else:
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "arm result must be a mapping")
    if arm.arm_id not in COMPARISON_ARMS:
        raise ScientificReportError(
            CODE_ARM_UNKNOWN,
            f"arm_id={arm.arm_id!r}; expected one of {list(COMPARISON_ARMS)}",
        )
    if not arm.cohorts:
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, f"arm {arm.arm_id!r} has no cohorts")
    for cohort in arm.cohorts:
        _coerce_cohort(cohort)
    return arm


def _coerce_provenance(raw: InputProvenance | Mapping[str, Any]) -> InputProvenance:
    if isinstance(raw, InputProvenance):
        block = raw
    elif isinstance(raw, Mapping):
        mapping_raw = raw.get("mapping_provenance")
        block = InputProvenance(
            source=_require_str(raw.get("source"), "provenance.source"),
            mapping_provenance=(
                mapping_raw if isinstance(mapping_raw, str) else ""
            ),
            input_set_sha256=raw.get("input_set_sha256"),
            note=str(raw.get("note") or ""),
        )
    else:
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "provenance must be a mapping")
    if not isinstance(block.mapping_provenance, str) or not block.mapping_provenance.strip():
        raise ScientificReportError(
            CODE_PROVENANCE_MAPPING_UNDECLARED,
            f"provenance source={block.source!r} must state mapping_provenance explicitly "
            f"(expected e.g. {MAPPING_PROVENANCE!r} from the bundle/proposal docs)",
        )
    return block


def _coerce_run_entry(raw: RunLogEntry | Mapping[str, Any]) -> RunLogEntry:
    if isinstance(raw, RunLogEntry):
        return raw
    if not isinstance(raw, Mapping):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "run-log entry must be a mapping")
    parameters = raw.get("parameters")
    if parameters is None:
        parameters = {}
    return RunLogEntry(
        run_id=_require_str(raw.get("run_id"), "run.run_id"),
        timestamp=_require_str(raw.get("timestamp"), "run.timestamp"),
        split=_require_str(raw.get("split"), "run.split"),
        parameters=parameters if isinstance(parameters, Mapping) else {},
        input_set_sha256=raw.get("input_set_sha256"),
    )


def _coerce_lock(raw: VersionLock | Mapping[str, Any]) -> VersionLock:
    if isinstance(raw, VersionLock):
        return raw
    if not isinstance(raw, Mapping):
        raise ScientificReportError(CODE_LOCK_INPUT_INVALID, "lock must be a mapping")
    versions = raw.get("strategy_versions")
    if not isinstance(versions, Mapping) or not versions:
        raise ScientificReportError(
            CODE_LOCK_INPUT_INVALID, "lock.strategy_versions must be a non-empty mapping"
        )
    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise ScientificReportError(CODE_LOCK_INPUT_INVALID, "lock.config must be a mapping")
    return VersionLock(
        report_id=_require_str(raw.get("report_id"), "lock.report_id"),
        strategy_versions={str(k): str(v) for k, v in versions.items()},
        config=dict(config),
        content_sha256=_require_str(raw.get("content_sha256"), "lock.content_sha256"),
        created_at=_require_str(raw.get("created_at"), "lock.created_at"),
    )


# ---------------------------------------------------------------------------
# Stratification guards (typed refusals, never silent).
# ---------------------------------------------------------------------------
def assert_population_unlocked(
    stage: str,
    *,
    lock: VersionLock | Mapping[str, Any] | None = None,
) -> VersionLock:
    """Refuse full-population (183,460 scan_ready) runs before the L4 lock.

    L1–L3 are Demo24 + subset stratification only (裁决记录 §7.5).  At L4 a
    :class:`VersionLock` record must exist before the full population may be
    consumed.  The guard never executes the population itself.
    """
    if stage not in STAGES:
        raise ScientificReportError(
            CODE_STAGE_UNKNOWN, f"stage={stage!r}; expected one of {list(STAGES)}"
        )
    if stage != STAGE_L4:
        raise ScientificReportError(
            CODE_POPULATION_LOCKED_BEFORE_L4,
            f"stage={stage}: full population ({POPULATION_SCAN_READY_N} scan_ready) is "
            "locked until the L4 version lock; L1 is Demo24 + subsets only",
        )
    if lock is None:
        raise ScientificReportError(
            CODE_POPULATION_LOCK_REQUIRED,
            "L4 stage requires a VersionLock record before full-population runs",
        )
    return _coerce_lock(lock)


def assert_valid_not_tuned(
    run_log: Sequence[RunLogEntry | Mapping[str, Any]],
) -> None:
    """Refuse strategy-parameter changes after any valid-split-touching run.

    The 8 valid rows are locked after input re-check (design §13.4).  The
    first valid-touching entry freezes its parameter snapshot; any later
    entry whose parameters differ is tuning after touching valid → error.
    An untouched valid set (no ``split == "valid"`` entry) passes.  Run-log
    order is chronological; timestamps are carried for audit, not re-sorted.
    """
    entries = [_coerce_run_entry(entry) for entry in run_log]
    valid_indices = [index for index, entry in enumerate(entries) if entry.touched_valid]
    if not valid_indices:
        return
    first_valid = valid_indices[0]
    locked_key = _params_key(entries[first_valid].parameters)
    for entry in entries[first_valid + 1 :]:
        if _params_key(entry.parameters) != locked_key:
            raise ScientificReportError(
                CODE_VALID_TUNED_AFTER_VALID_RUN,
                f"run {entry.run_id!r} at {entry.timestamp} changes strategy parameters "
                f"after valid-touching run {entries[first_valid].run_id!r}; valid rows "
                "must stay locked — re-declare the valid set as development and build "
                "an independent validation set (design §13.4)",
            )


def assert_demo24_scope(cohort_id: str, scope: str) -> None:
    """Demo24 reports must carry the development-mechanism-coverage scope."""
    if cohort_id != COHORT_DEMO24:
        return
    if scope == SCOPE_POPULATION_ESTIMATE:
        raise ScientificReportError(
            CODE_DEMO24_SCOPE_MISMATCH,
            f"cohort {COHORT_DEMO24!r} must carry scope={SCOPE_DEVELOPMENT_MECHANISM_COVERAGE!r}, "
            f"not {SCOPE_POPULATION_ESTIMATE!r}; {DEMO24_SCOPE_NOTE}",
        )


# ---------------------------------------------------------------------------
# Version lock.
# ---------------------------------------------------------------------------
def default_strategy_versions() -> dict[str, str]:
    """Strategy versions frozen by default into a :class:`VersionLock`."""
    return {
        "registry_version": REGISTRY_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "failure_tree_version": FAILURE_TREE_VERSION,
        "plan_freeze_version": PLAN_FREEZE_VERSION,
        "path_request_version": PATH_REQUEST_VERSION,
        "scientific_report_schema": SCHEMA_SCIENTIFIC_REPORT,
        "mapping_provenance": MAPPING_PROVENANCE,
    }


def lock_version(
    report_id: str,
    *,
    strategy_versions: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> VersionLock:
    """Freeze strategy versions + config into a lock record.

    The record digest covers ``report_id``, ``strategy_versions`` and
    ``config``; changing either input changes ``content_sha256``.  Default
    versions come from the live scan-strategy modules
    (:func:`default_strategy_versions`).
    """
    report_id = _require_str(report_id, "report_id")
    versions_raw = (
        dict(strategy_versions) if strategy_versions is not None else default_strategy_versions()
    )
    if not versions_raw:
        raise ScientificReportError(
            CODE_LOCK_INPUT_INVALID, "strategy_versions must be a non-empty mapping"
        )
    versions = {str(k): str(v) for k, v in versions_raw.items()}
    config_snapshot = dict(config) if config is not None else {}
    payload = {
        "report_id": report_id,
        "strategy_versions": versions,
        "config": config_snapshot,
    }
    digest = sha256_bytes(stable_json_dumps(payload).encode("utf-8"))
    return VersionLock(
        report_id=report_id,
        strategy_versions=versions,
        config=config_snapshot,
        content_sha256=digest,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Report assembly (pure; arms are data).
# ---------------------------------------------------------------------------
def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _unit_success_cost(core_hours: float, n_successes: int) -> float | None:
    if n_successes <= 0 or core_hours <= 0:
        return None
    return core_hours / n_successes


def _project_cohort(cohort: CohortResult) -> dict[str, JSONValue]:
    """Project one cohort into the §13.4 field set (reaction vs candidate)."""
    histogram = _coerce_histogram(cohort.failure_code_histogram)
    return {
        "cohort_id": cohort.cohort_id,
        "scope": cohort.scope,
        "n_total": cohort.n_total,
        "n_review_pending_refusals": cohort.n_review_pending_refusals,
        "n_compilable": cohort.n_compilable,
        "n_executed": cohort.n_executed,
        "n_numerically_usable": cohort.n_numerically_usable,
        "n_target_path_compatible": cohort.n_target_path_compatible,
        "n_strict_ts": cohort.n_strict_ts,
        "core_hours": cohort.core_hours,
        "unit_success_cost": _unit_success_cost(
            cohort.core_hours, cohort.n_target_path_compatible
        ),
        "failure_code_histogram": histogram,
        "n_candidates_total": cohort.n_candidates_total,
        "n_candidates_target_path_compatible": cohort.n_candidates_target_path_compatible,
        "reaction_target_path_rate": _rate(cohort.n_target_path_compatible, cohort.n_total),
        "candidate_target_path_rate": _rate(
            cohort.n_candidates_target_path_compatible, cohort.n_candidates_total
        ),
        "consumes_valid": cohort.consumes_valid,
        "valid_used_at": cohort.valid_used_at,
        "valid_input_set_sha256": cohort.valid_input_set_sha256,
        "denominator_note": DENOMINATOR_NOTE,
    }


def _project_arm(arm: ArmResult, budget_key: str) -> dict[str, JSONValue]:
    """Project one arm: new target-path rate + cost under the shared budget."""
    n_reactions_total = sum(cohort.n_total for cohort in arm.cohorts)
    n_reactions_target_path = sum(cohort.n_target_path_compatible for cohort in arm.cohorts)
    n_candidates_total = sum(cohort.n_candidates_total for cohort in arm.cohorts)
    n_candidates_target_path = sum(
        cohort.n_candidates_target_path_compatible for cohort in arm.cohorts
    )
    arm_doc: dict[str, JSONValue] = {
        "arm_id": arm.arm_id,
        "arm_label": ARM_LABELS[arm.arm_id],
        "shared_budget_key": budget_key,
        "budget": arm.budget.to_doc(),
        "n_reactions_total": n_reactions_total,
        "n_reactions_target_path_compatible": n_reactions_target_path,
        "new_target_path_rate": _rate(n_reactions_target_path, n_reactions_total),
        "n_candidates_total": n_candidates_total,
        "n_candidates_target_path_compatible": n_candidates_target_path,
        "candidate_target_path_rate": _rate(n_candidates_target_path, n_candidates_total),
        "core_hours": arm.core_hours,
        "unit_success_cost": _unit_success_cost(arm.core_hours, n_reactions_target_path),
        "cohorts": [_project_cohort(cohort) for cohort in arm.cohorts],
        "denominator_note": DENOMINATOR_NOTE,
    }
    if arm.arm_id == ARM_PATH_NEB:
        arm_doc["method_channel"] = CHANNEL_NEB
        arm_doc["channel_note"] = NEB_ARM_NOTE
    return arm_doc


def build_scientific_report(
    arm_results: Sequence[ArmResult | Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    report_id: str | None = None,
    stage: str = STAGE_L1,
    input_provenance: Sequence[InputProvenance | Mapping[str, Any]] | None = None,
    population_lock: VersionLock | Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a sealed ``ScientificReport`` document (pure, deterministic).

    Arms are data; assembly performs no I/O.  Typed refusals cover: unknown
    arm ids, budget mismatch across arms (the shared budget must be fixed),
    Demo24 scope mislabelling, undeclared valid use (timestamp + input-set
    hash required), undeclared ``mapping_provenance``, and any
    ``population_estimate`` cohort consumed before the L4 lock.
    """
    if stage not in STAGES:
        raise ScientificReportError(
            CODE_STAGE_UNKNOWN, f"stage={stage!r}; expected one of {list(STAGES)}"
        )
    if not isinstance(config, Mapping):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "config must be a mapping")
    arms_raw = list(arm_results)
    if not arms_raw:
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "arm_results must be non-empty")
    arms = [_coerce_arm(entry) for entry in arms_raw]

    seen_arm_ids = [arm.arm_id for arm in arms]
    if len(set(seen_arm_ids)) != len(seen_arm_ids):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "duplicate arm_id in arm_results")

    # Fixed shared budget: every arm must carry the identical budget.
    budget_key = stable_json_dumps(arms[0].budget.to_doc())
    for arm in arms[1:]:
        if stable_json_dumps(arm.budget.to_doc()) != budget_key:
            raise ScientificReportError(
                CODE_BUDGET_MISMATCH,
                f"arm {arm.arm_id!r} budget differs from {arms[0].arm_id!r}; "
                "comparison arms must run under one fixed shared budget (design §13.4)",
            )

    # Cohort-level discipline: demo24 scope + valid-use declaration.
    any_valid_use = False
    valid_used_at: str | None = None
    valid_input_set_sha256: str | None = None
    claims_population_estimate = False
    has_demo24 = False
    for arm in arms:
        for cohort in arm.cohorts:
            assert_demo24_scope(cohort.cohort_id, cohort.scope)
            if cohort.cohort_id == COHORT_DEMO24:
                has_demo24 = True
            if cohort.scope == SCOPE_POPULATION_ESTIMATE:
                claims_population_estimate = True
            if cohort.consumes_valid:
                any_valid_use = True
                if not cohort.valid_used_at or not cohort.valid_input_set_sha256:
                    raise ScientificReportError(
                        CODE_VALID_USE_UNDECLARED,
                        f"cohort {cohort.cohort_id!r} consumes valid-split rows and must "
                        "carry valid_used_at + valid_input_set_sha256 (design §13.4)",
                    )
                valid_used_at = cohort.valid_used_at
                valid_input_set_sha256 = cohort.valid_input_set_sha256

    # Population-estimate claims require the L4 lock (guard only, no execution).
    lock_doc: dict[str, JSONValue] | None = None
    full_population_unlocked = False
    if claims_population_estimate:
        lock = assert_population_unlocked(stage, lock=population_lock)
        lock_doc = lock.to_doc()
        full_population_unlocked = True
    elif population_lock is not None:
        lock_doc = _coerce_lock(population_lock).to_doc()

    # Input provenance: mapping_provenance must be explicit on every block.
    if input_provenance is None:
        provenance_blocks = [
            InputProvenance(
                source="g2_scan_ready_export_contracts_v1",
                mapping_provenance=MAPPING_PROVENANCE,
                note=(
                    "lineage mapping provenance declared explicitly from the "
                    "bundle/proposal docs (design §13.4)"
                ),
            )
        ]
    else:
        provenance_blocks = [_coerce_provenance(entry) for entry in input_provenance]
    if not provenance_blocks:
        raise ScientificReportError(
            CODE_PROVENANCE_MAPPING_UNDECLARED,
            "input_provenance must carry at least one block with mapping_provenance",
        )

    if report_id is None:
        report_id = f"scan-strategy-report-{sha256_bytes(stable_json_dumps(dict(config)).encode('utf-8'))[:12]}"
    report_id = _require_str(report_id, "report_id")

    cohort_ids = sorted({cohort.cohort_id for arm in arms for cohort in arm.cohorts})
    doc: dict[str, Any] = {
        "schema_name": OBJECT_SCIENTIFIC_REPORT,
        "schema_version": SCHEMA_SCIENTIFIC_REPORT,
        "object_id": report_id,
        "created_at": created_at
        if created_at is not None
        else datetime.now(UTC).isoformat(timespec="seconds"),
        "producer": {"name": "pes2ts", "version": "0.1"},
        "input_refs": [],
        "status": "assembled",
        "stage": stage,
        "shared_budget": arms[0].budget.to_doc(),
        "shared_budget_key": budget_key,
        "arms": [_project_arm(arm, budget_key) for arm in arms],
        "cohort_ids": cohort_ids,
        "input_provenance": [block.to_doc() for block in provenance_blocks],
        "valid_discipline": {
            "valid_used": any_valid_use,
            "valid_used_at": valid_used_at,
            "valid_input_set_sha256": valid_input_set_sha256,
            "n_train": N_TRAIN_DEMO,
            "n_valid": N_VALID_DEMO,
            "note": VALID_LOCK_NOTE,
        },
        "population": {
            "scan_ready_n": POPULATION_SCAN_READY_N,
            "stage": stage,
            "claims_population_estimate": claims_population_estimate,
            "full_population_unlocked": full_population_unlocked,
            "population_lock": lock_doc,
            "note": (
                "full population may only be consumed after the L4 version lock "
                "(裁决记录 §7.5); this report never executes the population"
            ),
        },
        "demo24": (
            {
                "cohort_id": COHORT_DEMO24,
                "scope": SCOPE_DEVELOPMENT_MECHANISM_COVERAGE,
                "n_total": N_DEMO_TOTAL,
                "coverage": DEMO24_COVERAGE.to_doc(),
                "note": DEMO24_SCOPE_NOTE,
            }
            if has_demo24
            else None
        ),
        "comparison_arms": [
            {"arm_id": arm_id, "arm_label": ARM_LABELS[arm_id]} for arm_id in COMPARISON_ARMS
        ],
        "cost_note": COST_NOTE_COUNTED_VS_MEASURED,
        "denominator_note": DENOMINATOR_NOTE,
        "config": dict(config),
        "config_sha256": sha256_bytes(stable_json_dumps(dict(config)).encode("utf-8")),
    }
    return seal_document(doc)


def dumps_scientific_report(document: Mapping[str, Any]) -> str:
    """Digest-check and serialize one assembled report to stable JSON."""
    doc = dict(document)
    if seal_document(doc)["content_sha256"] != doc.get("content_sha256"):
        raise ScientificReportError(CODE_REPORT_INPUT_INVALID, "$.content_sha256: digest mismatch")
    return stable_json_dumps(doc)


__all__ = [
    "ARM_COUPLED_1D",
    "ARM_GRAPH_SINGLE_B",
    "ARM_LABELS",
    "ARM_OLD_SINGLE_B",
    "ARM_PATH_NEB",
    "ARM_TIMING_DIRECTION",
    "CODE_ARM_UNKNOWN",
    "CODE_BUDGET_MISMATCH",
    "CODE_COHORT_INVALID",
    "CODE_DENOMINATOR_INVALID",
    "CODE_DEMO24_SCOPE_MISMATCH",
    "CODE_LOCK_INPUT_INVALID",
    "CODE_POPULATION_LOCKED_BEFORE_L4",
    "CODE_POPULATION_LOCK_REQUIRED",
    "CODE_PROVENANCE_MAPPING_UNDECLARED",
    "CODE_REPORT_INPUT_INVALID",
    "CODE_STAGE_UNKNOWN",
    "CODE_VALID_TUNED_AFTER_VALID_RUN",
    "CODE_VALID_USE_UNDECLARED",
    "COHORT_DEMO24",
    "COHORT_FULL_POPULATION",
    "COHORT_TRAIN16",
    "COHORT_VALID8",
    "COMPARISON_ARMS",
    "DENOMINATOR_NOTE",
    "DEMO24_SCOPE_NOTE",
    "N_DEMO_TOTAL",
    "N_TRAIN_DEMO",
    "N_VALID_DEMO",
    "OBJECT_SCIENTIFIC_REPORT",
    "OBJECT_VERSION_LOCK",
    "POPULATION_SCAN_READY_N",
    "SCOPE_DEVELOPMENT_MECHANISM_COVERAGE",
    "SCOPE_POPULATION_ESTIMATE",
    "SCHEMA_SCIENTIFIC_REPORT",
    "SCHEMA_VERSION_LOCK",
    "STAGE_L1",
    "STAGE_L2",
    "STAGE_L3",
    "STAGE_L4",
    "STAGES",
    "VALID_LOCK_NOTE",
    "ArmResult",
    "CohortResult",
    "InputProvenance",
    "RunLogEntry",
    "ScientificReportError",
    "SharedBudget",
    "VersionLock",
    "assert_demo24_scope",
    "assert_population_unlocked",
    "assert_valid_not_tuned",
    "build_scientific_report",
    "default_strategy_versions",
    "dumps_scientific_report",
    "lock_version",
]
