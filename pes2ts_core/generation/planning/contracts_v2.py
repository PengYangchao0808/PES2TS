"""Scan-strategy v2 document contracts: proposals, frozen plans, capabilities.

Typed document factories for the graph-theoretic PES scan-strategy selector
(design §10.1, `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`):

- ``g1_strategy_proposal_v1`` (``StrategyProposal``): finite, reasoned
  strategy candidates.  Proposals may be generated from ``needs_review``
  inputs; ``execution_eligible`` is **always** ``False`` at proposal stage.
  Per-candidate typed ``failure_reasons`` and whole-case ``blocking_reasons``
  are stored separately.
- ``g1_generation_plan_v2`` (``GenerationPlan``): frozen execution plan whose
  payload is a discriminated union ``ScanCandidateV2 | PathCandidateV1``
  (discriminator ``candidate_kind``).  ``ScanCandidateV2`` covers
  ``SINGLE_1D`` / ``COUPLED_1D`` / ``SCHEDULED_1D`` scan modes;
  ``PathCandidateV1`` covers path requests method-conditionally (ADR-0001/
  ADR-0002): ``NEB`` carries endpoint geometries + an ``image_chain``
  parameter block (ORCA ``%geom Path`` shape); ``XTB_PATH`` carries the same
  endpoints plus a ``path_recipe`` assembly/recipe parameter block and **no**
  image chain (execution is a planning-plane dispatch to the ACP XTB_PATH
  backend, never ORCA per-point hashes).
- ``orca_capabilities_v1`` (``BackendCapability``): backend capability
  registry record matching the ``orca_capabilities_v1.json`` shape.

Identity sealing reuses ``pes2ts_core.contracts.seal_document`` verbatim, so
the volatile-key exclusion semantics are identical to v1: ``created_at``,
``generated_at`` and ``producer`` are volatile (excluded from the digest);
``content_sha256`` is stable over the frozen content.  Validation always
rejects any document containing a key in ``contracts.FORBIDDEN_TRUTH_KEYS`` or
``g1.v2_verify.FORBIDDEN_EXPORT_KEYS`` (unioned, lower-cased, single source).

This module follows the ``pes2ts_core/contracts.py`` idiom (stdlib-only dict
factories + list-returning validators).  Legacy single-B ``ScanPlan`` exporter
stubs are deliberately absent; they arrive with the compatibility-export todo.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from typing import Any, Final, TypeGuard

from pes2ts_core.contracts import (
    FORBIDDEN_TRUTH_KEYS as _V1_TRUTH_KEYS,
)
from pes2ts_core.contracts import (
    ContractError,
    seal_document,
)
from pes2ts_core.g1.v2_verify import (
    FORBIDDEN_EXPORT_KEYS as _V2_EXPORT_KEYS,
)
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Schema names and versions (interface-commitment table, 谱系归一裁决 §3).
# ---------------------------------------------------------------------------
OBJECT_STRATEGY_PROPOSAL: Final[str] = "StrategyProposal"
OBJECT_GENERATION_PLAN: Final[str] = "GenerationPlan"
OBJECT_BACKEND_CAPABILITY: Final[str] = "BackendCapability"
SCHEMA_STRATEGY_PROPOSAL: Final[str] = "g1_strategy_proposal_v1"
SCHEMA_GENERATION_PLAN: Final[str] = "g1_generation_plan_v2"
SCHEMA_BACKEND_CAPABILITY: Final[str] = "orca_capabilities_v1"
OBJECTS: Final[frozenset[str]] = frozenset({
    OBJECT_STRATEGY_PROPOSAL, OBJECT_GENERATION_PLAN, OBJECT_BACKEND_CAPABILITY,
})
SCHEMA_VERSIONS: Final[dict[str, str]] = {
    OBJECT_STRATEGY_PROPOSAL: SCHEMA_STRATEGY_PROPOSAL,
    OBJECT_GENERATION_PLAN: SCHEMA_GENERATION_PLAN,
    OBJECT_BACKEND_CAPABILITY: SCHEMA_BACKEND_CAPABILITY,
}

#: Every key that must never appear anywhere in a v2 document.  Single source:
#: the v1 truth blacklist plus the v2 export blacklist (lower-cased).
FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {key.lower() for key in _V1_TRUTH_KEYS}
    | {key.lower() for key in _V2_EXPORT_KEYS}
)

BASE_REQUIRED: Final[frozenset[str]] = frozenset({
    "schema_name", "schema_version", "object_id", "created_at", "producer",
    "input_refs", "content_sha256", "status",
})

STATUSES: Final[dict[str, frozenset[str]]] = {
    OBJECT_STRATEGY_PROPOSAL: frozenset({"proposed", "needs_review", "rejected", "accepted"}),
    OBJECT_GENERATION_PLAN: frozenset({"frozen", "superseded"}),
    OBJECT_BACKEND_CAPABILITY: frozenset({"active", "retired"}),
}

SPLITS: Final[frozenset[str]] = frozenset({"train", "valid", "test", "unassigned"})

# ---------------------------------------------------------------------------
# Candidate discriminated-union vocabulary.
# ---------------------------------------------------------------------------
CANDIDATE_KIND_SCAN: Final[str] = "ScanCandidateV2"
CANDIDATE_KIND_PATH: Final[str] = "PathCandidateV1"
CANDIDATE_KINDS: Final[frozenset[str]] = frozenset({CANDIDATE_KIND_SCAN, CANDIDATE_KIND_PATH})

MODE_SINGLE_1D: Final[str] = "SINGLE_1D"
MODE_COUPLED_1D: Final[str] = "COUPLED_1D"
MODE_SCHEDULED_1D: Final[str] = "SCHEDULED_1D"
SCAN_MODES: Final[tuple[str, ...]] = (MODE_SINGLE_1D, MODE_COUPLED_1D, MODE_SCHEDULED_1D)

DRIVER_KINDS: Final[tuple[str, ...]] = ("B", "A", "D")
COVERAGE_KINDS: Final[tuple[str, ...]] = ("direct", "coupled_monitor", "uncovered")
DIRECTIONS: Final[tuple[str, ...]] = ("R_to_P", "P_to_R")
ENDPOINTS: Final[tuple[str, ...]] = ("R", "P")
SCHEDULE_KINDS: Final[tuple[str, ...]] = ("linear", "event_A_early", "event_A_late", "smoothstep")
CAPABILITY_CHECK_STATUSES: Final[frozenset[str]] = frozenset({"pass", "fail", "unknown"})
CAPABILITY_MODES: Final[tuple[str, ...]] = (*SCAN_MODES, "PATH_NEB")
#: Path method vocabulary (ADR-0001/ADR-0002).  ``NEB`` compiles to the ORCA
#: ``%geom Path`` shape; ``XTB_PATH`` is a planning-plane dispatch executed
#: through the ACP XTB_PATH backend as a recipe (never ORCA per-point hashes).
#: Adding a member never invalidates previously sealed documents: validation
#: is per-candidate and method-conditional, and old NEB plans are not re-sealed.
METHOD_NEB: Final[str] = "NEB"
METHOD_XTB_PATH: Final[str] = "XTB_PATH"
PATH_METHOD_KINDS: Final[tuple[str, ...]] = (METHOD_NEB, METHOD_XTB_PATH)
COMPILED_KINDS: Final[tuple[str, ...]] = ("hashes", "recipe")
EPISTEMIC_ENDPOINT_HYPOTHESIS: Final[str] = "endpoint_hypothesis"
#: Upper bound of native scan coordinates (config ``scan_strategy.max_scan_coordinates``).
MAX_SCAN_DRIVERS: Final[int] = 3

# Fields that exist on exactly one side of the plan-candidate union.
_SCAN_ONLY_KEYS: Final[frozenset[str]] = frozenset({
    "mode", "drivers", "lambda_values", "schedule_id", "schedule_kind", "assembly_id",
})
_PATH_ONLY_KEYS: Final[frozenset[str]] = frozenset({
    "endpoint_geometries", "image_chain", "method_kind", "path_recipe",
})

# ---------------------------------------------------------------------------
# Per-kind field dictionaries (contracts.py OBJECT_FIELDS/OPTIONAL_FIELDS idiom).
# ---------------------------------------------------------------------------
OBJECT_FIELDS: Final[dict[str, frozenset[str]]] = {
    OBJECT_STRATEGY_PROPOSAL: frozenset({
        "reaction_id", "case_id", "split", "source_case_sha256",
        "graph_input", "graph_features", "family", "motif_tags", "rule_trace",
        "epistemic_status", "candidates", "execution_eligible", "reasons",
        "blocking_reasons",
    }),
    OBJECT_GENERATION_PLAN: frozenset({
        "reaction_id", "case_id", "split", "plan_id", "plan_version",
        "source_proposal_sha256", "source_case_sha256", "policy_hashes",
        "backend", "candidate_graph", "candidates", "budget", "compiled",
        "quality_tests",
    }),
    OBJECT_BACKEND_CAPABILITY: frozenset({
        "engine_version", "adapter_version", "supported_modes", "coordinate_kinds",
        "max_scan_coordinates", "point_limits", "custom_schedule_support",
        "constraint_support", "method_element_coverage", "probe_receipts",
    }),
}
OPTIONAL_FIELDS: Final[dict[str, frozenset[str]]] = {
    OBJECT_STRATEGY_PROPOSAL: frozenset(),
    OBJECT_GENERATION_PLAN: frozenset({"supersedes"}),
    OBJECT_BACKEND_CAPABILITY: frozenset(),
}

#: Proposal-stage candidate (pre-freeze; scan modes only for now).
PROPOSAL_CANDIDATE_REQUIRED: Final[frozenset[str]] = frozenset({
    "candidate_id", "mode", "start_endpoint", "direction", "assembly_id",
    "anchor_reason", "drivers", "monitors", "guards", "event_coverage",
    "lambda_values", "schedule_id", "required_capabilities", "capability_check",
    "budget", "failure_reasons",
})
PROPOSAL_CANDIDATE_OPTIONAL: Final[frozenset[str]] = frozenset({
    "schedule_kind", "run_if", "pass_if", "fallback_ids", "expected_cost", "extensions",
})

#: Frozen plan scan payload.
SCAN_CANDIDATE_REQUIRED: Final[frozenset[str]] = frozenset({
    "candidate_kind", "candidate_id", "mode", "start_endpoint", "direction",
    "assembly_id", "anchor_reason", "drivers", "monitors", "guards",
    "event_coverage", "lambda_values", "schedule_id", "required_capabilities",
    "budget", "failure_reasons",
})
SCAN_CANDIDATE_OPTIONAL: Final[frozenset[str]] = frozenset({
    "schedule_kind", "run_if", "pass_if", "fallback_ids", "expected_cost", "extensions",
})

#: Frozen plan path payload.  Method-conditional (ADR-0001/ADR-0002):
#: ``image_chain`` is required only for ``NEB``; ``XTB_PATH`` carries a
#: ``path_recipe`` assembly/recipe parameter block instead (no image chain,
#: no ORCA per-point hashes — execution defers to the ACP XTB_PATH backend).
#: Absent ``method_kind`` means ``NEB``, so already-frozen NEB documents that
#: omit it keep validating unchanged.
PATH_CANDIDATE_REQUIRED: Final[frozenset[str]] = frozenset({
    "candidate_kind", "candidate_id", "n_atoms", "endpoint_geometries",
    "start_endpoint", "direction", "anchor_reason", "failure_reasons",
})
PATH_CANDIDATE_OPTIONAL: Final[frozenset[str]] = frozenset({
    "method_kind", "image_chain", "path_recipe", "required_capabilities",
    "budget", "run_if", "pass_if", "fallback_ids", "expected_cost", "extensions",
})

SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


def _is_number(value: Any) -> TypeGuard[int | float]:
    """True for JSON numbers (bool is not a number)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> TypeGuard[int]:
    """True for JSON integers (bool is not an integer)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_str(value: Any) -> TypeGuard[str]:
    """True for non-empty strings."""
    return isinstance(value, str) and bool(value)


def _finite_tree(value: Any, path: str = "$", issues: list[str] | None = None) -> list[str]:
    """Return every non-finite number issue in the JSON tree at *value*."""
    issues = [] if issues is None else issues
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(f"{path}: non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}", issues)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _finite_tree(child, f"{path}[{i}]", issues)
    return issues


def _forbidden_key_issues(value: Any, path: str = "$", issues: list[str] | None = None) -> list[str]:
    """Return every forbidden evaluation/truth key issue in the JSON tree."""
    issues = [] if issues is None else issues
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                issues.append(f"{path}.{key}: forbidden evaluation/truth field")
            _forbidden_key_issues(child, f"{path}.{key}", issues)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _forbidden_key_issues(child, f"{path}[{i}]", issues)
    return issues


def _check_sha256(value: Any, path: str, issues: list[str]) -> None:
    """Append an issue when *value* is not a lowercase SHA256 hex digest."""
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        issues.append(f"{path}: must be a lowercase SHA256")


def _check_allowed_keys(
    data: dict[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
    path: str,
    issues: list[str],
) -> None:
    """Append missing-required and unknown-field issues for one object."""
    for key in sorted(required - data.keys()):
        issues.append(f"{path}.{key}: required field missing")
    allowed = required | optional | {"extensions"}
    for key in sorted(data.keys() - allowed):
        issues.append(f"{path}.{key}: unknown field")


def _reason_list_issues(reasons: Any, path: str, issues: list[str]) -> None:
    """Validate a typed reason list (``{"code", "detail"?}`` objects)."""
    if reasons is None:
        return
    if not isinstance(reasons, list):
        issues.append(f"{path}: must be an array")
        return
    for i, reason in enumerate(reasons):
        rpath = f"{path}[{i}]"
        if not isinstance(reason, dict):
            issues.append(f"{rpath}: must be an object with a typed code")
            continue
        if not _is_nonempty_str(reason.get("code")):
            issues.append(f"{rpath}.code: must be a non-empty string")
        if "detail" in reason and reason["detail"] is not None and not isinstance(reason["detail"], str):
            issues.append(f"{rpath}.detail: must be a string when present")
        for key in sorted(reason.keys() - {"code", "detail", "extensions"}):
            issues.append(f"{rpath}.{key}: unknown field")


def _int_list_ok(value: Any) -> TypeGuard[list[int]]:
    """True for a non-empty list of positive integers."""
    return (
        isinstance(value, list) and bool(value)
        and all(_is_int(item) and item > 0 for item in value)
    )


def _endpoint_direction_issues(candidate: dict[str, Any], path: str, issues: list[str]) -> None:
    """Append issues when start_endpoint/direction are invalid or inconsistent."""
    start = candidate.get("start_endpoint")
    direction = candidate.get("direction")
    if start not in ENDPOINTS:
        issues.append(f"{path}.start_endpoint: must be 'R' or 'P'")
    if direction not in DIRECTIONS:
        issues.append(f"{path}.direction: must be 'R_to_P' or 'P_to_R'")
    if start in ENDPOINTS and direction in DIRECTIONS and (start == "R") != (direction == "R_to_P"):
        issues.append(f"{path}.direction: must start from start_endpoint ({start})")


def _budget_issues(budget: Any, path: str, issues: list[str]) -> None:
    """Validate a frozen per-candidate/per-plan budget object."""
    if not isinstance(budget, dict):
        issues.append(f"{path}: must be an object")
        return
    for key in ("max_attempts", "max_cpu_hours", "max_wall_seconds"):
        if key not in budget:
            issues.append(f"{path}.{key}: required field missing")
    if "max_attempts" in budget and (not _is_int(budget["max_attempts"]) or budget["max_attempts"] < 0):
        issues.append(f"{path}.max_attempts: must be a non-negative integer")
    if "max_cpu_hours" in budget and (not _is_number(budget["max_cpu_hours"]) or budget["max_cpu_hours"] < 0):
        issues.append(f"{path}.max_cpu_hours: must be a non-negative number")
    if "max_wall_seconds" in budget and (not _is_int(budget["max_wall_seconds"]) or budget["max_wall_seconds"] < 0):
        issues.append(f"{path}.max_wall_seconds: must be a non-negative integer")


def _string_list_issues(value: Any, path: str, issues: list[str]) -> None:
    """Append issues when *value* is not a list of non-empty strings."""
    if not isinstance(value, list):
        issues.append(f"{path}: must be an array")
        return
    for i, entry in enumerate(value):
        if not _is_nonempty_str(entry):
            issues.append(f"{path}[{i}]: must be a non-empty string")


def _driver_issues(driver: Any, path: str, issues: list[str]) -> None:
    """Validate one scan driver (B/A/D coordinate definition)."""
    if not isinstance(driver, dict):
        issues.append(f"{path}: must be an object")
        return
    if driver.get("kind") not in DRIVER_KINDS:
        issues.append(f"{path}.kind: must be one of {', '.join(DRIVER_KINDS)}")
    maps = driver.get("maps")
    if not _int_list_ok(maps):
        issues.append(f"{path}.maps: must be a non-empty array of positive integers")
    elif len(set(maps)) != len(maps):
        issues.append(f"{path}.maps: atom map ids must be unique")
    if not _is_nonempty_str(driver.get("unit")):
        issues.append(f"{path}.unit: must be a non-empty string")
    if driver.get("schedule_values") is not None and "schedule_values" in driver:
        values = driver["schedule_values"]
        if not isinstance(values, list) or any(not _is_number(v) or not math.isfinite(v) for v in values):
            issues.append(f"{path}.schedule_values: must be an array of finite numbers or null")
    if driver.get("index0") is not None and "index0" in driver:
        index0 = driver["index0"]
        if not isinstance(index0, list) or any(not _is_int(v) or v < 0 for v in index0):
            issues.append(f"{path}.index0: must be an array of non-negative integers or null")


def _monitor_issues(monitor: Any, path: str, issues: list[str]) -> None:
    """Validate one monitor (non-constraining measurement)."""
    if not isinstance(monitor, dict):
        issues.append(f"{path}: must be an object")
        return
    if not _is_nonempty_str(monitor.get("measurement")):
        issues.append(f"{path}.measurement: must be a non-empty string")
    if not _is_nonempty_str(monitor.get("target_test")):
        issues.append(f"{path}.target_test: must be a non-empty string")
    if not isinstance(monitor.get("evidence_available"), bool):
        issues.append(f"{path}.evidence_available: must be a boolean")
    if "maps" in monitor and monitor["maps"] is not None and not _int_list_ok(monitor["maps"]):
        issues.append(f"{path}.maps: must be a non-empty array of positive integers")
    if "region" in monitor and monitor["region"] is not None and not _is_nonempty_str(monitor["region"]):
        issues.append(f"{path}.region: must be a non-empty string when present")


def _guard_issues(guard: Any, path: str, issues: list[str]) -> None:
    """Validate one guard (constraint-like release rule)."""
    if not isinstance(guard, dict):
        issues.append(f"{path}: must be an object")
        return
    if not _is_nonempty_str(guard.get("kind")):
        issues.append(f"{path}.kind: must be a non-empty string")
    if not isinstance(guard.get("values"), list):
        issues.append(f"{path}.values: must be an array")
    if not _is_nonempty_str(guard.get("release_rule")):
        issues.append(f"{path}.release_rule: must be a non-empty string")
    if not _is_nonempty_str(guard.get("rationale")):
        issues.append(f"{path}.rationale: must be a non-empty string")


def _event_coverage_issues(entries: Any, path: str, issues: list[str]) -> None:
    """Validate the per-candidate event-coverage records."""
    if not isinstance(entries, list):
        issues.append(f"{path}: must be an array")
        return
    for i, entry in enumerate(entries):
        epath = f"{path}[{i}]"
        if not isinstance(entry, dict):
            issues.append(f"{epath}: must be an object")
            continue
        if not _is_nonempty_str(entry.get("event_id")):
            issues.append(f"{epath}.event_id: must be a non-empty string")
        if entry.get("coverage") not in COVERAGE_KINDS:
            issues.append(f"{epath}.coverage: must be one of {', '.join(COVERAGE_KINDS)}")


def _scan_body_issues(candidate: dict[str, Any], path: str, issues: list[str]) -> None:
    """Validate the shared scan-candidate body (proposal and frozen plan)."""
    if not _is_nonempty_str(candidate.get("candidate_id")):
        issues.append(f"{path}.candidate_id: must be a non-empty string")
    mode = candidate.get("mode")
    if mode not in SCAN_MODES:
        issues.append(f"{path}.mode: must be one of {', '.join(SCAN_MODES)}")
    _endpoint_direction_issues(candidate, path, issues)
    for field in ("assembly_id", "anchor_reason", "schedule_id"):
        if not _is_nonempty_str(candidate.get(field)):
            issues.append(f"{path}.{field}: must be a non-empty string")
    drivers = candidate.get("drivers")
    if not isinstance(drivers, list) or not drivers:
        issues.append(f"{path}.drivers: scan candidates require at least one driver")
    else:
        if mode == MODE_SINGLE_1D and len(drivers) != 1:
            issues.append(f"{path}.drivers: SINGLE_1D requires exactly one driver")
        elif mode in (MODE_COUPLED_1D, MODE_SCHEDULED_1D) and not (2 <= len(drivers) <= MAX_SCAN_DRIVERS):
            issues.append(f"{path}.drivers: {mode} requires 2..{MAX_SCAN_DRIVERS} drivers")
        for i, driver in enumerate(drivers):
            _driver_issues(driver, f"{path}.drivers[{i}]", issues)
    monitors = candidate.get("monitors")
    if not isinstance(monitors, list):
        issues.append(f"{path}.monitors: must be an array")
    else:
        for i, monitor in enumerate(monitors):
            _monitor_issues(monitor, f"{path}.monitors[{i}]", issues)
    guards = candidate.get("guards")
    if not isinstance(guards, list):
        issues.append(f"{path}.guards: must be an array")
    else:
        for i, guard in enumerate(guards):
            _guard_issues(guard, f"{path}.guards[{i}]", issues)
    _event_coverage_issues(candidate.get("event_coverage"), f"{path}.event_coverage", issues)
    lambdas = candidate.get("lambda_values")
    if not isinstance(lambdas, list) or not lambdas:
        issues.append(f"{path}.lambda_values: must be a non-empty array shared by every driver")
    elif any(not _is_number(v) or not math.isfinite(v) for v in lambdas):
        issues.append(f"{path}.lambda_values: entries must be finite numbers")
    schedule_kind = candidate.get("schedule_kind")
    if mode == MODE_SCHEDULED_1D and schedule_kind not in SCHEDULE_KINDS:
        issues.append(f"{path}.schedule_kind: required for SCHEDULED_1D")
    elif schedule_kind is not None and schedule_kind not in SCHEDULE_KINDS:
        issues.append(f"{path}.schedule_kind: invalid schedule kind")
    if "required_capabilities" in candidate and candidate["required_capabilities"] is not None:
        _string_list_issues(candidate["required_capabilities"], f"{path}.required_capabilities", issues)
    elif "required_capabilities" not in candidate:
        issues.append(f"{path}.required_capabilities: required field missing")
    _budget_issues(candidate.get("budget"), f"{path}.budget", issues)
    _reason_list_issues(candidate.get("failure_reasons"), f"{path}.failure_reasons", issues)
    if "run_if" in candidate and candidate["run_if"] is not None and not isinstance(candidate["run_if"], (str, bool)):
        issues.append(f"{path}.run_if: must be a string, boolean, or null")
    if "pass_if" in candidate and candidate["pass_if"] is not None and not isinstance(candidate["pass_if"], (str, bool)):
        issues.append(f"{path}.pass_if: must be a string, boolean, or null")
    if "fallback_ids" in candidate and candidate["fallback_ids"] is not None:
        _string_list_issues(candidate["fallback_ids"], f"{path}.fallback_ids", issues)


def _proposal_candidate_issues(candidate: Any, path: str, issues: list[str]) -> None:
    """Validate one proposal-stage candidate payload."""
    if not isinstance(candidate, dict):
        issues.append(f"{path}: must be an object")
        return
    _check_allowed_keys(candidate, PROPOSAL_CANDIDATE_REQUIRED, PROPOSAL_CANDIDATE_OPTIONAL, path, issues)
    _scan_body_issues(candidate, path, issues)
    capability_check = candidate.get("capability_check")
    if not isinstance(capability_check, dict):
        issues.append(f"{path}.capability_check: must be an object")
    else:
        if capability_check.get("status") not in CAPABILITY_CHECK_STATUSES:
            allowed = ", ".join(sorted(CAPABILITY_CHECK_STATUSES))
            issues.append(f"{path}.capability_check.status: must be one of {allowed}")
        if "missing" in capability_check and capability_check["missing"] is not None:
            _string_list_issues(capability_check["missing"], f"{path}.capability_check.missing", issues)


def _scan_candidate_issues(candidate: dict[str, Any], path: str, issues: list[str]) -> None:
    """Validate one frozen-plan ``ScanCandidateV2`` payload."""
    if candidate.get("candidate_kind") != CANDIDATE_KIND_SCAN:
        issues.append(f"{path}.candidate_kind: expected '{CANDIDATE_KIND_SCAN}'")
    _check_allowed_keys(candidate, SCAN_CANDIDATE_REQUIRED, SCAN_CANDIDATE_OPTIONAL, path, issues)
    _scan_body_issues(candidate, path, issues)


def _geometry_block_issues(block: Any, n_atoms: Any, path: str, issues: list[str]) -> None:
    """Validate one endpoint geometry block (N×3 finite coordinates)."""
    if not isinstance(block, list):
        issues.append(f"{path}: must be an array of 3D coordinates")
        return
    if _is_int(n_atoms) and len(block) != n_atoms:
        issues.append(f"{path}: must contain exactly n_atoms={n_atoms} coordinate rows")
    for i, row in enumerate(block):
        rpath = f"{path}[{i}]"
        if not isinstance(row, list) or len(row) != 3:
            issues.append(f"{rpath}: expected one 3D coordinate")
            continue
        if any(not _is_number(v) or not math.isfinite(v) for v in row):
            issues.append(f"{rpath}: coordinates must be finite numbers")


def _path_candidate_issues(candidate: dict[str, Any], path: str, issues: list[str]) -> None:
    """Validate one frozen-plan ``PathCandidateV1`` payload (method-conditional).

    ``NEB`` (or an absent ``method_kind``, for already-frozen documents)
    requires the ``image_chain`` parameter block; ``XTB_PATH`` requires a
    non-empty ``path_recipe`` assembly/recipe block and rejects ``image_chain``
    (ADR-0002: xTB PATH executes through ACP as a recipe, never ORCA
    ``%geom Path`` / per-point hashes).
    """
    if candidate.get("candidate_kind") != CANDIDATE_KIND_PATH:
        issues.append(f"{path}.candidate_kind: expected '{CANDIDATE_KIND_PATH}'")
    _check_allowed_keys(candidate, PATH_CANDIDATE_REQUIRED, PATH_CANDIDATE_OPTIONAL, path, issues)
    if not _is_nonempty_str(candidate.get("candidate_id")):
        issues.append(f"{path}.candidate_id: must be a non-empty string")
    if not _is_nonempty_str(candidate.get("anchor_reason")):
        issues.append(f"{path}.anchor_reason: must be a non-empty string")
    _endpoint_direction_issues(candidate, path, issues)
    n_atoms = candidate.get("n_atoms")
    if not _is_int(n_atoms) or n_atoms < 1:
        issues.append(f"{path}.n_atoms: must be a positive integer")
    geometries = candidate.get("endpoint_geometries")
    if not isinstance(geometries, dict):
        issues.append(f"{path}.endpoint_geometries: must be an object with reactant and product")
    else:
        for side in ("reactant", "product"):
            _geometry_block_issues(geometries.get(side), n_atoms, f"{path}.endpoint_geometries.{side}", issues)
    method_kind = candidate.get("method_kind")
    if method_kind is not None and method_kind not in PATH_METHOD_KINDS:
        issues.append(f"{path}.method_kind: must be one of {', '.join(PATH_METHOD_KINDS)}")
    resolved_method = method_kind if method_kind is not None else METHOD_NEB
    if resolved_method == METHOD_XTB_PATH and method_kind in PATH_METHOD_KINDS:
        if "image_chain" in candidate and candidate["image_chain"] is not None:
            issues.append(
                f"{path}.image_chain: NEB-only field on an XTB_PATH candidate "
                "(xTB PATH carries path_recipe, executed through ACP per ADR-0002)"
            )
        path_recipe = candidate.get("path_recipe")
        if not isinstance(path_recipe, dict) or not path_recipe:
            issues.append(
                f"{path}.path_recipe: XTB_PATH candidates require a non-empty "
                "recipe/assembly parameter block"
            )
    else:
        image_chain = candidate.get("image_chain")
        if not isinstance(image_chain, dict):
            issues.append(f"{path}.image_chain: must be an object (required for NEB)")
        else:
            n_images = image_chain.get("n_images")
            if not _is_int(n_images) or n_images < 1:
                issues.append(f"{path}.image_chain.n_images: must be a positive integer")
        if "path_recipe" in candidate and candidate["path_recipe"] is not None:
            issues.append(
                f"{path}.path_recipe: XTB_PATH-only field on a {resolved_method} candidate"
            )
    if "required_capabilities" in candidate and candidate["required_capabilities"] is not None:
        _string_list_issues(candidate["required_capabilities"], f"{path}.required_capabilities", issues)
    if candidate.get("budget") is not None and "budget" in candidate:
        _budget_issues(candidate["budget"], f"{path}.budget", issues)
    if "run_if" in candidate and candidate["run_if"] is not None and not isinstance(candidate["run_if"], (str, bool)):
        issues.append(f"{path}.run_if: must be a string, boolean, or null")
    if "pass_if" in candidate and candidate["pass_if"] is not None and not isinstance(candidate["pass_if"], (str, bool)):
        issues.append(f"{path}.pass_if: must be a string, boolean, or null")
    if "fallback_ids" in candidate and candidate["fallback_ids"] is not None:
        _string_list_issues(candidate["fallback_ids"], f"{path}.fallback_ids", issues)
    _reason_list_issues(candidate.get("failure_reasons"), f"{path}.failure_reasons", issues)


def _plan_candidate_issues(candidate: Any, path: str, issues: list[str]) -> None:
    """Validate one discriminated-union payload of a frozen plan."""
    if not isinstance(candidate, dict):
        issues.append(f"{path}: must be an object")
        return
    kind = candidate.get("candidate_kind")
    if kind == CANDIDATE_KIND_SCAN:
        for key in sorted(_PATH_ONLY_KEYS & candidate.keys()):
            issues.append(f"{path}.{key}: path-only field on a {CANDIDATE_KIND_SCAN} payload")
        _scan_candidate_issues(candidate, path, issues)
    elif kind == CANDIDATE_KIND_PATH:
        for key in sorted(_SCAN_ONLY_KEYS & candidate.keys()):
            issues.append(f"{path}.{key}: scan-only field on a {CANDIDATE_KIND_PATH} payload")
        _path_candidate_issues(candidate, path, issues)
    else:
        issues.append(f"{path}.candidate_kind: must be '{CANDIDATE_KIND_SCAN}' or '{CANDIDATE_KIND_PATH}'")


def _proposal_issues(document: dict[str, Any]) -> list[str]:
    """Return every structural issue in one StrategyProposal document."""
    issues: list[str] = []
    for field in ("reaction_id", "case_id", "family"):
        if not _is_nonempty_str(document.get(field)):
            issues.append(f"$.{field}: must be a non-empty string")
    if document.get("split") not in SPLITS:
        issues.append("$.split: invalid split")
    _check_sha256(document.get("source_case_sha256"), "$.source_case_sha256", issues)
    graph_input = document.get("graph_input")
    if not isinstance(graph_input, dict):
        issues.append("$.graph_input: must be an object")
    else:
        _check_sha256(graph_input.get("endpoint_graph_sha256"), "$.graph_input.endpoint_graph_sha256", issues)
        if not _is_nonempty_str(graph_input.get("normalization_version")):
            issues.append("$.graph_input.normalization_version: must be a non-empty string")
        if not isinstance(graph_input.get("mapping_equivalence"), dict):
            issues.append("$.graph_input.mapping_equivalence: must be an object")
    graph_features = document.get("graph_features")
    if not isinstance(graph_features, dict):
        issues.append("$.graph_features: must be an object")
    else:
        for key in ("edit_components", "context_support", "events", "typed_couplings"):
            if key not in graph_features:
                issues.append(f"$.graph_features.{key}: required field missing")
    for field in ("motif_tags", "rule_trace", "reasons", "blocking_reasons", "candidates"):
        if not isinstance(document.get(field), list):
            issues.append(f"$.{field}: must be an array")
    if document.get("epistemic_status") != EPISTEMIC_ENDPOINT_HYPOTHESIS:
        issues.append(f"$.epistemic_status: must be {EPISTEMIC_ENDPOINT_HYPOTHESIS}")
    if document.get("execution_eligible") is not False:
        issues.append("$.execution_eligible: proposals are never execution-eligible")
    motif_tags = document.get("motif_tags")
    if isinstance(motif_tags, list) and any(not _is_nonempty_str(tag) for tag in motif_tags):
        issues.append("$.motif_tags: entries must be non-empty strings")
    rule_trace = document.get("rule_trace")
    if isinstance(rule_trace, list):
        for i, entry in enumerate(rule_trace):
            if not isinstance(entry, dict) or not _is_nonempty_str(entry.get("rule_id")):
                issues.append(f"$.rule_trace[{i}]: entries must be objects with a non-empty rule_id")
    _reason_list_issues(document.get("reasons"), "$.reasons", issues)
    _reason_list_issues(document.get("blocking_reasons"), "$.blocking_reasons", issues)
    candidates = document.get("candidates")
    if isinstance(candidates, list):
        for i, candidate in enumerate(candidates):
            _proposal_candidate_issues(candidate, f"$.candidates[{i}]", issues)
    if document.get("status") == "needs_review" and not document.get("reasons"):
        issues.append("$.reasons: required when the proposal status is needs_review")
    if document.get("status") == "rejected" and not document.get("blocking_reasons"):
        issues.append("$.blocking_reasons: rejected proposal must explain refusal")
    if isinstance(candidates, list) and not candidates and not document.get("reasons") and not document.get("blocking_reasons"):
        issues.append("$.reasons: a proposal without candidates must retain reasons or blocking_reasons")
    return issues


def _candidate_graph_issues(graph: Any, path: str, issues: list[str]) -> None:
    """Validate the frozen candidate graph (immutable node IDs, terminal states)."""
    if not isinstance(graph, dict):
        issues.append(f"{path}: must be an object")
        return
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        issues.append(f"{path}.nodes: must be a non-empty array")
        return
    node_ids: set[str] = set()
    for i, node in enumerate(nodes):
        npath = f"{path}.nodes[{i}]"
        if not isinstance(node, dict):
            issues.append(f"{npath}: must be an object")
            continue
        node_id = node.get("node_id")
        if not _is_nonempty_str(node_id):
            issues.append(f"{npath}.node_id: must be a non-empty string")
        elif node_id in node_ids:
            issues.append(f"{npath}.node_id: duplicate node identity")
        else:
            node_ids.add(node_id)
        if not _is_nonempty_str(node.get("state")):
            issues.append(f"{npath}.state: must be a non-empty string")
        if not isinstance(node.get("terminal"), bool):
            issues.append(f"{npath}.terminal: must be a boolean")
    edges = graph.get("edges")
    if not isinstance(edges, list):
        issues.append(f"{path}.edges: must be an array")
        return
    for i, edge in enumerate(edges):
        epath = f"{path}.edges[{i}]"
        if not isinstance(edge, dict):
            issues.append(f"{epath}: must be an object")
            continue
        for end in ("from", "to"):
            ref = edge.get(end)
            if not _is_nonempty_str(ref):
                issues.append(f"{epath}.{end}: must be a non-empty node id")
            elif ref not in node_ids:
                issues.append(f"{epath}.{end}: references unknown node {ref!r}")


def _compiled_issues(compiled: Any, path: str, issues: list[str]) -> None:
    """Validate the frozen compile binding (point hashes or sealed recipe)."""
    if not isinstance(compiled, dict):
        issues.append(f"{path}: must be an object")
        return
    kind = compiled.get("kind")
    if kind == "hashes":
        entries = compiled.get("entries")
        if not isinstance(entries, list) or not entries:
            issues.append(f"{path}.entries: hash-sealed plans require at least one point input hash")
        else:
            for i, entry in enumerate(entries):
                epath = f"{path}.entries[{i}]"
                if not isinstance(entry, dict):
                    issues.append(f"{epath}: must be an object")
                    continue
                if not _is_int(entry.get("point_index")) or entry["point_index"] < 0:
                    issues.append(f"{epath}.point_index: must be a non-negative integer")
                _check_sha256(entry.get("input_sha256"), f"{epath}.input_sha256", issues)
    elif kind == "recipe":
        recipe = compiled.get("recipe")
        if not isinstance(recipe, dict) or not recipe:
            issues.append(f"{path}.recipe: recipe-sealed plans require a non-empty frozen recipe object")
    else:
        issues.append(f"{path}.kind: must be 'hashes' or 'recipe'")


def _quality_tests_issues(tests: Any, path: str, issues: list[str]) -> None:
    """Validate the frozen quality-test list."""
    if not isinstance(tests, list):
        issues.append(f"{path}: must be an array")
        return
    for i, test in enumerate(tests):
        tpath = f"{path}[{i}]"
        if not isinstance(test, dict) or not _is_nonempty_str(test.get("test_id")):
            issues.append(f"{tpath}: must be an object with a non-empty test_id")


def _plan_issues(document: dict[str, Any]) -> list[str]:
    """Return every structural issue in one GenerationPlan document."""
    issues: list[str] = []
    for field in ("reaction_id", "case_id", "plan_id"):
        if not _is_nonempty_str(document.get(field)):
            issues.append(f"$.{field}: must be a non-empty string")
    if document.get("split") not in SPLITS:
        issues.append("$.split: invalid split")
    plan_version = document.get("plan_version")
    if not _is_int(plan_version) or plan_version < 1:
        issues.append("$.plan_version: must be a positive integer")
    _check_sha256(document.get("source_proposal_sha256"), "$.source_proposal_sha256", issues)
    _check_sha256(document.get("source_case_sha256"), "$.source_case_sha256", issues)
    policy_hashes = document.get("policy_hashes")
    if not isinstance(policy_hashes, dict) or not policy_hashes:
        issues.append("$.policy_hashes: must be a non-empty object of frozen policy digests")
    elif any(not _is_nonempty_str(key) or not _is_nonempty_str(value) for key, value in policy_hashes.items()):
        issues.append("$.policy_hashes: keys and digests must be non-empty strings")
    backend = document.get("backend")
    if not isinstance(backend, dict):
        issues.append("$.backend: must be an object")
    else:
        for field in ("engine", "adapter_version", "method"):
            if not _is_nonempty_str(backend.get(field)):
                issues.append(f"$.backend.{field}: must be a non-empty string")
        for field in ("resources", "electronic_state"):
            if backend.get(field) is not None and not isinstance(backend[field], dict):
                issues.append(f"$.backend.{field}: must be an object when present")
    _candidate_graph_issues(document.get("candidate_graph"), "$.candidate_graph", issues)
    _budget_issues(document.get("budget"), "$.budget", issues)
    _compiled_issues(document.get("compiled"), "$.compiled", issues)
    _quality_tests_issues(document.get("quality_tests"), "$.quality_tests", issues)
    candidates = document.get("candidates")
    if not isinstance(candidates, list):
        issues.append("$.candidates: must be an array")
    elif not candidates:
        issues.append("$.candidates: a frozen plan requires at least one candidate")
    else:
        for i, candidate in enumerate(candidates):
            _plan_candidate_issues(candidate, f"$.candidates[{i}]", issues)
        compiled = document.get("compiled")
        has_xtb_path = any(
            isinstance(candidate, dict)
            and candidate.get("candidate_kind") == CANDIDATE_KIND_PATH
            and candidate.get("method_kind") == METHOD_XTB_PATH
            for candidate in candidates
        )
        if has_xtb_path and isinstance(compiled, dict) and compiled.get("kind") != "recipe":
            issues.append(
                "$.compiled: XTB_PATH path candidates require a recipe binding "
                "(compiled.kind='recipe'); ORCA per-point hashes never apply (ADR-0002)"
            )
    supersedes = document.get("supersedes")
    if supersedes is not None and not _is_nonempty_str(supersedes):
        issues.append("$.supersedes: must be a non-empty string when present")
    if document.get("status") == "superseded" and not supersedes:
        issues.append("$.supersedes: superseded plans must reference the replacing plan")
    return issues


def _capability_issues(document: dict[str, Any]) -> list[str]:
    """Return every structural issue in one BackendCapability document."""
    issues: list[str] = []
    for field in ("engine_version", "adapter_version"):
        if not _is_nonempty_str(document.get(field)):
            issues.append(f"$.{field}: must be a non-empty string")
    modes = document.get("supported_modes")
    if not isinstance(modes, list):
        issues.append("$.supported_modes: must be an array")
    else:
        for i, mode in enumerate(modes):
            if mode not in CAPABILITY_MODES:
                issues.append(f"$.supported_modes[{i}]: must be one of {', '.join(CAPABILITY_MODES)}")
    kinds = document.get("coordinate_kinds")
    if not isinstance(kinds, list):
        issues.append("$.coordinate_kinds: must be an array")
    else:
        for i, kind in enumerate(kinds):
            if kind not in DRIVER_KINDS:
                issues.append(f"$.coordinate_kinds[{i}]: must be one of {', '.join(DRIVER_KINDS)}")
    max_coordinates = document.get("max_scan_coordinates")
    if not _is_int(max_coordinates) or max_coordinates < 0:
        issues.append("$.max_scan_coordinates: must be a non-negative integer")
    limits = document.get("point_limits")
    if not isinstance(limits, dict):
        issues.append("$.point_limits: must be an object")
    else:
        baseline = limits.get("baseline")
        maximum = limits.get("max")
        for field, value in (("baseline", baseline), ("max", maximum)):
            if not _is_int(value) or value < 1:
                issues.append(f"$.point_limits.{field}: must be a positive integer")
        if _is_int(baseline) and _is_int(maximum) and baseline > maximum:
            issues.append("$.point_limits: baseline must not exceed max")
    if not isinstance(document.get("custom_schedule_support"), bool):
        issues.append("$.custom_schedule_support: must be a boolean")
    constraints = document.get("constraint_support")
    if not isinstance(constraints, dict):
        issues.append("$.constraint_support: must be an object")
    else:
        for field in ("native_scan", "per_point_constraints", "simul_scan"):
            if not isinstance(constraints.get(field), bool):
                issues.append(f"$.constraint_support.{field}: must be a boolean")
    coverage = document.get("method_element_coverage")
    if not isinstance(coverage, dict):
        issues.append("$.method_element_coverage: must be an object")
    else:
        for method, elements in coverage.items():
            if not _is_nonempty_str(method):
                issues.append("$.method_element_coverage: method keys must be non-empty strings")
            if not isinstance(elements, list) or any(not _is_nonempty_str(element) for element in elements):
                issues.append(f"$.method_element_coverage.{method}: must be an array of element symbols")
    receipts = document.get("probe_receipts")
    if not isinstance(receipts, list):
        issues.append("$.probe_receipts: must be an array")
    else:
        for i, receipt in enumerate(receipts):
            rpath = f"$.probe_receipts[{i}]"
            if not isinstance(receipt, dict):
                issues.append(f"{rpath}: must be an object")
                continue
            if not _is_nonempty_str(receipt.get("probe_id")):
                issues.append(f"{rpath}.probe_id: must be a non-empty string")
            if not _is_nonempty_str(receipt.get("status")):
                issues.append(f"{rpath}.status: must be a non-empty string")
            if receipt.get("receipt_sha256") is not None:
                _check_sha256(receipt["receipt_sha256"], f"{rpath}.receipt_sha256", issues)
    return issues


def validate_v2_document(document: Any) -> list[str]:
    """Return all structural issues in one scan-strategy v2 document."""
    if not isinstance(document, dict):
        return ["$: document root must be an object"]
    kind = document.get("schema_name")
    if kind not in OBJECTS:
        return ["$.schema_name: unknown scan-strategy contract object"]
    issues: list[str] = []
    for key in sorted(BASE_REQUIRED - document.keys()):
        issues.append(f"$.{key}: required field missing")
    for key in sorted((OBJECT_FIELDS[kind] - OPTIONAL_FIELDS[kind]) - document.keys()):
        issues.append(f"$.{key}: required field missing")
    # Optional fields belong to the accepted set: superseded plans carry
    # ``supersedes``, so excluding OPTIONAL_FIELDS made that key unvalidatable.
    allowed = BASE_REQUIRED | OBJECT_FIELDS[kind] | OPTIONAL_FIELDS[kind] | {"extensions"}
    for key in sorted(document.keys() - allowed):
        issues.append(f"$.{key}: unknown field; use a namespaced extension if needed")
    if document.get("schema_version") != SCHEMA_VERSIONS[kind]:
        issues.append("$.schema_version: unsupported version")
    if not _is_nonempty_str(document.get("object_id")):
        issues.append("$.object_id: must be a non-empty string")
    if document.get("status") not in STATUSES[kind]:
        issues.append(f"$.status: invalid {kind} status")
    if not isinstance(document.get("producer"), dict):
        issues.append("$.producer: must be an object")
    if not isinstance(document.get("input_refs"), list):
        issues.append("$.input_refs: must be an array")
    stamp = document.get("created_at")
    try:
        datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        issues.append("$.created_at: must be ISO-8601")
    if "content_sha256" in document:
        _check_sha256(document.get("content_sha256"), "$.content_sha256", issues)
    issues.extend(_finite_tree(document))
    issues.extend(_forbidden_key_issues(document))
    if kind == OBJECT_STRATEGY_PROPOSAL:
        issues.extend(_proposal_issues(document))
    elif kind == OBJECT_GENERATION_PLAN:
        issues.extend(_plan_issues(document))
    else:
        issues.extend(_capability_issues(document))
    return issues


def _make_document(schema_name: str, object_id: str, status: str, **fields: Any) -> dict[str, Any]:
    """Assemble, seal, and validate one v2 document (raises on any issue)."""
    doc: dict[str, Any] = {
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSIONS[schema_name],
        "object_id": object_id,
        "created_at": fields.pop("created_at", datetime.now(UTC).isoformat(timespec="seconds")),
        "producer": {"name": "pes2ts", "version": "0.1"},
        "input_refs": [],
        "status": status,
        **fields,
    }
    doc = seal_document(doc)
    problems = validate_v2_document(doc)
    if problems:
        raise ContractError("; ".join(problems))
    return doc


def make_strategy_proposal(object_id: str, status: str, **fields: Any) -> dict[str, Any]:
    """Build a sealed ``g1_strategy_proposal_v1`` document.

    ``execution_eligible`` is forced to ``False``; passing ``True`` raises
    :class:`ContractError`.  ``needs_review`` inputs are allowed (the demo
    cohort is entirely needs_review).
    """
    if fields.pop("execution_eligible", False) is not False:
        raise ContractError("$.execution_eligible: proposals are never execution-eligible")
    fields["execution_eligible"] = False
    return _make_document(OBJECT_STRATEGY_PROPOSAL, object_id, status, **fields)


def make_generation_plan(object_id: str, status: str, **fields: Any) -> dict[str, Any]:
    """Build a sealed ``g1_generation_plan_v2`` document (frozen execution plan)."""
    return _make_document(OBJECT_GENERATION_PLAN, object_id, status, **fields)


def make_backend_capability(object_id: str, status: str, **fields: Any) -> dict[str, Any]:
    """Build a sealed ``orca_capabilities_v1`` registry document."""
    return _make_document(OBJECT_BACKEND_CAPABILITY, object_id, status, **fields)


def dumps_v2_document(document: dict[str, Any]) -> str:
    """Validate, digest-check, and serialize one v2 document to stable JSON."""
    problems = validate_v2_document(document)
    if problems:
        raise ContractError("; ".join(problems))
    if seal_document(document)["content_sha256"] != document["content_sha256"]:
        raise ContractError("$.content_sha256: digest mismatch")
    return stable_json_dumps(document)


def loads_v2_document(payload: str) -> dict[str, Any]:
    """Parse, validate, and digest-check one v2 document from stable JSON."""
    try:
        doc = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContractError(str(exc)) from exc
    if not isinstance(doc, dict):
        raise ContractError("contract root must be an object")
    problems = validate_v2_document(doc)
    if problems:
        raise ContractError("; ".join(problems))
    if seal_document(doc)["content_sha256"] != doc["content_sha256"]:
        raise ContractError("$.content_sha256: digest mismatch")
    return doc


__all__ = [
    "BASE_REQUIRED",
    "CANDIDATE_KIND_PATH",
    "CANDIDATE_KIND_SCAN",
    "CANDIDATE_KINDS",
    "CAPABILITY_CHECK_STATUSES",
    "CAPABILITY_MODES",
    "COVERAGE_KINDS",
    "COMPILED_KINDS",
    "ContractError",
    "DIRECTIONS",
    "DRIVER_KINDS",
    "ENDPOINTS",
    "EPISTEMIC_ENDPOINT_HYPOTHESIS",
    "FORBIDDEN_KEYS",
    "MAX_SCAN_DRIVERS",
    "METHOD_NEB",
    "METHOD_XTB_PATH",
    "MODE_COUPLED_1D",
    "MODE_SCHEDULED_1D",
    "MODE_SINGLE_1D",
    "OBJECTS",
    "OBJECT_BACKEND_CAPABILITY",
    "OBJECT_FIELDS",
    "OBJECT_GENERATION_PLAN",
    "OBJECT_STRATEGY_PROPOSAL",
    "OPTIONAL_FIELDS",
    "PATH_CANDIDATE_OPTIONAL",
    "PATH_CANDIDATE_REQUIRED",
    "PATH_METHOD_KINDS",
    "PROPOSAL_CANDIDATE_OPTIONAL",
    "PROPOSAL_CANDIDATE_REQUIRED",
    "SCHEDULE_KINDS",
    "SCAN_CANDIDATE_OPTIONAL",
    "SCAN_CANDIDATE_REQUIRED",
    "SCAN_MODES",
    "SCHEMA_BACKEND_CAPABILITY",
    "SCHEMA_GENERATION_PLAN",
    "SCHEMA_STRATEGY_PROPOSAL",
    "SCHEMA_VERSIONS",
    "SPLITS",
    "STATUSES",
    "dumps_v2_document",
    "loads_v2_document",
    "make_backend_capability",
    "make_generation_plan",
    "make_strategy_proposal",
    "seal_document",
    "validate_v2_document",
]
