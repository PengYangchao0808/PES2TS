"""GenerationPlanV2 freeze, budget discipline, and the frozen failure tree.

Implements design §10.1/§10.2/§8.3
(``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``):

- :func:`freeze_generation_plan` projects a release-gate-passing
  ``StrategyProposal`` candidate plus its compile binding (a todo-19
  ``CompiledRequest``, a hashes/recipe mapping, or ``None`` for a sealed
  candidate recipe) into a sealed ``g1_generation_plan_v2`` document via
  ``contracts_v2.make_generation_plan``.  Immutable node IDs, explicit
  terminal states, frozen coordinates/monitors/tolerances/budget, and a
  dimension bundle (rule versions, complete graph hash, atom order, quality
  tests, fallback tree) live under ``extensions.freeze``; the plan
  ``content_sha256`` therefore covers every execution-behavior dimension —
  mutating any of them yields a different digest (version-bump semantics).
- The failure tree is a **closed vocabulary** (design §10.2 table plus
  ``GATE_REFUSED``) with a machine-checkable predicate registry over a
  replayable :class:`FailureState`.  Free-text failure reasons are rejected.
- :class:`BudgetLedger` accounts for all budget consumers (directions,
  assemblies, schedules, retries, endpoint preparations) and preserves the
  spent/max denominator in the ``BUDGET_EXHAUSTED`` failure record while
  obtained paths stay retained.
- :func:`verify_generation_plan` performs structural (contracts) plus
  semantic verification: frozen fields present, digests consistent,
  failure-tree codes inside the closed vocabulary, budget coherent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.contracts import seal_document
from pes2ts_core.scan_strategy.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    CANDIDATE_KIND_SCAN,
    OBJECT_GENERATION_PLAN,
    SCHEMA_GENERATION_PLAN,
    make_generation_plan,
    validate_v2_document,
)
from pes2ts_core.scan_strategy.path_request import (
    RECIPE_PATH_REQUEST_V1 as _RECIPE_PATH_REQUEST_V1,
)
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps

# allow: SIZE_OK — plan-named todo-23 single module (freeze + failure tree +
# budget ledger + semantic verify; precedent contracts_v2/registry.py).

# ---------------------------------------------------------------------------
# Frozen versions and vocabulary constants.
# ---------------------------------------------------------------------------
PLAN_FREEZE_VERSION: Final[str] = "g1_generation_plan_freeze_v1"
REGISTRY_VERSION: Final[str] = "strategy_registry_v1"
SCHEDULE_VERSION: Final[str] = "schedules_v1"
FAILURE_TREE_VERSION: Final[str] = "failure_tree_v1"
RECIPE_CANDIDATE_V1: Final[str] = "pes2ts_candidate_recipe_v1"

DEFAULT_ENGINE: Final[str] = "orca"
DEFAULT_METHOD: Final[str] = "B3LYP-D3"
DEFAULT_ADAPTER_VERSION: Final[str] = "acp-adapter-v0"

#: Machine-checkable quality-test identifiers frozen into every plan.
DEFAULT_QUALITY_TEST_IDS: Final[tuple[str, ...]] = (
    "endpoint_reached",
    "driver_residual_within_tolerance",
    "target_connectivity_achieved",
    "path_continuity",
    "all_drivers_tracked",
    "budget_accounted",
)

#: Budget categories counted against the frozen plan budget (design §8.3).
BUDGET_ACCOUNTING_CATEGORIES: Final[tuple[str, ...]] = (
    "directions",
    "assemblies",
    "schedules",
    "retries",
    "endpoint_preparations",
)

#: Closed failure-tree vocabulary — design §10.2 enumeration plus
#: ``GATE_REFUSED`` (release/freeze gate not passed).  Never free text.
FAILURE_TREE_CODE_ORDER: Final[tuple[str, ...]] = (
    "MAP_INVALID",
    "MAP_AMBIGUOUS",
    "REPRESENTATION_AMBIGUOUS",
    "ELECTRONIC_STATE_UNRESOLVED",
    "SPECIAL_ELECTRONIC_STATE_REQUIRED",
    "ASSEMBLY_REQUIRED",
    "ENDPOINT_GEOMETRY_CONFLICT",
    "ENDPOINT_UNSTABLE_AT_METHOD",
    "NO_VALID_DRIVER",
    "EVENT_UNCOVERED",
    "CONSTRAINT_INFEASIBLE",
    "COORDINATE_DEGENERATE",
    "BACKEND_CAPABILITY_MISSING",
    "SCF_FAILED",
    "OPT_FAILED",
    "CONSTRAINT_RESIDUAL",
    "WRONG_CONNECTIVITY",
    "NON_TARGET_REACTION",
    "STEREO_MISMATCH",
    "PATH_DISCONTINUITY",
    "HYSTERESIS",
    "INTERMEDIATE_CANDIDATE",
    "MONOTONIC_PROFILE",
    "PEAK_NOT_BRACKETED",
    "BUDGET_EXHAUSTED",
    "GATE_REFUSED",
)
FAILURE_TREE_CODES: Final[frozenset[str]] = frozenset(FAILURE_TREE_CODE_ORDER)


class PlanFreezeError(ValueError):
    """Typed freeze/verify refusal; ``code`` is a stable machine token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Budget discipline (design §8.3).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BudgetLedger:
    """Immutable cumulative budget accounting for one frozen plan.

    Hard channels: ``max_attempts`` (always armed), ``max_cpu_hours`` and
    ``max_wall_seconds`` (armed only when positive — a non-positive maximum
    records that the channel carries no independent cap, consistent with the
    ACP contract where CPU time is unmeasured).  Category counters record
    where budget went; they never replace the hard channels.  Exhaustion
    terminates further candidates while obtained paths are preserved; the
    failure record keeps the spent/max denominator.
    """

    max_attempts: int
    max_cpu_hours: float = 0.0
    max_wall_seconds: int = 0
    spent_attempts: int = 0
    spent_cpu_hours: float = 0.0
    spent_wall_seconds: int = 0
    spent_directions: int = 0
    spent_assemblies: int = 0
    spent_schedules: int = 0
    spent_retries: int = 0
    spent_endpoint_preparations: int = 0

    def exhausted(self) -> bool:
        """True when any armed channel has spent ≥ its maximum."""
        if self.max_attempts > 0 and self.spent_attempts >= self.max_attempts:
            return True
        if self.max_cpu_hours > 0 and self.spent_cpu_hours >= self.max_cpu_hours:
            return True
        if self.max_wall_seconds > 0 and self.spent_wall_seconds >= self.max_wall_seconds:
            return True
        return False

    def record(
        self,
        *,
        attempts: int = 0,
        cpu_hours: float = 0.0,
        wall_seconds: int = 0,
        directions: int = 0,
        assemblies: int = 0,
        schedules: int = 0,
        retries: int = 0,
        endpoint_preparations: int = 0,
    ) -> BudgetLedger:
        """Return a new ledger with the given increments added."""
        increments = {
            "attempts": attempts,
            "cpu_hours": cpu_hours,
            "wall_seconds": wall_seconds,
            "directions": directions,
            "assemblies": assemblies,
            "schedules": schedules,
            "retries": retries,
            "endpoint_preparations": endpoint_preparations,
        }
        for name, value in increments.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise PlanFreezeError(
                    "PLAN_BUDGET_RECORD_INVALID", f"{name} must be a non-negative number"
                )
        return BudgetLedger(
            max_attempts=self.max_attempts,
            max_cpu_hours=self.max_cpu_hours,
            max_wall_seconds=self.max_wall_seconds,
            spent_attempts=self.spent_attempts + int(attempts),
            spent_cpu_hours=self.spent_cpu_hours + float(cpu_hours),
            spent_wall_seconds=self.spent_wall_seconds + int(wall_seconds),
            spent_directions=self.spent_directions + int(directions),
            spent_assemblies=self.spent_assemblies + int(assemblies),
            spent_schedules=self.spent_schedules + int(schedules),
            spent_retries=self.spent_retries + int(retries),
            spent_endpoint_preparations=self.spent_endpoint_preparations
            + int(endpoint_preparations),
        )

    def spent_summary(self) -> dict[str, Any]:
        """Full spent side, including every accounting category."""
        return {
            "attempts": self.spent_attempts,
            "cpu_hours": self.spent_cpu_hours,
            "wall_seconds": self.spent_wall_seconds,
            "directions": self.spent_directions,
            "assemblies": self.spent_assemblies,
            "schedules": self.spent_schedules,
            "retries": self.spent_retries,
            "endpoint_preparations": self.spent_endpoint_preparations,
        }

    def max_summary(self) -> dict[str, Any]:
        """Denominator side of the budget."""
        return {
            "max_attempts": self.max_attempts,
            "max_cpu_hours": self.max_cpu_hours,
            "max_wall_seconds": self.max_wall_seconds,
        }

    def exhaustion_record(
        self, obtained_paths: Sequence[Any] = ()
    ) -> dict[str, Any]:
        """BUDGET_EXHAUSTED failure record preserving spent/max and paths."""
        return {
            "code": "BUDGET_EXHAUSTED",
            "budget_spent": self.spent_summary(),
            "budget_max": self.max_summary(),
            "obtained_paths": [dict(path) if isinstance(path, Mapping) else path for path in obtained_paths],
        }


@dataclass(frozen=True, slots=True)
class FailureState:
    """Replayable inputs for the frozen failure-tree predicate registry.

    One boolean per design §10.2 branch (plus ``gate_pass``); the budget
    channel is carried by a :class:`BudgetLedger`.
    """

    mapping_invalid: bool = False
    mapping_ambiguous: bool = False
    representation_ambiguous: bool = False
    electronic_state_unresolved: bool = False
    special_electronic_state_required: bool = False
    assembly_required: bool = False
    endpoint_geometry_conflict: bool = False
    endpoint_unstable_at_method: bool = False
    no_valid_driver: bool = False
    event_uncovered: bool = False
    constraint_infeasible: bool = False
    coordinate_degenerate: bool = False
    backend_capability_missing: bool = False
    scf_failed: bool = False
    opt_failed: bool = False
    constraint_residual: bool = False
    wrong_connectivity: bool = False
    non_target_reaction: bool = False
    stereo_mismatch: bool = False
    path_discontinuity: bool = False
    hysteresis: bool = False
    intermediate_candidate: bool = False
    monotonic_profile: bool = False
    peak_not_bracketed: bool = False
    gate_pass: bool = True
    budget: BudgetLedger | None = None


#: Closed vocabulary code → computable predicate over :class:`FailureState`.
FAILURE_PREDICATES: Final[dict[str, Callable[[FailureState], bool]]] = {
    "MAP_INVALID": lambda state: state.mapping_invalid,
    "MAP_AMBIGUOUS": lambda state: state.mapping_ambiguous,
    "REPRESENTATION_AMBIGUOUS": lambda state: state.representation_ambiguous,
    "ELECTRONIC_STATE_UNRESOLVED": lambda state: state.electronic_state_unresolved,
    "SPECIAL_ELECTRONIC_STATE_REQUIRED": lambda state: state.special_electronic_state_required,
    "ASSEMBLY_REQUIRED": lambda state: state.assembly_required,
    "ENDPOINT_GEOMETRY_CONFLICT": lambda state: state.endpoint_geometry_conflict,
    "ENDPOINT_UNSTABLE_AT_METHOD": lambda state: state.endpoint_unstable_at_method,
    "NO_VALID_DRIVER": lambda state: state.no_valid_driver,
    "EVENT_UNCOVERED": lambda state: state.event_uncovered,
    "CONSTRAINT_INFEASIBLE": lambda state: state.constraint_infeasible,
    "COORDINATE_DEGENERATE": lambda state: state.coordinate_degenerate,
    "BACKEND_CAPABILITY_MISSING": lambda state: state.backend_capability_missing,
    "SCF_FAILED": lambda state: state.scf_failed,
    "OPT_FAILED": lambda state: state.opt_failed,
    "CONSTRAINT_RESIDUAL": lambda state: state.constraint_residual,
    "WRONG_CONNECTIVITY": lambda state: state.wrong_connectivity,
    "NON_TARGET_REACTION": lambda state: state.non_target_reaction,
    "STEREO_MISMATCH": lambda state: state.stereo_mismatch,
    "PATH_DISCONTINUITY": lambda state: state.path_discontinuity,
    "HYSTERESIS": lambda state: state.hysteresis,
    "INTERMEDIATE_CANDIDATE": lambda state: state.intermediate_candidate,
    "MONOTONIC_PROFILE": lambda state: state.monotonic_profile,
    "PEAK_NOT_BRACKETED": lambda state: state.peak_not_bracketed,
    "BUDGET_EXHAUSTED": lambda state: state.budget is not None and state.budget.exhausted(),
    "GATE_REFUSED": lambda state: not state.gate_pass,
}


def is_failure_code(code: str) -> bool:
    """True when *code* is inside the closed failure-tree vocabulary."""
    return code in FAILURE_TREE_CODES


def assert_failure_code(code: str) -> None:
    """Raise :class:`PlanFreezeError` when *code* is outside the vocabulary."""
    if not is_failure_code(code):
        raise PlanFreezeError("FAILURE_CODE_UNKNOWN", str(code))


def active_failure_codes(state: FailureState) -> tuple[str, ...]:
    """Return every vocabulary code whose predicate holds for *state*."""
    return tuple(
        code for code in FAILURE_TREE_CODE_ORDER if FAILURE_PREDICATES[code](state)
    )


# ---------------------------------------------------------------------------
# Freeze-time projection helpers.
# ---------------------------------------------------------------------------
def compiled_binding_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the candidate's per-point generation recipe as a compiled binding.

    Used when no backend compile has been run: the frozen drivers, schedule
    and lambda grid fully determine per-point target values given endpoint
    geometries (design §10.1 "sealed per-point input-generation recipe").
    """
    drivers: list[dict[str, Any]] = []
    raw_drivers = candidate.get("drivers")
    if isinstance(raw_drivers, (list, tuple)):
        for driver in raw_drivers:
            if not isinstance(driver, Mapping):
                continue
            entry: dict[str, Any] = {
                "kind": driver.get("kind"),
                "maps": list(driver.get("maps") or ()),
                "unit": driver.get("unit"),
            }
            if driver.get("schedule_values") is not None:
                entry["schedule_values"] = list(driver["schedule_values"])
            drivers.append(entry)
    recipe = {
        "recipe_kind": RECIPE_CANDIDATE_V1,
        "mode": candidate.get("mode"),
        "schedule_id": candidate.get("schedule_id"),
        "schedule_kind": candidate.get("schedule_kind") or "linear",
        "lambda_values": list(candidate.get("lambda_values") or ()),
        "drivers": drivers,
        "generation": (
            "q_j(lambda_i)=q_j(start)+s_j(lambda_i)*(q_j(end)-q_j(start)); "
            "s_j from schedule_values when present else lambda; "
            "per-point inputs assembled at execution from endpoint geometries"
        ),
    }
    return {"kind": "recipe", "recipe": recipe}


def _compiled_binding(
    compiled: Any, candidate: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Return ``(contracts_block, compiled_doc, atom_order)`` for one binding."""
    if compiled is None:
        block = compiled_binding_from_candidate(candidate)
        doc = {
            "kind": "recipe",
            "compiled_kind": "recipe",
            "recipe_kind": RECIPE_CANDIDATE_V1,
            "recipe": block["recipe"],
            "candidate_id": candidate.get("candidate_id"),
        }
        return block, doc, None
    if isinstance(compiled, Mapping):
        kind = compiled.get("kind") or compiled.get("compiled_kind")
        if kind == "hashes":
            entries = compiled.get("entries")
            if not isinstance(entries, list) or not entries:
                point_hashes = compiled.get("point_input_sha256")
                if not isinstance(point_hashes, list) or not point_hashes:
                    raise PlanFreezeError(
                        "COMPILED_KIND_INVALID",
                        "hashes binding needs entries or point_input_sha256",
                    )
                entries = [
                    {"point_index": index, "input_sha256": digest}
                    for index, digest in enumerate(point_hashes)
                ]
            block = {"kind": "hashes", "entries": list(entries)}
        elif kind == "recipe":
            recipe = compiled.get("recipe")
            if not isinstance(recipe, Mapping) or not recipe:
                raise PlanFreezeError(
                    "COMPILED_RECIPE_INVALID",
                    "recipe binding needs a non-empty recipe object",
                )
            block = {"kind": "recipe", "recipe": dict(recipe)}
        else:
            raise PlanFreezeError("COMPILED_KIND_INVALID", str(kind))
        return block, dict(compiled), compiled.get("atom_rows")
    doc = compiled.to_doc()
    kind = doc.get("compiled_kind")
    if kind == "hashes":
        point_hashes = doc.get("point_input_sha256") or []
        if not point_hashes:
            raise PlanFreezeError(
                "COMPILED_KIND_INVALID", "hashes binding needs point input hashes"
            )
        block = {
            "kind": "hashes",
            "entries": [
                {"point_index": index, "input_sha256": digest}
                for index, digest in enumerate(point_hashes)
            ],
        }
    elif kind == "recipe":
        recipe = doc.get("recipe")
        if not isinstance(recipe, Mapping) or not recipe:
            raise PlanFreezeError(
                "COMPILED_RECIPE_INVALID",
                "recipe binding needs a non-empty recipe object",
            )
        block = {"kind": "recipe", "recipe": dict(recipe)}
    else:
        raise PlanFreezeError("COMPILED_KIND_INVALID", str(kind))
    return block, dict(doc), doc.get("atom_rows")


def _scan_candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Project one proposal candidate into a frozen ``ScanCandidateV2``."""
    payload: dict[str, Any] = {
        "candidate_kind": CANDIDATE_KIND_SCAN,
        "candidate_id": candidate["candidate_id"],
        "mode": candidate["mode"],
        "start_endpoint": candidate["start_endpoint"],
        "direction": candidate["direction"],
        "assembly_id": candidate["assembly_id"],
        "anchor_reason": candidate["anchor_reason"],
        "drivers": [dict(driver) for driver in candidate["drivers"]],
        "monitors": [dict(monitor) for monitor in candidate["monitors"]],
        "guards": [dict(guard) for guard in candidate["guards"]],
        "event_coverage": [dict(row) for row in candidate["event_coverage"]],
        "lambda_values": list(candidate["lambda_values"]),
        "schedule_id": candidate["schedule_id"],
        "required_capabilities": list(candidate["required_capabilities"]),
        "budget": dict(candidate["budget"]),
        "failure_reasons": [],
    }
    for optional_key in ("schedule_kind", "run_if", "pass_if", "expected_cost"):
        value = candidate.get(optional_key)
        if value is not None:
            payload[optional_key] = value
    fallback_ids = candidate.get("fallback_ids")
    if fallback_ids is not None:
        payload["fallback_ids"] = list(fallback_ids)
    extensions = candidate.get("extensions")
    if isinstance(extensions, Mapping):
        payload["extensions"] = dict(extensions)
    capability_check = candidate.get("capability_check")
    if isinstance(capability_check, Mapping):
        merged = dict(payload.get("extensions") or {})
        merged["capability_check"] = dict(capability_check)
        payload["extensions"] = merged
    return payload


def _nonnegative_number(raw: Any, default: float) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)) and raw >= 0:
        return float(raw)
    if isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return default
        return value if value >= 0 else default
    return default


def _stricter_positive(values: Sequence[float], default: float) -> float:
    positives = [value for value in values if value > 0]
    return min(positives) if positives else default


def _plan_budget(
    config: Mapping[str, Any] | None, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Freeze the stricter-of case/candidate budget (design §8.3)."""
    section: Mapping[str, Any] = {}
    if config is not None:
        raw = config.get("scan_strategy")
        if isinstance(raw, Mapping):
            freeze_section = raw.get("freeze")
            if isinstance(freeze_section, Mapping):
                section = freeze_section
    candidate_budgets = [
        candidate.get("budget") if isinstance(candidate.get("budget"), Mapping) else {}
        for candidate in candidates
    ]
    attempt_values = [
        float(budget["max_attempts"])
        for budget in candidate_budgets
        if isinstance(budget.get("max_attempts"), int)
        and not isinstance(budget.get("max_attempts"), bool)
        and budget["max_attempts"] > 0
    ]
    max_attempts = int(
        _stricter_positive(
            [
                *attempt_values,
                *(
                    [float(section["max_attempts"])]
                    if isinstance(section.get("max_attempts"), int)
                    and not isinstance(section.get("max_attempts"), bool)
                    and section["max_attempts"] > 0
                    else []
                ),
            ],
            1.0,
        )
    )
    if max_attempts < 1:
        raise PlanFreezeError("BUDGET_INVALID", "frozen plan max_attempts must be >= 1")
    max_cpu_hours = _stricter_positive(
        [
            *[
                _nonnegative_number(budget.get("max_cpu_hours"), 0.0)
                for budget in candidate_budgets
            ],
            _nonnegative_number(section.get("max_cpu_hours"), 0.0),
        ],
        0.0,
    )
    max_wall_seconds = int(
        _stricter_positive(
            [
                *[
                    float(budget["max_wall_seconds"])
                    for budget in candidate_budgets
                    if isinstance(budget.get("max_wall_seconds"), int)
                    and not isinstance(budget.get("max_wall_seconds"), bool)
                ],
                *(
                    [float(section["max_wall_seconds"])]
                    if isinstance(section.get("max_wall_seconds"), int)
                    and not isinstance(section.get("max_wall_seconds"), bool)
                    else []
                ),
            ],
            0.0,
        )
    )
    return {
        "max_attempts": max_attempts,
        "max_cpu_hours": max_cpu_hours,
        "max_wall_seconds": max_wall_seconds,
        "accounting": list(BUDGET_ACCOUNTING_CATEGORIES),
    }


def _candidate_graph(
    plan_id: str, candidate_id: str
) -> tuple[dict[str, Any], str]:
    """Build the frozen candidate graph with immutable IDs and terminals."""
    nodes = [
        {"node_id": f"{plan_id}:n-start", "state": "frozen_ready", "terminal": False},
        {
            "node_id": f"{plan_id}:n-exec",
            "state": f"execute:{candidate_id}",
            "terminal": False,
        },
        {
            "node_id": f"{plan_id}:n-fallback",
            "state": "next_fallback_candidate",
            "terminal": False,
        },
        {
            "node_id": f"{plan_id}:n-ok",
            "state": "path_saved_target_reached",
            "terminal": True,
        },
        {
            "node_id": f"{plan_id}:n-budget",
            "state": "budget_exhausted_preserve_paths",
            "terminal": True,
        },
        {
            "node_id": f"{plan_id}:n-fail",
            "state": "terminal_failure_preserved",
            "terminal": True,
        },
    ]
    start = nodes[0]["node_id"]
    execute = nodes[1]["node_id"]
    fallback = nodes[2]["node_id"]
    ok = nodes[3]["node_id"]
    budget = nodes[4]["node_id"]
    failure = nodes[5]["node_id"]
    edges = [
        {"from": start, "to": execute},
        {"from": execute, "to": ok},
        {"from": execute, "to": fallback},
        {"from": fallback, "to": execute},
        {"from": execute, "to": budget},
        {"from": execute, "to": failure},
    ]
    graph_sha = sha256_bytes(
        stable_json_dumps({"nodes": nodes, "edges": edges}).encode("utf-8")
    )
    graph = {"nodes": nodes, "edges": edges, "graph_sha256": graph_sha}
    return graph, graph_sha


def _backend_block(compiled_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Project the frozen backend identity from the compiled binding."""
    recipe: Mapping[str, Any] = {}
    raw_recipe = compiled_doc.get("recipe")
    if isinstance(raw_recipe, Mapping):
        recipe = raw_recipe
    engine = recipe.get("engine") or DEFAULT_ENGINE
    method = recipe.get("method") or DEFAULT_METHOD
    adapter_version = recipe.get("adapter_version") or DEFAULT_ADAPTER_VERSION
    backend: dict[str, Any] = {
        "engine": str(engine),
        "adapter_version": str(adapter_version),
        "method": str(method),
    }
    if recipe.get("nprocs") is not None:
        backend["resources"] = {"nprocs": recipe["nprocs"]}
    electronic: dict[str, Any] = {}
    if recipe.get("charge") is not None:
        electronic["charge"] = recipe["charge"]
    if recipe.get("multiplicity") is not None:
        electronic["multiplicity"] = recipe["multiplicity"]
    if electronic:
        backend["electronic_state"] = electronic
    return backend


def _quality_tests(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    ids = DEFAULT_QUALITY_TEST_IDS
    if config is not None:
        raw = config.get("scan_strategy")
        if isinstance(raw, Mapping):
            freeze_section = raw.get("freeze")
            if isinstance(freeze_section, Mapping):
                override = freeze_section.get("quality_tests")
                if isinstance(override, (list, tuple)) and override:
                    ids = tuple(str(entry) for entry in override)
    return [{"test_id": test_id} for test_id in ids]


def _atom_order_issues(atom_order: Any, path: str) -> list[str]:
    if atom_order is None:
        return []
    if not isinstance(atom_order, (list, tuple)) or not atom_order:
        return [f"{path}: must be a non-empty array of positive integers"]
    issues: list[str] = []
    seen: set[int] = set()
    for index, entry in enumerate(atom_order):
        if isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0:
            issues.append(f"{path}[{index}]: must be a positive integer")
        elif entry in seen:
            issues.append(f"{path}[{index}]: duplicate map id")
        else:
            seen.add(entry)
    return issues


def _freeze_block_issues(
    freeze_block: Mapping[str, Any], document: Mapping[str, Any]
) -> list[str]:
    """Semantic checks over the frozen dimension bundle."""
    issues: list[str] = []
    for field in ("freeze_version", "registry_version", "schedule_version"):
        value = freeze_block.get(field)
        if not isinstance(value, str) or not value:
            issues.append(f"$.extensions.freeze.{field}: must be a non-empty string")
    policy_hashes = document.get("policy_hashes")
    if isinstance(policy_hashes, Mapping):
        for field in ("registry_version", "schedule_version"):
            if (
                isinstance(freeze_block.get(field), str)
                and field in policy_hashes
                and policy_hashes[field] != freeze_block[field]
            ):
                issues.append(
                    f"$.policy_hashes.{field}: must match extensions.freeze.{field}"
                )
        endpoint = policy_hashes.get("endpoint_graph_sha256")
        if isinstance(endpoint, str) and freeze_block.get("endpoint_graph_sha256") not in (
            None,
            endpoint,
        ):
            issues.append(
                "$.policy_hashes.endpoint_graph_sha256: must match extensions.freeze"
            )
    for field in ("endpoint_graph_sha256", "candidate_graph_sha256", "graph_sha256"):
        value = freeze_block.get(field)
        if value is not None and (
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            issues.append(f"$.extensions.freeze.{field}: must be a lowercase SHA256")
    issues.extend(_atom_order_issues(freeze_block.get("atom_order"), "$.extensions.freeze.atom_order"))
    failure_tree = freeze_block.get("failure_tree")
    if not isinstance(failure_tree, Mapping):
        issues.append("$.extensions.freeze.failure_tree: must be an object")
    else:
        version = failure_tree.get("version")
        if not isinstance(version, str) or not version:
            issues.append("$.extensions.freeze.failure_tree.version: must be a non-empty string")
        codes = failure_tree.get("codes")
        if not isinstance(codes, list) or not codes:
            issues.append("$.extensions.freeze.failure_tree.codes: must be a non-empty array")
        else:
            for index, code in enumerate(codes):
                if not is_failure_code(str(code)):
                    issues.append(
                        f"$.extensions.freeze.failure_tree.codes[{index}]: "
                        f"unknown failure-tree code {code!r}"
                    )
    compiled_request = freeze_block.get("compiled_request")
    candidates = document.get("candidates")
    if compiled_request is not None:
        if not isinstance(compiled_request, Mapping):
            issues.append("$.extensions.freeze.compiled_request: must be an object")
        else:
            kind = compiled_request.get("compiled_kind") or compiled_request.get("kind")
            compiled = document.get("compiled")
            if isinstance(compiled, Mapping) and compiled.get("kind") != kind:
                issues.append(
                    "$.compiled.kind: must match extensions.freeze.compiled_request"
                )
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                request_id = compiled_request.get("candidate_id")
                if (
                    isinstance(request_id, str)
                    and isinstance(first, Mapping)
                    and first.get("candidate_id") not in (None, request_id)
                ):
                    issues.append(
                        "$.extensions.freeze.compiled_request.candidate_id: "
                        "must match the selected candidate"
                    )
            if kind == "hashes":
                entries = (
                    compiled.get("entries")
                    if isinstance(compiled, Mapping)
                    else None
                )
                point_hashes = compiled_request.get("point_input_sha256")
                if isinstance(entries, list) and isinstance(point_hashes, list):
                    expected = [
                        {"point_index": index, "input_sha256": digest}
                        for index, digest in enumerate(point_hashes)
                    ]
                    if entries != expected:
                        issues.append(
                            "$.compiled.entries: must project "
                            "compiled_request.point_input_sha256"
                        )
    budget_accounting = freeze_block.get("budget_accounting")
    if budget_accounting is not None:
        if not isinstance(budget_accounting, list):
            issues.append("$.extensions.freeze.budget_accounting: must be an array")
        else:
            allowed = set(BUDGET_ACCOUNTING_CATEGORIES)
            for index, entry in enumerate(budget_accounting):
                if entry not in allowed:
                    issues.append(
                        f"$.extensions.freeze.budget_accounting[{index}]: "
                        f"unknown budget category {entry!r}"
                    )
    return issues


def _failure_vocabulary_issues(document: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    candidates = document.get("candidates")
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            reasons = candidate.get("failure_reasons")
            if not isinstance(reasons, list):
                continue
            if reasons:
                issues.append(
                    f"$.candidates[{index}].failure_reasons: frozen candidates "
                    "must carry clean failure_reasons"
                )
            for reason_index, reason in enumerate(reasons):
                if not isinstance(reason, Mapping):
                    continue
                code = reason.get("code")
                if isinstance(code, str) and not is_failure_code(code):
                    issues.append(
                        f"$.candidates[{index}].failure_reasons[{reason_index}].code: "
                        f"unknown failure-tree code {code!r}"
                    )
    return issues


def _budget_coherence_issues(document: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    budget = document.get("budget")
    if isinstance(budget, Mapping):
        max_attempts = budget.get("max_attempts")
        if isinstance(max_attempts, int) and not isinstance(max_attempts, bool) and max_attempts < 1:
            issues.append("$.budget.max_attempts: frozen plans must authorize >= 1 attempt")
        accounting = budget.get("accounting")
        if accounting is not None:
            if not isinstance(accounting, list):
                issues.append("$.budget.accounting: must be an array")
            else:
                allowed = set(BUDGET_ACCOUNTING_CATEGORIES)
                for index, entry in enumerate(accounting):
                    if entry not in allowed:
                        issues.append(
                            f"$.budget.accounting[{index}]: unknown budget category {entry!r}"
                        )
        candidates = document.get("candidates")
        if isinstance(candidates, list) and isinstance(max_attempts, int):
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    continue
                candidate_budget = candidate.get("budget")
                if not isinstance(candidate_budget, Mapping):
                    continue
                candidate_attempts = candidate_budget.get("max_attempts")
                if (
                    isinstance(candidate_attempts, int)
                    and not isinstance(candidate_attempts, bool)
                    and candidate_attempts > 0
                    and candidate_attempts < max_attempts
                ):
                    issues.append(
                        f"$.candidates[{index}].budget.max_attempts: must not be "
                        "stricter than the frozen plan budget"
                    )
    return issues


def _graph_terminal_issues(document: Mapping[str, Any]) -> list[str]:
    graph = document.get("candidate_graph")
    if not isinstance(graph, Mapping):
        return []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return []
    terminals = [
        node
        for node in nodes
        if isinstance(node, Mapping) and node.get("terminal") is True
    ]
    if not terminals:
        return ["$.candidate_graph.nodes: at least one explicit terminal state required"]
    return []


def verify_generation_plan(document: Any) -> list[str]:
    """Return every structural + semantic problem in one GenerationPlan doc."""
    if not isinstance(document, dict):
        return ["$: document root must be an object"]
    if document.get("schema_name") != OBJECT_GENERATION_PLAN:
        return [f"$.schema_name: expected '{OBJECT_GENERATION_PLAN}'"]
    issues = list(validate_v2_document(document))
    resealed = seal_document(document).get("content_sha256")
    if resealed != document.get("content_sha256"):
        issues.append("$.content_sha256: digest mismatch under re-seal")
    extensions = document.get("extensions")
    freeze_block: Mapping[str, Any] | None = None
    if not isinstance(extensions, Mapping):
        issues.append("$.extensions: frozen dimension bundle missing")
    else:
        raw_block = extensions.get("freeze")
        if not isinstance(raw_block, Mapping):
            issues.append("$.extensions.freeze: frozen dimension bundle missing")
        else:
            freeze_block = raw_block
    if freeze_block is not None:
        issues.extend(_freeze_block_issues(freeze_block, document))
    issues.extend(_failure_vocabulary_issues(document))
    issues.extend(_budget_coherence_issues(document))
    issues.extend(_graph_terminal_issues(document))
    return issues


def freeze_generation_plan(
    proposal: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    compiled: Any,
    config: Mapping[str, Any] | None = None,
    *,
    plan_version: int = 1,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Freeze one release-gate-passing candidate into a ``g1_generation_plan_v2``.

    ``compiled`` is a todo-19 ``CompiledRequest`` (duck-typed ``to_doc()``),
    a hashes/recipe mapping, or ``None`` to seal the candidate's own
    per-point generation recipe.  The returned document is sealed and
    contract-valid; ``content_sha256`` covers rule versions, the complete
    graph hash, atom order, assembly, coordinate schedules, direction,
    method, budget, quality tests and the fallback tree.
    """
    if not isinstance(proposal, Mapping):
        raise PlanFreezeError("PROPOSAL_INVALID", "proposal must be an object")
    proposal_issues = validate_v2_document(dict(proposal))
    if proposal_issues:
        raise PlanFreezeError("PROPOSAL_INVALID", "; ".join(proposal_issues[:5]))
    if proposal.get("schema_version") != "g1_strategy_proposal_v1":
        raise PlanFreezeError("PROPOSAL_INVALID", "expected a g1_strategy_proposal_v1 document")
    if proposal.get("blocking_reasons"):
        raise PlanFreezeError("PROPOSAL_INVALID", "proposal carries blocking_reasons")
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PlanFreezeError("PROPOSAL_INVALID", "proposal has no candidates")
    candidate_id = selected_candidate.get("candidate_id")
    matched: Mapping[str, Any] | None = None
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id:
            matched = candidate
            break
    if matched is None:
        raise PlanFreezeError(
            "CANDIDATE_NOT_IN_PROPOSAL", f"candidate_id {candidate_id!r} not in proposal"
        )
    capability_check = matched.get("capability_check")
    if not isinstance(capability_check, Mapping) or capability_check.get("status") != "pass":
        raise PlanFreezeError(
            "CAPABILITY_NOT_PASSED",
            f"candidate {candidate_id!r} capability_check is not pass",
        )
    failure_reasons = matched.get("failure_reasons")
    if failure_reasons:
        raise PlanFreezeError(
            "CANDIDATE_NOT_CLEAN",
            f"candidate {candidate_id!r} carries failure_reasons",
        )
    if not isinstance(plan_version, int) or isinstance(plan_version, bool) or plan_version < 1:
        raise PlanFreezeError("PROPOSAL_INVALID", "plan_version must be a positive integer")
    reaction_id = str(proposal["reaction_id"])
    plan_id = f"{reaction_id}:gp-v{plan_version}"
    compiled_block, compiled_doc, atom_order = _compiled_binding(compiled, matched)
    graph, graph_sha = _candidate_graph(plan_id, str(candidate_id))
    endpoint_graph_sha = str(
        (proposal.get("graph_input") or {}).get("endpoint_graph_sha256")
        or proposal.get("content_sha256")
        or ""
    )
    complete_graph_sha = sha256_bytes(
        stable_json_dumps(
            {
                "endpoint_graph_sha256": endpoint_graph_sha,
                "candidate_graph_sha256": graph_sha,
            }
        ).encode("utf-8")
    )
    fallback_ids = list(matched.get("fallback_ids") or ())
    alternate_ids = [
        str(candidate["candidate_id"])
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") not in (None, candidate_id)
    ]
    quality_tests = _quality_tests(config)
    plan_budget = _plan_budget(config, [matched])
    policy_hashes = {
        "registry_version": REGISTRY_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "freeze_version": PLAN_FREEZE_VERSION,
        "endpoint_graph_sha256": endpoint_graph_sha,
    }
    freeze_block: dict[str, Any] = {
        "freeze_version": PLAN_FREEZE_VERSION,
        "registry_version": REGISTRY_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "endpoint_graph_sha256": endpoint_graph_sha,
        "candidate_graph_sha256": graph_sha,
        "graph_sha256": complete_graph_sha,
        "selected_candidate_id": str(candidate_id),
        "quality_test_ids": [row["test_id"] for row in quality_tests],
        "fallback_tree": {str(candidate_id): fallback_ids},
        "alternate_candidate_ids": alternate_ids,
        "budget_accounting": list(BUDGET_ACCOUNTING_CATEGORIES),
        "compiled_request": compiled_doc,
        "failure_tree": {
            "version": FAILURE_TREE_VERSION,
            "codes": list(FAILURE_TREE_CODE_ORDER),
        },
    }
    if atom_order is not None:
        freeze_block["atom_order"] = list(atom_order)
    fields: dict[str, Any] = {
        "reaction_id": reaction_id,
        "case_id": str(proposal["case_id"]),
        "split": str(proposal["split"]),
        "plan_id": plan_id,
        "plan_version": plan_version,
        "source_proposal_sha256": str(proposal["content_sha256"]),
        "source_case_sha256": str(proposal["source_case_sha256"]),
        "policy_hashes": policy_hashes,
        "backend": _backend_block(compiled_doc),
        "candidate_graph": graph,
        "candidates": [_scan_candidate_payload(matched)],
        "budget": plan_budget,
        "compiled": compiled_block,
        "quality_tests": quality_tests,
        "extensions": {"freeze": freeze_block},
    }
    if supersedes is not None:
        fields["supersedes"] = supersedes
    return make_generation_plan(plan_id, "frozen", **fields)


def _path_candidate_atom_rows(candidate: Mapping[str, Any]) -> list[int] | None:
    """Frozen atom order carried in the candidate's path_request extension."""
    extensions = candidate.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    block = extensions.get("path_request")
    if not isinstance(block, Mapping):
        return None
    raw = block.get("atom_rows")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    return [int(entry) for entry in raw]


def _path_compiled_binding(
    compiled: Any, candidate: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    """Compiled binding for a PathCandidateV1 freeze.

    ``compiled=None`` seals a path-generation recipe (endpoint xyz blocks in
    the frozen atom order + ``%geom Path`` image chain; per-image recovery
    owned by the backend).  Any other binding is delegated to the scan
    freeze path (``CompiledRequest`` / hashes / recipe mappings).
    """
    if compiled is not None:
        return _compiled_binding(compiled, candidate)
    atom_rows = _path_candidate_atom_rows(candidate)
    image_chain = candidate.get("image_chain")
    image_chain_doc = dict(image_chain) if isinstance(image_chain, Mapping) else {}
    geometries = candidate.get("endpoint_geometries")
    recipe: dict[str, Any] = {
        "recipe_kind": _RECIPE_PATH_REQUEST_V1,
        "engine": DEFAULT_ENGINE,
        "method": DEFAULT_METHOD,
        "adapter_version": DEFAULT_ADAPTER_VERSION,
        "method_kind": candidate.get("method_kind") or "NEB",
        "n_atoms": candidate.get("n_atoms"),
        "n_images": image_chain_doc.get("n_images"),
        "image_chain": image_chain_doc,
        "atom_rows": list(atom_rows) if atom_rows is not None else None,
        "endpoint_geometries": dict(geometries) if isinstance(geometries, Mapping) else {},
        "generation": (
            "NEB path: frozen endpoint xyz blocks in the common atom order + "
            "%geom Path image chain; per-image energies/gradients/convergence "
            "recovered by the backend under the PathRequest recovery protocol"
        ),
    }
    block = {"kind": "recipe", "recipe": recipe}
    doc = {
        "kind": "recipe",
        "compiled_kind": "recipe",
        "recipe_kind": _RECIPE_PATH_REQUEST_V1,
        "recipe": recipe,
        "candidate_id": candidate.get("candidate_id"),
    }
    return block, doc, atom_rows


def freeze_path_candidate_plan(
    path_candidate: Mapping[str, Any],
    *,
    reaction_id: str,
    case_id: str,
    split: str = "unassigned",
    source_case_sha256: str | None = None,
    endpoint_graph_sha256: str | None = None,
    compiled: Any = None,
    config: Mapping[str, Any] | None = None,
    plan_version: int = 1,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Freeze one PathRequest-derived ``PathCandidateV1`` into a plan document.

    ``path_candidate`` is the payload produced by
    ``PathRequest.to_path_candidate`` (contracts_v2 ``PathCandidateV1``).
    Proposals cannot carry path candidates (contracts proposal candidates are
    scan-shaped), so path freezes take the candidate directly — the same
    frozen-plan discipline applies: clean ``failure_reasons``, sealed
    content digest, explicit budget/quality/failure-tree dimensions.
    ``source_case_sha256`` is required (identity binding is never guessed).
    """
    if not isinstance(path_candidate, Mapping):
        raise PlanFreezeError("PATH_CANDIDATE_INVALID", "path_candidate must be an object")
    if path_candidate.get("candidate_kind") != CANDIDATE_KIND_PATH:
        raise PlanFreezeError(
            "PATH_CANDIDATE_INVALID",
            f"candidate_kind={path_candidate.get('candidate_kind')!r}; "
            f"expected {CANDIDATE_KIND_PATH!r}",
        )
    candidate_id = path_candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise PlanFreezeError("PATH_CANDIDATE_INVALID", "candidate_id must be a non-empty string")
    failure_reasons = path_candidate.get("failure_reasons")
    if failure_reasons:
        raise PlanFreezeError(
            "CANDIDATE_NOT_CLEAN",
            f"candidate {candidate_id!r} carries failure_reasons",
        )
    if not isinstance(source_case_sha256, str) or len(source_case_sha256) != 64:
        raise PlanFreezeError(
            "PATH_CANDIDATE_INVALID", "source_case_sha256 must be a 64-char digest"
        )
    if not isinstance(plan_version, int) or isinstance(plan_version, bool) or plan_version < 1:
        raise PlanFreezeError("PATH_CANDIDATE_INVALID", "plan_version must be a positive integer")
    payload = dict(path_candidate)
    payload.setdefault("failure_reasons", [])
    plan_id = f"{reaction_id}:gp-v{plan_version}"
    compiled_block, compiled_doc, atom_order = _path_compiled_binding(compiled, payload)
    graph, graph_sha = _candidate_graph(plan_id, candidate_id)
    extensions = payload.get("extensions")
    request_block = extensions.get("path_request") if isinstance(extensions, Mapping) else None
    endpoint_sha = str(
        endpoint_graph_sha256
        or (
            request_block.get("bundle_content_sha256")
            if isinstance(request_block, Mapping)
            else None
        )
        or ""
    )
    complete_graph_sha = sha256_bytes(
        stable_json_dumps(
            {
                "endpoint_graph_sha256": endpoint_sha,
                "candidate_graph_sha256": graph_sha,
            }
        ).encode("utf-8")
    )
    fallback_ids = list(payload.get("fallback_ids") or ())
    quality_tests = _quality_tests(config)
    plan_budget = _plan_budget(config, [payload])
    policy_hashes = {
        "registry_version": REGISTRY_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "freeze_version": PLAN_FREEZE_VERSION,
        "endpoint_graph_sha256": endpoint_sha,
    }
    freeze_block: dict[str, Any] = {
        "freeze_version": PLAN_FREEZE_VERSION,
        "registry_version": REGISTRY_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "endpoint_graph_sha256": endpoint_sha,
        "candidate_graph_sha256": graph_sha,
        "graph_sha256": complete_graph_sha,
        "selected_candidate_id": candidate_id,
        "quality_test_ids": [row["test_id"] for row in quality_tests],
        "fallback_tree": {candidate_id: fallback_ids},
        "alternate_candidate_ids": [],
        "budget_accounting": list(BUDGET_ACCOUNTING_CATEGORIES),
        "compiled_request": compiled_doc,
        "failure_tree": {
            "version": FAILURE_TREE_VERSION,
            "codes": list(FAILURE_TREE_CODE_ORDER),
        },
    }
    if atom_order is not None:
        freeze_block["atom_order"] = list(atom_order)
    fields: dict[str, Any] = {
        "reaction_id": reaction_id,
        "case_id": case_id,
        "split": split,
        "plan_id": plan_id,
        "plan_version": plan_version,
        "source_proposal_sha256": str(payload.get("source_proposal_sha256") or source_case_sha256),
        "source_case_sha256": source_case_sha256,
        "policy_hashes": policy_hashes,
        "backend": _backend_block(compiled_doc),
        "candidate_graph": graph,
        "candidates": [payload],
        "budget": plan_budget,
        "compiled": compiled_block,
        "quality_tests": quality_tests,
        "extensions": {"freeze": freeze_block},
    }
    if supersedes is not None:
        fields["supersedes"] = supersedes
    return make_generation_plan(plan_id, "frozen", **fields)


__all__ = [
    "BUDGET_ACCOUNTING_CATEGORIES",
    "DEFAULT_QUALITY_TEST_IDS",
    "FAILURE_PREDICATES",
    "FAILURE_TREE_CODES",
    "FAILURE_TREE_CODE_ORDER",
    "FAILURE_TREE_VERSION",
    "PLAN_FREEZE_VERSION",
    "REGISTRY_VERSION",
    "SCHEDULE_VERSION",
    "BudgetLedger",
    "FailureState",
    "PlanFreezeError",
    "active_failure_codes",
    "assert_failure_code",
    "compiled_binding_from_candidate",
    "freeze_generation_plan",
    "freeze_path_candidate_plan",
    "is_failure_code",
    "verify_generation_plan",
]
