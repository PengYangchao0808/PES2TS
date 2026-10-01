"""One-dimensional schedules and the finite candidate tree (todo 14).

Implements design §8 / §8.1 / §8.3 of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``:

- **Named schedule kinds only** — ``linear`` (s=λ), ``event_A_early`` /
  ``event_A_late`` (windowed smoothstep applied to the two event groups a
  coupling graph can explain; never per-bond permutations), and ``smoothstep``
  (s = 3u²−2u³ with u = clip((λ−a)/(b−a), 0, 1); the default window (0, 1)
  gives s(0)=0, s(0.5)=0.5, s(1)=1).  Arbitrary non-uniform λ lists are
  **not** generated here; custom-list compilation is todo 19 and unsmoked
  ``Simul_Scan`` is never produced by this module.
- **Shared λ list** — every driver of one candidate uses the *same* λ ∈ [0, 1]
  grid with ``N`` points from todo 13's point-count rule
  (``FeasibilityReport.n_points``; baseline fallback from config).  Non-uniformity
  lives in the per-driver schedule map s_j(λ), not in per-driver λ lists.
- **First-version budget** — ≤ ``schedule_budget`` schedules per candidate and a
  shared total-scan budget ``max_total_candidates.scan`` accounted conservatively
  as schedules × ``direction_budget`` (direction *choice* is todo 15; this module
  only enforces the budget arithmetic).  Over-budget schedules are recorded with
  ``PRUNED_BY_BUDGET`` — traceable, never silently unproposed.
- **Pre-frozen refinement tree** — adaptive refinement may only pick from a closed
  set encoded by :class:`RefinementRule`: ``N→2N−1`` midpoint halving
  (endpoints preserved), per-interval local refinement, and a frozen
  ``max_refinement_count``.  :func:`refinement_options` enumerates every one-step
  option of a node upfront; runtime never invents new schedule kinds or arbitrary
  λ deformations.
- **Double-distance integrity** — D–H / A–H hydrogen partner distances stay two
  separate driver coordinates; a difference single-B is never representable here
  and any pool/candidate that smuggles one in is rejected
  (:func:`assert_schedule_integrity`).

Pure functions only: no file IO, no ORCA/ACP compilation, no data-tree reads,
no truth.  Inputs are the todo-12 coordinate pool, the todo-13 feasibility
report, and the loaded config mapping.  Direction/assembly *choice* (todo 15)
and selector ranking/budget policy (todo 17) live in their own modules.
"""

# noqa: SIZE_OK — plan-named todo-14 single module (design §8 one-dimensional
# schedules + §8.1 finite schedule kinds + §8.3 points/budget/refinement tree);
# precedent: geometry_feasibility.py (todo 13), coordinate_pool.py (todo 12).

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.scan_strategy.contracts_v2 import (
    MODE_COUPLED_1D,
    MODE_SCHEDULED_1D,
    MODE_SINGLE_1D,
    SCHEDULE_KINDS as CONTRACT_SCHEDULE_KINDS,
)
from pes2ts_core.scan_strategy.coordinate_pool import (
    GEOM_HH,
    GEOM_PARTNER,
    KIND_A,
    KIND_B,
    KIND_D,
    PARTNER_ACCEPTOR,
    PARTNER_DONOR,
    REASON_NO_DRIVER_COORDINATES,
    ROLE_DRIVER,
    CoordinatePool,
    CoordinateRecord,
    DriverSetCandidate,
    EventCoverage,
)
from pes2ts_core.scan_strategy.geometry_feasibility import (
    DEFAULT_POINT_BASELINE,
    DEFAULT_POINT_MAX,
    FeasibilityReport,
)
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Schema + vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_SCHEDULES: Final[str] = "g1_scan_schedules_v1"

#: Design §8.1 schedule kinds (single source: contracts_v2.SCHEDULE_KINDS).
SCHEDULE_LINEAR: Final[str] = "linear"
SCHEDULE_EVENT_A_EARLY: Final[str] = "event_A_early"
SCHEDULE_EVENT_A_LATE: Final[str] = "event_A_late"
SCHEDULE_SMOOTHSTEP: Final[str] = "smoothstep"
SCHEDULE_KIND_ORDER: Final[tuple[str, ...]] = CONTRACT_SCHEDULE_KINDS

#: Design §8.3 budget-pruning trace code.
CODE_PRUNED_BY_BUDGET: Final[str] = "PRUNED_BY_BUDGET"

#: Integrity violation codes (raised, never silently absorbed).
CODE_H_DOUBLE_DISTANCE_COLLAPSED: Final[str] = "H_DOUBLE_DISTANCE_COLLAPSED"
CODE_DIFFERENCE_COORDINATE_FORBIDDEN: Final[str] = "DIFFERENCE_COORDINATE_FORBIDDEN"
CODE_DRIVER_ID_UNRESOLVED: Final[str] = "DRIVER_ID_UNRESOLVED"
CODE_DRIVER_KIND_INVALID: Final[str] = "DRIVER_KIND_INVALID"
CODE_DRIVER_ROLE_MISMATCH: Final[str] = "DRIVER_ROLE_MISMATCH"
CODE_DRIVER_DUPLICATE: Final[str] = "DRIVER_DUPLICATE"
CODE_SMOKE_NOT_PRODUCED_HERE: Final[str] = "UNSCHEDULED_COMPILE_NOT_HERE"

INTEGRITY_VIOLATION_PREFIX: Final[str] = "SCHEDULE_INTEGRITY_VIOLATION"

#: Frozen smoothstep windows (TODO: calibrate — 待校准).
WINDOW_FULL: Final[tuple[float, float]] = (0.0, 1.0)
WINDOW_EARLY: Final[tuple[float, float]] = (0.0, 0.5)
WINDOW_LATE: Final[tuple[float, float]] = (0.5, 1.0)

#: Frozen refinement vocabulary (design §8.3; TODO: calibrate max count).
REFINEMENT_KIND_DOUBLING: Final[str] = "doubling"
REFINEMENT_KIND_LOCAL_INTERVAL: Final[str] = "local_interval"
REFINEMENT_KINDS: Final[tuple[str, ...]] = (
    REFINEMENT_KIND_DOUBLING,
    REFINEMENT_KIND_LOCAL_INTERVAL,
)
DEFAULT_MAX_REFINEMENT_COUNT: Final[int] = 2  # TODO: calibrate — 待校准

#: λ / s rounding so serialization is byte-stable across runs.
LAMBDA_DECIMALS: Final[int] = 12

#: Geometry-kind / note fragments that would encode a difference coordinate.
_FORBIDDEN_GEOMETRY_FRAGMENTS: Final[tuple[str, ...]] = (
    "difference",
    "delta",
    "relative",
    "d1_minus",
    "d2_minus",
)

_SUPPORTED_KIND_ARITY: Final[dict[str, int]] = {KIND_B: 2, KIND_A: 3, KIND_D: 4}


# ---------------------------------------------------------------------------
# Policy (config boundary; scan_strategy section of config/defaults.yaml).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SchedulePolicy:
    """First-version schedule/direction/total budget policy (design §8.1/§8.3)."""

    schedule_budget: int
    direction_budget: int
    assembly_candidate_budget: int
    max_total_candidates_scan: int
    max_total_candidates_path_neb: int
    point_baseline: int
    point_max: int
    max_refinement_count: int

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "schedule_budget": self.schedule_budget,
            "direction_budget": self.direction_budget,
            "assembly_candidate_budget": self.assembly_candidate_budget,
            "max_total_candidates_scan": self.max_total_candidates_scan,
            "max_total_candidates_path_neb": self.max_total_candidates_path_neb,
            "point_baseline": self.point_baseline,
            "point_max": self.point_max,
            "max_refinement_count": self.max_refinement_count,
        }


def policy_from_config(config: Mapping[str, Any]) -> SchedulePolicy:
    """Parse ``scan_strategy`` schedule-budget keys from a loaded config mapping.

    Missing keys fall back to the documented defaults mirroring
    ``config/defaults.yaml`` (todo 2).  Present-but-invalid values raise
    ``ValueError`` with a ``POLICY_INVALID:`` prefix.  The optional
    ``scan_strategy.max_refinement_count`` key is read when present; this
    module does not modify ``config/defaults.yaml``.
    """
    section = config.get("scan_strategy")
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("POLICY_INVALID: scan_strategy must be a mapping")
    totals = section.get("max_total_candidates")
    if totals is None:
        totals = {}
    if not isinstance(totals, Mapping):
        raise ValueError(
            "POLICY_INVALID: scan_strategy.max_total_candidates must be a mapping"
        )
    limits = section.get("point_limits")
    if limits is None:
        limits = {}
    if not isinstance(limits, Mapping):
        raise ValueError("POLICY_INVALID: scan_strategy.point_limits must be a mapping")

    schedule_budget = _int_key(section, "schedule_budget", 3)
    direction_budget = _int_key(section, "direction_budget", 2)
    assembly_budget = _int_key(section, "assembly_candidate_budget", 2)
    max_scan = _int_key(totals, "scan", 6)
    max_path = _int_key(totals, "path_neb", 1)
    baseline = _int_key(limits, "baseline", DEFAULT_POINT_BASELINE)
    point_max = _int_key(limits, "max", DEFAULT_POINT_MAX)
    max_refinement = _int_key(section, "max_refinement_count", DEFAULT_MAX_REFINEMENT_COUNT)

    if schedule_budget < 0:
        raise ValueError("POLICY_INVALID: schedule_budget must be >= 0")
    if direction_budget < 1:
        raise ValueError("POLICY_INVALID: direction_budget must be >= 1")
    if assembly_budget < 0:
        raise ValueError("POLICY_INVALID: assembly_candidate_budget must be >= 0")
    if max_scan < 0 or max_path < 0:
        raise ValueError(
            "POLICY_INVALID: max_total_candidates entries must be >= 0"
        )
    if baseline < 2:
        raise ValueError("POLICY_INVALID: point_limits.baseline must be >= 2")
    if point_max < baseline:
        raise ValueError(
            "POLICY_INVALID: point_limits.max must be >= point_limits.baseline"
        )
    if max_refinement < 0:
        raise ValueError("POLICY_INVALID: max_refinement_count must be >= 0")
    return SchedulePolicy(
        schedule_budget=schedule_budget,
        direction_budget=direction_budget,
        assembly_candidate_budget=assembly_budget,
        max_total_candidates_scan=max_scan,
        max_total_candidates_path_neb=max_path,
        point_baseline=baseline,
        point_max=point_max,
        max_refinement_count=max_refinement,
    )


def _int_key(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"POLICY_INVALID: {key} must be an integer")
    if isinstance(raw, str):
        try:
            return int(raw, 10)
        except ValueError:
            raise ValueError(f"POLICY_INVALID: {key} must be an integer") from None
    if not isinstance(raw, int):
        raise ValueError(f"POLICY_INVALID: {key} must be an integer")
    return raw


# ---------------------------------------------------------------------------
# Schedule math primitives (design §8 / §8.1).
# ---------------------------------------------------------------------------
def smoothstep_s(lam: float, window: tuple[float, float]) -> float:
    """Return s = 3u²−2u³ with u = clip((λ−a)/(b−a), 0, 1) for ``window=(a, b)``.

    Endpoints are preserved for any finite window: s(a)=0, s(b)=1; outside the
    window s clamps to 0/1.  The full-range window (0, 1) yields s(0)=0,
    s(0.5)=0.5, s(1)=1 and is monotonically non-decreasing on [0, 1].
    """
    a, b = window
    if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
        raise ValueError(
            f"SCHEDULE_WINDOW_INVALID: window {window!r} must be finite with b > a"
        )
    if not math.isfinite(lam):
        raise ValueError(f"SCHEDULE_LAMBDA_INVALID: λ={lam!r} must be finite")
    u = min(1.0, max(0.0, (lam - a) / (b - a)))
    return 3.0 * u * u - 2.0 * u * u * u


def uniform_lambda_grid(n_points: int) -> tuple[float, ...]:
    """Return the shared uniform λ grid on [0, 1] with ``n_points`` entries.

    ``N = max(2, n_points)``; values are rounded to :data:`LAMBDA_DECIMALS` so
    serialization is byte-stable.
    """
    n = max(2, int(n_points))
    return tuple(round(i / (n - 1), LAMBDA_DECIMALS) for i in range(n))


def double_lambda_grid(lam: Sequence[float]) -> tuple[float, ...]:
    """Return the N→2N−1 midpoint-halving refinement of ``lam``.

    One midpoint is inserted between every adjacent pair; endpoints are
    preserved exactly (design §8.3 ``N→2N−1``).
    """
    values = [float(v) for v in lam]
    if len(values) < 2:
        raise ValueError("SCHEDULE_GRID_INVALID: need at least 2 λ points")
    out: list[float] = []
    for index, left in enumerate(values):
        out.append(round(left, LAMBDA_DECIMALS))
        if index + 1 < len(values):
            out.append(round(0.5 * (left + values[index + 1]), LAMBDA_DECIMALS))
    return tuple(out)


def refine_lambda_interval(
    lam: Sequence[float], start_index: int, end_index: int
) -> tuple[float, ...]:
    """Return ``lam`` with midpoints inserted in each pair in [start, end].

    For a single adjacent pair ``(i, i+1)`` this yields N+1 points and touches
    only that interval; endpoints of the whole grid are preserved.
    """
    values = [float(v) for v in lam]
    n = len(values)
    if n < 2:
        raise ValueError("SCHEDULE_GRID_INVALID: need at least 2 λ points")
    if not (0 <= start_index < end_index < n):
        raise ValueError(
            f"SCHEDULE_INTERVAL_INVALID: indices ({start_index}, {end_index}) "
            f"out of range for N={n}"
        )
    out: list[float] = []
    for index in range(n):
        out.append(round(values[index], LAMBDA_DECIMALS))
        if start_index <= index < end_index:
            out.append(round(0.5 * (values[index] + values[index + 1]), LAMBDA_DECIMALS))
    return tuple(out)


def s_value_for_window(kind: str, window: tuple[float, float] | None, lam: float) -> float:
    """Return the schedule map s(λ) for one driver under a named kind/window.

    ``window=None`` is the linear map s=λ (the honest default for drivers that
    no named early/late group explains — never an arbitrary non-uniform list).
    """
    if window is None:
        value = float(lam)
    else:
        value = smoothstep_s(lam, window)
    return round(value, LAMBDA_DECIMALS)


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventGroup:
    """One event group of a candidate's drivers (for event_A schedules)."""

    event_id: str
    driver_coordinate_ids: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "event_id": self.event_id,
            "driver_coordinate_ids": list(self.driver_coordinate_ids),
        }


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    """One named schedule of one coordinate candidate.

    ``lambda_values`` is the **shared** λ list; every driver of the candidate
    uses it.  Non-uniformity lives in ``s_by_driver`` / ``driver_windows``.
    """

    schedule_id: str
    kind: str
    n_points: int
    depth: int
    lambda_values: tuple[float, ...]
    s_by_driver: tuple[tuple[str, tuple[float, ...]], ...]
    driver_windows: tuple[tuple[str, tuple[float, float] | None], ...]
    suggested_mode: str
    requires_backend_smoke: bool
    notes: str | None

    def s_of(self, driver_id: str) -> tuple[float, ...]:
        """Return the s(λ) list of one driver; raises ``KeyError`` when absent."""
        for cid, values in self.s_by_driver:
            if cid == driver_id:
                return values
        raise KeyError(driver_id)

    def window_of(self, driver_id: str) -> tuple[float, float] | None:
        """Return the schedule window of one driver (None = linear s=λ)."""
        for cid, window in self.driver_windows:
            if cid == driver_id:
                return window
        raise KeyError(driver_id)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "schedule_id": self.schedule_id,
            "kind": self.kind,
            "n_points": self.n_points,
            "depth": self.depth,
            "lambda_values": list(self.lambda_values),
            "s_by_driver": [
                {"driver_id": cid, "s_values": list(values)}
                for cid, values in self.s_by_driver
            ],
            "driver_windows": [
                {
                    "driver_id": cid,
                    "window": None if window is None else [window[0], window[1]],
                }
                for cid, window in self.driver_windows
            ],
            "suggested_mode": self.suggested_mode,
            "requires_backend_smoke": self.requires_backend_smoke,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class PrunedCandidate:
    """One over-budget schedule, recorded with ``PRUNED_BY_BUDGET``.

    Never silently unproposed: the would-be schedule id, kind and λ list are
    retained so todo 17 can surface the pruning in ``failure_reasons``.
    """

    pruned_id: str
    code: str
    schedule_id: str
    kind: str
    n_points: int
    lambda_values: tuple[float, ...]
    reason: str
    budget: SchedulePolicy

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "pruned_id": self.pruned_id,
            "code": self.code,
            "schedule_id": self.schedule_id,
            "kind": self.kind,
            "n_points": self.n_points,
            "lambda_values": list(self.lambda_values),
            "reason": self.reason,
            "budget": self.budget.to_record(),
        }


@dataclass(frozen=True, slots=True)
class RefinementRule:
    """Pre-frozen refinement vocabulary + depth cap (design §8.3).

    The closed set is ``doubling`` (N→2N−1) plus ``local_interval`` (per
    adjacent-pair midpoint insertion).  ``max_refinement_count`` is the frozen
    depth cap; runtime selection never invents new kinds.
    """

    max_refinement_count: int
    kinds: tuple[str, ...] = REFINEMENT_KINDS

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "max_refinement_count": self.max_refinement_count,
            "kinds": list(self.kinds),
        }


DEFAULT_REFINEMENT_RULE: Final[RefinementRule] = RefinementRule(
    max_refinement_count=DEFAULT_MAX_REFINEMENT_COUNT
)


@dataclass(frozen=True, slots=True)
class RefinementOption:
    """One one-step refinement of a schedule node (closed-set member)."""

    option_id: str
    kind: str
    parent_schedule_id: str
    parent_kind: str
    parent_n_points: int
    n_points: int
    depth: int
    lambda_values: tuple[float, ...]
    s_by_driver: tuple[tuple[str, tuple[float, ...]], ...]
    driver_windows: tuple[tuple[str, tuple[float, float] | None], ...]
    local_interval: tuple[int, int] | None
    window: tuple[float, float] | None
    notes: str

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "option_id": self.option_id,
            "kind": self.kind,
            "parent_schedule_id": self.parent_schedule_id,
            "parent_kind": self.parent_kind,
            "parent_n_points": self.parent_n_points,
            "n_points": self.n_points,
            "depth": self.depth,
            "lambda_values": list(self.lambda_values),
            "s_by_driver": [
                {"driver_id": cid, "s_values": list(values)}
                for cid, values in self.s_by_driver
            ],
            "driver_windows": [
                {
                    "driver_id": cid,
                    "window": None if window is None else [window[0], window[1]],
                }
                for cid, window in self.driver_windows
            ],
            "local_interval": (
                None if self.local_interval is None else list(self.local_interval)
            ),
            "window": None if self.window is None else [self.window[0], self.window[1]],
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class RefinementTree:
    """Pre-frozen refinement tree of one schedule root (design §8.3).

    The tree is the transitive closure of :func:`refinement_options` under
    :attr:`rule`.  Every one-step option of every node is computable upfront;
    runtime never invents new schedules outside this closure.
    """

    root_schedule_id: str
    rule: RefinementRule

    def options_from(self, schedule: ScheduleSpec) -> tuple[RefinementOption, ...]:
        """Return the closed set of one-step refinements of ``schedule``."""
        return refinement_options(schedule, rule=self.rule)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "root_schedule_id": self.root_schedule_id,
            "rule": self.rule.to_record(),
        }


@dataclass(frozen=True, slots=True)
class CandidateWithFeasibility:
    """One todo-12 coordinate candidate plus its todo-13 feasibility verdict."""

    candidate: DriverSetCandidate
    pool: CoordinatePool
    feasibility: FeasibilityReport
    start_endpoint: str = "R"


@dataclass(frozen=True, slots=True)
class ScheduledCandidate:
    """Finite schedule tree of one coordinate candidate (design §8.1/§8.3).

    ``schedules`` are within budget; ``pruned`` carries every over-budget
    proposal with ``PRUNED_BY_BUDGET``.  Direction/assembly *choice* is todo 15;
    only the budget arithmetic (schedules × ``direction_budget``) is applied
    here.  No ORCA input is compiled — custom-list / ``Simul_Scan`` compilation
    is todo 19 and gated on backend smoke.
    """

    schema_version: str
    candidate_id: str
    driver_coordinate_ids: tuple[str, ...]
    monitor_coordinate_ids: tuple[str, ...]
    guard_coordinate_ids: tuple[str, ...]
    target_test_coordinate_ids: tuple[str, ...]
    event_coverage: tuple[EventCoverage, ...]
    event_groups: tuple[EventGroup, ...]
    n_points: int
    n_points_required: int | None
    point_budget_exceeded: bool
    start_endpoint: str
    schedules: tuple[ScheduleSpec, ...]
    pruned: tuple[PrunedCandidate, ...]
    refinement_rule: RefinementRule
    budget: SchedulePolicy
    feasibility_ok: bool
    feasibility_failure_codes: tuple[str, ...]
    integrity_ok: bool
    integrity_notes: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def n_schedules_active(self) -> int:
        """Count of within-budget schedules."""
        return len(self.schedules)

    @property
    def n_schedules_pruned(self) -> int:
        """Count of ``PRUNED_BY_BUDGET`` records."""
        return len(self.pruned)

    @property
    def n_schedules_proposed(self) -> int:
        """Count of named schedules generated before budget partition."""
        return len(self.schedules) + len(self.pruned)

    def lambda_by_schedule_id(self) -> dict[str, tuple[float, ...]]:
        """Return schedule_id → shared λ list for every proposed schedule."""
        out: dict[str, tuple[float, ...]] = {
            spec.schedule_id: spec.lambda_values for spec in self.schedules
        }
        for record in self.pruned:
            out[record.schedule_id] = record.lambda_values
        return out

    def refinement_tree(self, schedule_id: str) -> RefinementTree:
        """Return the pre-frozen refinement tree rooted at one active schedule."""
        for spec in self.schedules:
            if spec.schedule_id == schedule_id:
                return RefinementTree(
                    root_schedule_id=spec.schedule_id, rule=self.refinement_rule
                )
        raise KeyError(schedule_id)

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; never contains truth-derived keys."""
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "driver_coordinate_ids": list(self.driver_coordinate_ids),
            "monitor_coordinate_ids": list(self.monitor_coordinate_ids),
            "guard_coordinate_ids": list(self.guard_coordinate_ids),
            "target_test_coordinate_ids": list(self.target_test_coordinate_ids),
            "event_coverage": [row.to_record() for row in self.event_coverage],
            "event_groups": [group.to_record() for group in self.event_groups],
            "n_points": self.n_points,
            "n_points_required": self.n_points_required,
            "point_budget_exceeded": self.point_budget_exceeded,
            "start_endpoint": self.start_endpoint,
            "schedules": [spec.to_record() for spec in self.schedules],
            "pruned": [record.to_record() for record in self.pruned],
            "refinement_rule": self.refinement_rule.to_record(),
            "budget": self.budget.to_record(),
            "feasibility_ok": self.feasibility_ok,
            "feasibility_failure_codes": list(self.feasibility_failure_codes),
            "integrity_ok": self.integrity_ok,
            "integrity_notes": list(self.integrity_notes),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        """Canonical serialization (determinism test surface)."""
        return stable_json_dumps(self.to_doc())


# ---------------------------------------------------------------------------
# Refinement options (closed set; pre-frozen tree).
# ---------------------------------------------------------------------------
def refinement_options(
    schedule: ScheduleSpec, *, rule: RefinementRule | None = None
) -> tuple[RefinementOption, ...]:
    """Return the closed set of one-step refinements of ``schedule``.

    The set is fully enumerable upfront: one ``doubling`` option (N→2N−1,
    endpoints preserved) plus one ``local_interval`` option per adjacent λ pair.
    When ``schedule.depth`` has reached ``rule.max_refinement_count`` the set is
    empty — runtime can never invent deeper or differently-shaped schedules.
    """
    frozen = rule if rule is not None else DEFAULT_REFINEMENT_RULE
    if schedule.depth >= frozen.max_refinement_count:
        return ()
    if schedule.kind not in SCHEDULE_KIND_ORDER:
        raise ValueError(
            f"SCHEDULE_KIND_INVALID: {schedule.kind!r} not in {SCHEDULE_KIND_ORDER}"
        )
    options: list[RefinementOption] = []
    lam = schedule.lambda_values
    doubled = double_lambda_grid(lam)
    options.append(
        RefinementOption(
            option_id=f"ref:{schedule.schedule_id}:{REFINEMENT_KIND_DOUBLING}",
            kind=REFINEMENT_KIND_DOUBLING,
            parent_schedule_id=schedule.schedule_id,
            parent_kind=schedule.kind,
            parent_n_points=schedule.n_points,
            n_points=len(doubled),
            depth=schedule.depth + 1,
            lambda_values=doubled,
            s_by_driver=_recompute_s_by_driver(schedule, doubled),
            driver_windows=schedule.driver_windows,
            local_interval=None,
            window=None,
            notes="N→2N−1 midpoint halving; endpoints preserved",
        )
    )
    for index in range(len(lam) - 1):
        refined = refine_lambda_interval(lam, index, index + 1)
        options.append(
            RefinementOption(
                option_id=(
                    f"ref:{schedule.schedule_id}:{REFINEMENT_KIND_LOCAL_INTERVAL}:{index}"
                ),
                kind=REFINEMENT_KIND_LOCAL_INTERVAL,
                parent_schedule_id=schedule.schedule_id,
                parent_kind=schedule.kind,
                parent_n_points=schedule.n_points,
                n_points=len(refined),
                depth=schedule.depth + 1,
                lambda_values=refined,
                s_by_driver=_recompute_s_by_driver(schedule, refined),
                driver_windows=schedule.driver_windows,
                local_interval=(index, index + 1),
                window=(lam[index], lam[index + 1]),
                notes=(
                    f"local midpoint insertion in λ interval "
                    f"[{lam[index]}, {lam[index + 1]}]"
                ),
            )
        )
    return tuple(options)


def refinement_tree(
    schedule: ScheduleSpec, *, rule: RefinementRule | None = None
) -> RefinementTree:
    """Return the pre-frozen refinement tree rooted at ``schedule``."""
    frozen = rule if rule is not None else DEFAULT_REFINEMENT_RULE
    return RefinementTree(root_schedule_id=schedule.schedule_id, rule=frozen)


def _recompute_s_by_driver(
    parent: ScheduleSpec, lam_new: Sequence[float]
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Recompute per-driver s(λ) on a refined grid using the parent's windows."""
    windows = dict(parent.driver_windows)
    rows: list[tuple[str, tuple[float, ...]]] = []
    for driver_id in sorted(windows):
        window = windows[driver_id]
        rows.append(
            (
                driver_id,
                tuple(s_value_for_window(parent.kind, window, value) for value in lam_new),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Integrity (double-distance never collapsed to a difference single-B).
# ---------------------------------------------------------------------------
def assert_schedule_integrity(
    pool: CoordinatePool, candidate: DriverSetCandidate
) -> tuple[str, ...]:
    """Assert pool/candidate integrity for schedule generation (design §8.1).

    Hard rules enforced here:

    1. Every driver coordinate id resolves in the pool with role ``driver`` and
       a supported kind/arity (B=2, A=3, D=4 atoms).
    2. Driver ids are distinct and ``n_drivers`` matches their count.
    3. No pool coordinate may encode a *difference* single-B (D–H / A–H partner
       distances are two separate coordinates; a difference coordinate is
       forbidden by design and unrepresentable as a native B).
    4. Two hydrogen partner-distance/hh records that share an event and
       hydrogen map must keep distinct atom maps and distinct coordinate ids —
       a merge would collapse the double distance.

    Violations raise ``ValueError`` with an ``SCHEDULE_INTEGRITY_VIOLATION:``
    prefix; the caller (``build_schedules``) never emits schedules on top of an
    unverified pool.
    """
    notes: list[str] = []
    by_id = {record.coordinate_id: record for record in pool.coordinates}
    driver_ids = candidate.driver_coordinate_ids
    if len(set(driver_ids)) != len(driver_ids):
        raise ValueError(
            f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_DRIVER_DUPLICATE}: "
            f"driver ids {driver_ids} are not distinct"
        )
    for driver_id in driver_ids:
        record = by_id.get(driver_id)
        if record is None:
            raise ValueError(
                f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_DRIVER_ID_UNRESOLVED}: "
                f"{driver_id} not in pool"
            )
        if record.role != ROLE_DRIVER:
            raise ValueError(
                f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_DRIVER_ROLE_MISMATCH}: "
                f"{driver_id} has role {record.role!r}, expected {ROLE_DRIVER!r}"
            )
        arity = _SUPPORTED_KIND_ARITY.get(record.kind)
        if arity is None or len(record.atom_maps) != arity:
            raise ValueError(
                f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_DRIVER_KIND_INVALID}: "
                f"{driver_id} kind={record.kind!r} maps={record.atom_maps} "
                f"(expected one of {sorted(_SUPPORTED_KIND_ARITY)} with matching arity)"
            )
    if candidate.n_drivers != len(driver_ids):
        raise ValueError(
            f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_DRIVER_KIND_INVALID}: "
            f"n_drivers={candidate.n_drivers} != len(driver_coordinate_ids)="
            f"{len(driver_ids)} (drivers may never be merged)"
        )
    notes.append(f"driver_ids_resolved={len(driver_ids)}")

    # --- Difference-coordinate ban + double-distance non-collapse -----------
    partner_rows: dict[tuple[str, int], list[CoordinateRecord]] = {}
    for record in pool.coordinates:
        _assert_not_difference_coordinate(record)
        if record.role != ROLE_DRIVER:
            continue
        origin = record.origin
        if origin.source_kind != "hydrogen_event":
            continue
        if origin.geometry_kind not in (GEOM_PARTNER, GEOM_HH):
            continue
        if origin.partner_role not in (PARTNER_DONOR, PARTNER_ACCEPTOR):
            continue
        if origin.hydrogen_map is None:
            continue
        key = (origin.event_id or "", int(origin.hydrogen_map))
        partner_rows.setdefault(key, []).append(record)
    for key, rows in sorted(partner_rows.items()):
        maps = [row.atom_maps for row in rows]
        ids = [row.coordinate_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_H_DOUBLE_DISTANCE_COLLAPSED}: "
                f"H event {key[0]!r} map {key[1]} reuses coordinate ids {ids}"
            )
        if len(maps) != len(set(maps)):
            raise ValueError(
                f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_H_DOUBLE_DISTANCE_COLLAPSED}: "
                f"H event {key[0]!r} map {key[1]} has duplicate partner atom maps "
                f"{maps} — D–H/A–H distances must stay separate coordinates, "
                "never one difference single-B"
            )
    notes.append("no_difference_coordinates")
    notes.append("double_distance_records_distinct")
    return tuple(notes)


def _assert_not_difference_coordinate(record: CoordinateRecord) -> None:
    """Raise when a pool record claims to encode a difference coordinate."""
    origin = record.origin
    haystacks = (
        origin.geometry_kind or "",
        origin.note or "",
        origin.partner_role or "",
    )
    for text in haystacks:
        lowered = text.lower()
        for fragment in _FORBIDDEN_GEOMETRY_FRAGMENTS:
            if fragment in lowered:
                raise ValueError(
                    f"{INTEGRITY_VIOLATION_PREFIX}: {CODE_DIFFERENCE_COORDINATE_FORBIDDEN}: "
                    f"{record.coordinate_id} origin claims {text!r} — D–H/A–H "
                    "double distances are two driver coordinates, never a "
                    "difference single-B"
                )


# ---------------------------------------------------------------------------
# Schedule generation (design §8.1).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _ScheduleContext:
    """Shared inputs for one candidate's schedule expansion."""

    pool: CoordinatePool
    driver_ids: tuple[str, ...]
    lambda_values: tuple[float, ...]
    event_groups: tuple[EventGroup, ...]
    n_drivers: int


def build_schedules(
    candidate_with_feasibility: CandidateWithFeasibility, config: Mapping[str, Any]
) -> ScheduledCandidate:
    """Build the finite first-version schedule tree of one coordinate candidate.

    Positional API per todo 14.  Consumes the todo-12 ``DriverSetCandidate`` +
    ``CoordinatePool`` and the todo-13 ``FeasibilityReport``; the point count N
    comes from the feasibility rule (baseline fallback from config).  Returns
    within-budget :class:`ScheduleSpec` entries plus ``PRUNED_BY_BUDGET``
    records for every over-budget proposal.  Direction/assembly choice and
    selector ranking are *not* performed here.
    """
    policy = policy_from_config(config)
    entry = candidate_with_feasibility
    integrity_notes = assert_schedule_integrity(entry.pool, entry.candidate)
    feasibility_codes = tuple(failure.code for failure in entry.feasibility.failures)
    driver_ids = entry.candidate.driver_coordinate_ids
    notes: list[str] = []

    if not driver_ids:
        notes.append(REASON_NO_DRIVER_COORDINATES)
        return ScheduledCandidate(
            schema_version=SCHEMA_SCHEDULES,
            candidate_id=entry.candidate.candidate_id,
            driver_coordinate_ids=(),
            monitor_coordinate_ids=entry.candidate.monitor_coordinate_ids,
            guard_coordinate_ids=entry.candidate.guard_coordinate_ids,
            target_test_coordinate_ids=entry.candidate.target_test_coordinate_ids,
            event_coverage=entry.candidate.event_coverage,
            event_groups=(),
            n_points=policy.point_baseline,
            n_points_required=entry.feasibility.n_points_required,
            point_budget_exceeded=entry.feasibility.point_budget_exceeded,
            start_endpoint=entry.start_endpoint,
            schedules=(),
            pruned=(),
            refinement_rule=RefinementRule(max_refinement_count=policy.max_refinement_count),
            budget=policy,
            feasibility_ok=entry.feasibility.ok,
            feasibility_failure_codes=feasibility_codes,
            integrity_ok=True,
            integrity_notes=integrity_notes,
            notes=tuple(notes),
        )

    n_points = _resolve_n_points(entry.feasibility, policy)
    lam = uniform_lambda_grid(n_points)
    groups = _driver_event_groups(entry.pool, entry.candidate)
    kinds = _applicable_kinds(groups)
    context = _ScheduleContext(
        pool=entry.pool,
        driver_ids=driver_ids,
        lambda_values=lam,
        event_groups=groups,
        n_drivers=len(driver_ids),
    )
    proposed = tuple(
        _build_schedule(
            kind,
            f"sched:{entry.candidate.candidate_id}:{kind}",
            context,
        )
        for kind in kinds
    )
    active, pruned = _partition_by_budget(proposed, policy)
    if pruned:
        notes.append(
            f"{CODE_PRUNED_BY_BUDGET}:{len(pruned)} schedule(s) exceeded "
            "schedule_budget / max_total_candidates.scan arithmetic"
        )
    if entry.feasibility.point_budget_exceeded:
        notes.append("POINT_BUDGET_EXCEEDED from feasibility; decision owned by todo 17")
    if any(spec.requires_backend_smoke for spec in active):
        notes.append(
            "SCHEDULED_1D / coupled schedules are data only — custom-list and "
            "Simul_Scan compilation is todo 19; unsmoked Simul_Scan is never produced here"
        )

    return ScheduledCandidate(
        schema_version=SCHEMA_SCHEDULES,
        candidate_id=entry.candidate.candidate_id,
        driver_coordinate_ids=driver_ids,
        monitor_coordinate_ids=entry.candidate.monitor_coordinate_ids,
        guard_coordinate_ids=entry.candidate.guard_coordinate_ids,
        target_test_coordinate_ids=entry.candidate.target_test_coordinate_ids,
        event_coverage=entry.candidate.event_coverage,
        event_groups=groups,
        n_points=n_points,
        n_points_required=entry.feasibility.n_points_required,
        point_budget_exceeded=entry.feasibility.point_budget_exceeded,
        start_endpoint=entry.start_endpoint,
        schedules=active,
        pruned=pruned,
        refinement_rule=RefinementRule(max_refinement_count=policy.max_refinement_count),
        budget=policy,
        feasibility_ok=entry.feasibility.ok,
        feasibility_failure_codes=feasibility_codes,
        integrity_ok=True,
        integrity_notes=integrity_notes,
        notes=tuple(notes),
    )


def _resolve_n_points(feasibility: FeasibilityReport, policy: SchedulePolicy) -> int:
    """Return N from todo 13's point-count rule, else the config baseline."""
    if feasibility.n_points is not None:
        n = int(feasibility.n_points)
    else:
        n = policy.point_baseline
    return max(2, n)


def _driver_event_groups(
    pool: CoordinatePool, candidate: DriverSetCandidate
) -> tuple[EventGroup, ...]:
    """Partition candidate drivers by their primary (min) event id."""
    by_id = {record.coordinate_id: record for record in pool.coordinates}
    grouped: dict[str, list[str]] = {}
    for driver_id in candidate.driver_coordinate_ids:
        record = by_id[driver_id]
        if not record.event_ids:
            continue
        primary = min(record.event_ids)
        grouped.setdefault(primary, []).append(driver_id)
    return tuple(
        EventGroup(
            event_id=event_id,
            driver_coordinate_ids=tuple(sorted(members)),
        )
        for event_id, members in sorted(grouped.items())
    )


def _applicable_kinds(event_groups: Sequence[EventGroup]) -> tuple[str, ...]:
    """Return the named schedule kinds this candidate may carry (design §8.1).

    ``linear`` and ``smoothstep`` always apply.  ``event_A_early`` /
    ``event_A_late`` apply only when the coupling graph explains exactly two
    event groups among the candidate's drivers — never a per-bond permutation.
    """
    kinds: list[str] = [SCHEDULE_LINEAR]
    if len(event_groups) == 2:
        kinds.append(SCHEDULE_EVENT_A_EARLY)
        kinds.append(SCHEDULE_EVENT_A_LATE)
    kinds.append(SCHEDULE_SMOOTHSTEP)
    return tuple(kinds)


def _window_for_driver(
    kind: str, driver_id: str, context: _ScheduleContext
) -> tuple[float, float] | None:
    """Return one driver's schedule window; ``None`` means the linear map s=λ."""
    if kind == SCHEDULE_LINEAR:
        return None
    if kind == SCHEDULE_SMOOTHSTEP:
        return WINDOW_FULL
    if kind not in (SCHEDULE_EVENT_A_EARLY, SCHEDULE_EVENT_A_LATE):
        raise ValueError(f"SCHEDULE_KIND_INVALID: {kind!r} not in {SCHEDULE_KIND_ORDER}")
    groups = context.event_groups
    if len(groups) != 2:
        return None
    members_a = set(groups[0].driver_coordinate_ids)
    members_b = set(groups[1].driver_coordinate_ids)
    a_early = kind == SCHEDULE_EVENT_A_EARLY
    if driver_id in members_a:
        return WINDOW_EARLY if a_early else WINDOW_LATE
    if driver_id in members_b:
        return WINDOW_LATE if a_early else WINDOW_EARLY
    return None


def _build_schedule(
    kind: str, schedule_id: str, context: _ScheduleContext
) -> ScheduleSpec:
    """Build one named schedule spec (shared λ; per-driver s windows)."""
    if kind not in SCHEDULE_KIND_ORDER:
        raise ValueError(
            f"SCHEDULE_KIND_INVALID: {kind!r} not in {SCHEDULE_KIND_ORDER}"
        )
    lam = context.lambda_values
    driver_windows: list[tuple[str, tuple[float, float] | None]] = []
    for driver_id in sorted(context.driver_ids):
        window = _window_for_driver(kind, driver_id, context)
        driver_windows.append((driver_id, window))
    s_by_driver = tuple(
        (
            driver_id,
            tuple(s_value_for_window(kind, window, value) for value in lam),
        )
        for driver_id, window in driver_windows
    )
    if kind == SCHEDULE_LINEAR:
        suggested = MODE_SINGLE_1D if context.n_drivers == 1 else MODE_COUPLED_1D
        # SINGLE_1D linear rides the existing uniform single-B ACP path;
        # COUPLED_1D linear needs Simul_Scan, which is capability-gated (todo 19).
        smoke = context.n_drivers >= 2
        notes = "s_j=λ shared by every driver (design §8.1 baseline)"
    elif kind == SCHEDULE_SMOOTHSTEP:
        suggested = MODE_SCHEDULED_1D
        smoke = True
        notes = (
            "s=3u²−2u³ with u=clip(λ,0,1) on window (0,1); endpoints preserved; "
            "custom-list compilation is todo 19 (never produced here)"
        )
    else:
        suggested = MODE_SCHEDULED_1D
        smoke = True
        summary = ", ".join(
            f"{cid}:"
            + ("linear" if window is None else f"{window[0]}-{window[1]}")
            for cid, window in driver_windows
        )
        notes = (
            f"windowed smoothstep on the two explained event groups "
            f"(design §8.1 {kind}); per-driver windows: {summary}"
        )
    return ScheduleSpec(
        schedule_id=schedule_id,
        kind=kind,
        n_points=len(lam),
        depth=0,
        lambda_values=lam,
        s_by_driver=s_by_driver,
        driver_windows=tuple(driver_windows),
        suggested_mode=suggested,
        requires_backend_smoke=smoke,
        notes=notes,
    )


def _partition_by_budget(
    proposed: Sequence[ScheduleSpec], policy: SchedulePolicy
) -> tuple[tuple[ScheduleSpec, ...], tuple[PrunedCandidate, ...]]:
    """Split proposed schedules into within-budget active vs ``PRUNED_BY_BUDGET``.

    First-version accounting (design §8.1/§8.3): at most ``schedule_budget``
    schedules per candidate, and the projected total
    ``n_active × direction_budget`` must not exceed ``max_total_candidates.scan``
    (each schedule will later pair with up to ``direction_budget`` directions —
    direction *choice* is todo 15; assembly/mode share the same global cap).
    """
    active: list[ScheduleSpec] = []
    pruned: list[PrunedCandidate] = []
    per_schedule_slots = max(1, policy.direction_budget)
    for spec in proposed:
        if len(active) >= policy.schedule_budget:
            pruned.append(
                _pruned_record(
                    spec,
                    policy,
                    reason=(
                        f"schedule {len(active)} already at schedule_budget "
                        f"{policy.schedule_budget}"
                    ),
                )
            )
            continue
        projected = (len(active) + 1) * per_schedule_slots
        if projected > policy.max_total_candidates_scan:
            pruned.append(
                _pruned_record(
                    spec,
                    policy,
                    reason=(
                        f"projected total {projected} = "
                        f"(schedules {len(active) + 1}) × "
                        f"(direction_budget {per_schedule_slots}) exceeds "
                        f"max_total_candidates.scan "
                        f"{policy.max_total_candidates_scan}"
                    ),
                )
            )
            continue
        active.append(spec)
    return tuple(active), tuple(pruned)


def _pruned_record(
    spec: ScheduleSpec, policy: SchedulePolicy, *, reason: str
) -> PrunedCandidate:
    """Build the traceable ``PRUNED_BY_BUDGET`` record of one schedule."""
    return PrunedCandidate(
        pruned_id=f"pruned:{spec.schedule_id}",
        code=CODE_PRUNED_BY_BUDGET,
        schedule_id=spec.schedule_id,
        kind=spec.kind,
        n_points=spec.n_points,
        lambda_values=spec.lambda_values,
        reason=reason,
        budget=policy,
    )


__all__ = [
    "CODE_DIFFERENCE_COORDINATE_FORBIDDEN",
    "CODE_DRIVER_DUPLICATE",
    "CODE_DRIVER_ID_UNRESOLVED",
    "CODE_DRIVER_KIND_INVALID",
    "CODE_DRIVER_ROLE_MISMATCH",
    "CODE_H_DOUBLE_DISTANCE_COLLAPSED",
    "CODE_PRUNED_BY_BUDGET",
    "CODE_SMOKE_NOT_PRODUCED_HERE",
    "CandidateWithFeasibility",
    "DEFAULT_MAX_REFINEMENT_COUNT",
    "DEFAULT_REFINEMENT_RULE",
    "EventGroup",
    "INTEGRITY_VIOLATION_PREFIX",
    "LAMBDA_DECIMALS",
    "PrunedCandidate",
    "REFINEMENT_KIND_DOUBLING",
    "REFINEMENT_KIND_LOCAL_INTERVAL",
    "REFINEMENT_KINDS",
    "RefinementOption",
    "RefinementRule",
    "RefinementTree",
    "SCHEDULE_EVENT_A_EARLY",
    "SCHEDULE_EVENT_A_LATE",
    "SCHEDULE_KIND_ORDER",
    "SCHEDULE_LINEAR",
    "SCHEDULE_SMOOTHSTEP",
    "SCHEMA_SCHEDULES",
    "SchedulePolicy",
    "ScheduleSpec",
    "ScheduledCandidate",
    "WINDOW_EARLY",
    "WINDOW_FULL",
    "WINDOW_LATE",
    "assert_schedule_integrity",
    "build_schedules",
    "double_lambda_grid",
    "policy_from_config",
    "refine_lambda_interval",
    "refinement_options",
    "refinement_tree",
    "s_value_for_window",
    "smoothstep_s",
    "uniform_lambda_grid",
]
