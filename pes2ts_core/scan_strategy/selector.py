"""Selector core for the graph-theoretic PES scan-strategy pipeline (todo 17).

Pure orchestration layer: integrates the Wave-2 modules into one pipeline and
emits a sealed ``g1_strategy_proposal_v1`` document via
:func:`pes2ts_core.scan_strategy.contracts_v2.make_strategy_proposal`.

Pipeline (design §5.1/§6.2/§8.3/§10.1):

1. P0 chain — ``graph_rebuild`` inputs already supplied as an
   ``EndpointGraphBundle``; this module builds the reaction edit graph,
   endpoint context and typed event coupling graph from it.
2. Registry routing — ``registry.route_strategies`` (typed refusals
   ``SPECIAL_DOMAIN`` etc. propagate as whole-case blocking reasons).
3. Coordinate pool — ``coordinate_pool.build_coordinate_pool``.
4. Geometry feasibility — ``geometry_feasibility.assess_geometry_feasibility``
   per driver-set candidate × start endpoint.
5. Schedules — ``schedules.build_schedules`` on every candidate.
6. Direction/assembly — ``direction_assembly.resolve_direction_assembly``.
7. Capabilities — ``capabilities.capability_check`` per mode; unprobed modes
   are recorded ``unknown``/``fail`` and never silently enabled.
8. Ranking — six-layer tie-break (§8.3) + stable residual lexicographic order.
9. Deduplication then config budgets; over-budget candidates stay in the
   record with ``PRUNED_BY_BUDGET``.
10. Sealed proposal — ``execution_eligible`` is always ``false`` (the
    contracts factory enforces it); per-candidate ``failure_reasons`` stay
    separate from whole-case ``blocking_reasons``.

Also exports :func:`all_release_gates_pass`: false when the runnable
subgraph is empty; true only when every fallback reference named by a
candidate exists in the strategy registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pes2ts_core.g1.endpoint_context import EndpointElectronic, build_endpoint_context
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.event_coupling import (
    EventCouplingGraph,
    aromatic_regions_from_bundle,
    build_event_coupling_graph,
)
from pes2ts_core.g1.reaction_edit_graph import ReactionEditGraph, build_reaction_edit_graph
from pes2ts_core.scan_strategy.capabilities import (
    EffectiveCapability,
    capability_check,
    effective_capability,
)
from pes2ts_core.scan_strategy.contracts_v2 import (
    MODE_COUPLED_1D,
    MODE_SCHEDULED_1D,
    MODE_SINGLE_1D,
    make_strategy_proposal,
)
from pes2ts_core.scan_strategy.coordinate_pool import (
    CoordinatePool,
    CoordinateRecord,
    DriverSetCandidate,
    EndpointMaterials,
    build_coordinate_pool,
)
from pes2ts_core.scan_strategy.direction_assembly import (
    ComponentMaterial,
    DirectionAssemblyInputs,
    DriverBondSpec,
    DriverSetSpec,
    EndpointProfile,
    SideMaterial,
    fb_counts_from_edit_graph,
    resolve_direction_assembly,
)
from pes2ts_core.scan_strategy.geometry_feasibility import (
    assess_geometry_feasibility,
    policy_from_config as feasibility_policy_from_config,
)
from pes2ts_core.scan_strategy.registry import (
    STRATEGY_REGISTRY,
    route_strategies,
)
from pes2ts_core.scan_strategy.schedules import (
    CandidateWithFeasibility,
    PrunedCandidate,
    ScheduleSpec,
    ScheduledCandidate,
    build_schedules,
    uniform_lambda_grid,
)

__all__ = [
    "RankInputs",
    "StrategyProposal",
    "all_release_gates_pass",
    "propose_strategies",
    "rank_rank_inputs",
]

# allow: SIZE_OK — plan-named todo-17 single module (design §5.1 pipeline +
# §6.2/§8.3 rank/dedup/budget + §10.1 proposal projection + release gate);
# precedent: registry.py (todo 11) / geometry_feasibility.py (todo 13).

#: Sealed proposal document (``g1_strategy_proposal_v1`` payload).
StrategyProposal = dict[str, Any]

#: Selector pipeline trace rule (proposal ``rule_trace`` vocabulary).
R_SELECTOR_PIPELINE_V1: str = "R_SELECTOR_PIPELINE_V1"

#: Case-level reason emitted on every proposal built from endpoint hypotheses.
REASON_ENDPOINT_HYPOTHESIS: str = "PROPOSED_FROM_ENDPOINT_HYPOTHESIS"

#: Default capability triple resolved against the shipped registry.
_DEFAULT_ENGINE_ENTRY: str = "orca-acp-engine"
_DEFAULT_ADAPTER_ENTRY: str = "acp-adapter-v0"
_DEFAULT_DEPLOYMENT_ENTRY: str = "deployment-baseline"

#: Capability-status reliability rank (lower = more reliable).
_CAPABILITY_RANK: dict[str, int] = {"pass": 0, "unknown": 1, "fail": 2}

#: Schedule ids used when a candidate has no generated schedule rows.
_FALLBACK_SCHEDULE_ID: str = "sched:none"


# ---------------------------------------------------------------------------
# Ranking surface (design §6.2/§8.3) — pure, unit-testable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RankInputs:
    """Ranking projection of one assembled candidate view.

    Six decision layers in §8.3 order, then a stable residual tie-break on
    ``(sorted driver map tuples, kinds)`` — a deterministic total order that
    never consults map magnitude as chemical quality.
    """

    has_uncovered_event: bool
    geometry_ok: bool
    assembly_ready: bool
    capability_status: str
    constraint_rank: int
    n_follow_along: int
    cost: int
    driver_maps_key: tuple[tuple[int, ...], ...] = ()
    kinds_key: tuple[str, ...] = ()

    def ranking_key(self) -> tuple[Any, ...]:
        """Return the sort key; lower sorts first (better candidate)."""
        cap_rank = _CAPABILITY_RANK.get(self.capability_status, len(_CAPABILITY_RANK))
        return (
            1 if self.has_uncovered_event else 0,
            0 if self.geometry_ok else 1,
            (0 if self.assembly_ready else 1, cap_rank),
            self.constraint_rank,
            self.n_follow_along,
            self.cost,
            self.driver_maps_key,
            self.kinds_key,
        )


def rank_rank_inputs(views: Sequence[RankInputs]) -> list[RankInputs]:
    """Return *views* ordered best-first by :meth:`RankInputs.ranking_key`."""
    return sorted(views, key=lambda view: view.ranking_key())


def all_release_gates_pass(proposal: Mapping[str, Any]) -> bool:
    """Release-gate verdict for one StrategyProposal document (design §5.1).

    False when the runnable subgraph is empty (no candidates, or every
    candidate carries ``PRUNED_BY_BUDGET``).  True only when every fallback
    reference named by a candidate exists in the strategy registry.
    """
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    runnable = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and "PRUNED_BY_BUDGET"
        not in {
            str(reason.get("code"))
            for reason in candidate.get("failure_reasons") or ()
            if isinstance(reason, Mapping)
        }
    ]
    if not runnable:
        return False
    known = set(STRATEGY_REGISTRY)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return False
        for reference in candidate.get("fallback_ids") or ():
            if reference not in known:
                return False
    return True


# ---------------------------------------------------------------------------
# Small typed helpers.
# ---------------------------------------------------------------------------
def _reason(code: str, detail: str = "") -> dict[str, Any]:
    """One contract-shaped typed reason (``code`` + optional ``detail``)."""
    return {"code": code, "detail": detail}


def _int_policy_key(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"POLICY_INVALID: scan_strategy.{key} must be an integer")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise ValueError(
                f"POLICY_INVALID: scan_strategy.{key} must be an integer"
            ) from exc
    raise ValueError(f"POLICY_INVALID: scan_strategy.{key} must be an integer")


def _scan_budget(config: Mapping[str, Any] | None) -> int:
    """Return ``scan_strategy.max_total_candidates.scan`` (default 6)."""
    section: Mapping[str, Any] = {}
    if config is not None:
        raw = config.get("scan_strategy")
        if isinstance(raw, Mapping):
            section = raw
    caps = section.get("max_total_candidates")
    if not isinstance(caps, Mapping):
        return 6
    return _int_policy_key(caps, "scan", 6)


def _scan_policy(config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if config is None:
        return None
    raw = config.get("scan_strategy")
    return raw if isinstance(raw, Mapping) else None


def _normalize_materials(
    materials: EndpointMaterials | Mapping[str, Any] | None,
) -> EndpointMaterials | None:
    """Normalize whitelist materials input into typed endpoint coordinates."""
    if materials is None or isinstance(materials, EndpointMaterials):
        return materials
    if not isinstance(materials, Mapping):
        raise ValueError(
            "MATERIALS_SCHEMA_INVALID: expected EndpointMaterials, a mapping "
            "with r/p or r_coordinates/p_coordinates blocks, or None"
        )

    def _side_block(*keys: str) -> Mapping[int, tuple[float, float, float]]:
        for key in keys:
            raw = materials.get(key)
            if isinstance(raw, Mapping):
                parsed: dict[int, tuple[float, float, float]] = {}
                for map_key, value in raw.items():
                    map_id = int(map_key)
                    if isinstance(value, Mapping):
                        value = value.get("coordinates")
                    if not isinstance(value, list | tuple) or len(value) != 3:
                        raise ValueError(
                            "MATERIALS_SCHEMA_INVALID: "
                            f"{key}[{map_id}] must be a length-3 coordinate"
                        )
                    parsed[map_id] = (
                        float(value[0]), float(value[1]), float(value[2])
                    )
                return parsed
        return {}

    r_coordinates = _side_block("r_coordinates", "r")
    p_coordinates = _side_block("p_coordinates", "p")
    if not r_coordinates and not p_coordinates:
        raise ValueError(
            "MATERIALS_SCHEMA_INVALID: no r/p coordinate blocks found"
        )
    return EndpointMaterials(
        r_coordinates=r_coordinates, p_coordinates=p_coordinates
    )


def _electronic_from_materials(
    materials: EndpointMaterials | Mapping[str, Any] | None,
) -> dict[str, EndpointElectronic] | None:
    """Extract endpoint electronic states when the materials mapping carries them."""
    if not isinstance(materials, Mapping):
        return None
    raw = materials.get("endpoint_electronic")
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, EndpointElectronic] = {}
    for key, side in (("reactant", "reactant"), ("product", "product")):
        block = raw.get(key)
        if isinstance(block, EndpointElectronic):
            out[side] = block
        elif isinstance(block, Mapping):
            out[side] = EndpointElectronic(
                charge=int(block["charge"]), multiplicity=int(block["multiplicity"])
            )
    return out or None


def _elements_by_map(bundle: EndpointGraphBundle) -> dict[int, str]:
    return {node.map_id: node.element for node in bundle.r_graph.nodes}


def _bonded_pairs(bundle: EndpointGraphBundle) -> frozenset[tuple[int, int]]:
    pairs = {
        (edge.map_a, edge.map_b) for edge in bundle.r_graph.edges
    } | {(edge.map_a, edge.map_b) for edge in bundle.p_graph.edges}
    return frozenset(pairs)


def _edit_atoms(edit_graph: ReactionEditGraph) -> set[int]:
    atoms: set[int] = set()
    for record in edit_graph.edit_records():
        pair = record.get("pair")
        if isinstance(pair, list) and len(pair) == 2:
            atoms.add(int(pair[0]))
            atoms.add(int(pair[1]))
    return atoms


def _side_material(
    endpoint: str,
    graph: Any,
    coordinates: Mapping[int, tuple[float, float, float]],
    elements: Mapping[int, str],
    edit_atoms: set[int],
) -> SideMaterial:
    components: list[ComponentMaterial] = []
    for component in graph.components:
        comp_coordinates = {
            map_id: coordinates[map_id]
            for map_id in component.map_ids
            if map_id in coordinates
        }
        components.append(
            ComponentMaterial(
                str(component.component_id),
                comp_coordinates,
                {
                    map_id: elements[map_id]
                    for map_id in component.map_ids
                    if map_id in elements
                },
                bool(set(component.map_ids) & edit_atoms),
            )
        )
    return SideMaterial(endpoint, tuple(components))


def _driver_sets(
    pool: CoordinatePool,
) -> tuple[DriverSetSpec, ...]:
    specs: list[DriverSetSpec] = []
    for candidate in pool.candidates():
        bonds = tuple(
            DriverBondSpec(
                atom_maps=pool.coordinate(coordinate_id).atom_maps,
                kind=pool.coordinate(coordinate_id).kind,
                driver_id=coordinate_id,
        )
            for coordinate_id in candidate.driver_coordinate_ids
        )
        specs.append(
            DriverSetSpec(
                driver_set_id=candidate.candidate_id,
                route_strategy_id=candidate.route_strategy_id,
                driver_bonds=bonds,
            )
        )
    return tuple(specs)


def _anchor_reason_string(direction: Any) -> str:
    codes = [reason.code for reason in direction.anchor_reason]
    return "|".join(codes) if codes else "DIRECTION_LAYERED_COMPARISON"


def _monitor_payload(
    record: CoordinateRecord,
    target_test_ids: Sequence[str],
    pool: CoordinatePool,
) -> dict[str, Any]:
    maps = list(record.atom_maps)
    map_token = "+".join(str(m) for m in maps)
    measurement = f"{record.kind}:{map_token}"
    target_test = f"target:{record.kind}:{map_token}"
    for target_id in target_test_ids:
        try:
            target_record = pool.coordinate(target_id)
        except KeyError:
            continue
        target_maps = set(target_record.atom_maps)
        if set(maps) <= target_maps or target_maps <= set(maps):
            target_token = "+".join(str(m) for m in target_record.atom_maps)
            target_test = f"target:{target_record.kind}:{target_token}"
            break
    return {
        "measurement": measurement,
        "target_test": target_test,
        "evidence_available": True,
        "maps": maps,
    }


def _driver_payloads(
    pool: CoordinatePool,
    candidate: DriverSetCandidate,
    schedule: ScheduleSpec | None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for coordinate_id in candidate.driver_coordinate_ids:
        record = pool.coordinate(coordinate_id)
        schedule_values: list[float] | None = None
        if schedule is not None:
            try:
                schedule_values = list(schedule.s_of(coordinate_id))
            except KeyError:
                schedule_values = None
        payloads.append(
            {
                "kind": record.kind,
                "maps": list(record.atom_maps),
                "unit": record.units,
                "schedule_values": schedule_values,
                "index0": None,
            }
        )
    return payloads


def _mode_for_drivers(n_drivers: int) -> str:
    return MODE_SINGLE_1D if n_drivers == 1 else MODE_COUPLED_1D


def _contract_mode(suggested_mode: str, n_drivers: int) -> str:
    """Project a schedule's suggested mode onto the contracts_v2 vocabulary.

    ``g1_strategy_proposal_v1`` requires SINGLE_1D ⇒ exactly one driver and
    COUPLED/SCHEDULED_1D ⇒ 2..3 drivers.  Single-driver custom schedules
    (smoothstep/event windows) therefore project as SINGLE_1D with
    ``schedule_kind`` + per-driver ``schedule_values`` carrying the
    non-linearity (design §8.1); the schedule module's suggested mode is
    preserved in the candidate extensions.
    """
    if n_drivers == 1:
        return MODE_SINGLE_1D
    if suggested_mode in (MODE_COUPLED_1D, MODE_SCHEDULED_1D):
        return suggested_mode
    return MODE_COUPLED_1D


def _dedup_key(
    *,
    driver_identity: tuple[tuple[str, tuple[int, ...]], ...],
    direction: str,
    schedule_kind: str,
    assembly_key: str,
) -> tuple[Any, ...]:
    """Same drivers modulo order + same direction + kind + assembly-equivalent."""
    return (
        frozenset(driver_identity),
        direction,
        schedule_kind,
        assembly_key,
    )


@dataclass(frozen=True, slots=True)
class _AssembledCandidate:
    """One assembled (driver-set × direction × schedule) candidate view."""

    rank: RankInputs
    dedup_key: tuple[Any, ...]
    pool_candidate: DriverSetCandidate
    direction: Any
    schedule: ScheduleSpec | None
    pruned_schedule: PrunedCandidate | None
    scheduled: ScheduledCandidate | None
    n_points: int
    failure_reasons: tuple[dict[str, Any], ...]
    anchor_reason: str
    assembly_key: str
    capability_status: str
    capability_missing: tuple[str, ...]


def _assemble_candidates(
    *,
    pool: CoordinatePool,
    direction_result: Any,
    bundle: EndpointGraphBundle,
    materials: EndpointMaterials | None,
    config: Mapping[str, Any] | None,
    capability: EffectiveCapability,
) -> tuple[list[_AssembledCandidate], list[tuple[str, tuple[str, ...]]]]:
    """Cross pool candidates × direction candidates × schedules into views.

    Returns ``(views, skipped)`` where ``skipped`` lists pool candidates whose
    driver coordinates carry duplicate atom-map ids (contract-illegible under
    ``g1_strategy_proposal_v1``); they are never silently rewritten.
    """
    by_driver_set: dict[str, list[Any]] = {}
    for direction in direction_result.direction_candidates:
        key = direction.driver_set_id or ""
        by_driver_set.setdefault(key, []).append(direction)

    views: list[_AssembledCandidate] = []
    skipped: list[tuple[str, tuple[str, ...]]] = []
    for pool_candidate in pool.candidates():
        directions = by_driver_set.get(pool_candidate.candidate_id)
        if not directions:
            continue
        driver_ids = pool_candidate.driver_coordinate_ids
        bad_drivers = tuple(
            coordinate_id
            for coordinate_id in driver_ids
            if len(pool.coordinate(coordinate_id).atom_maps)
            != len(set(pool.coordinate(coordinate_id).atom_maps))
        )
        if bad_drivers:
            skipped.append((pool_candidate.candidate_id, bad_drivers))
            continue
        driver_kinds = tuple(pool.coordinate(cid).kind for cid in driver_ids)
        driver_maps_key = tuple(
            tuple(sorted(pool.coordinate(cid).atom_maps)) for cid in driver_ids
        )
        driver_identity = tuple(
            (pool.coordinate(cid).kind, tuple(sorted(pool.coordinate(cid).atom_maps)))
            for cid in driver_ids
        )
        for direction in directions:
            feasibility = assess_geometry_feasibility(
                pool_candidate,
                pool,
                bundle,
                materials,
                config if config is not None else {},
                start_endpoint=direction.start_endpoint,
            )
            common_failures: list[dict[str, Any]] = []
            scheduled: ScheduledCandidate | None
            try:
                scheduled = build_schedules(
                    CandidateWithFeasibility(
                        candidate=pool_candidate,
                        pool=pool,
                        feasibility=feasibility,
                        start_endpoint=direction.start_endpoint,
                    ),
                    config if config is not None else {},
                )
            except ValueError as exc:
                # Todo-14 integrity rejects any origin text containing
                # "difference"; todo-12 conformational D legitimately uses
                # geometry_kind="torsion_difference".  Orchestration layer
                # records the typed failure per candidate instead of crashing.
                scheduled = None
                common_failures.append(
                    _reason("SCHEDULE_INTEGRITY", str(exc))
                )
            assembly_id = direction.assembly_id or "asm-not-evaluated"
            assembly_key = f"{assembly_id}:{direction.start_endpoint}"
            if pool_candidate.route_review_required:
                common_failures.append(
                    _reason(
                        "ROUTE_REVIEW_REQUIRED",
                        str(pool_candidate.route_strategy_id or ""),
                    )
                )
            if pool_candidate.has_uncovered_event:
                common_failures.append(
                    _reason(
                        "UNCOVERED_EVENT",
                        str(pool_candidate.non_default_reason or "uncovered event"),
                    )
                )
            for failure in feasibility.failures:
                common_failures.append(
                    _reason(failure.code, failure.detail)
                )
            if feasibility.point_budget_exceeded:
                common_failures.append(
                    _reason(
                        "POINT_BUDGET_EXCEEDED",
                        f"n_points={feasibility.n_points}",
                    )
                )
            if not direction.ready:
                common_failures.append(
                    _reason(
                        "DIRECTION_NOT_READY",
                        "|".join(direction.readiness_blockers),
                    )
                )
            if scheduled is not None:
                for note in scheduled.integrity_notes:
                    common_failures.append(_reason("SCHEDULE_INTEGRITY", note))

            # Active schedules and schedule-level prunes are both recorded;
            # only pruned rows carry PRUNED_BY_BUDGET (§8.3: never silent).
            # When schedule generation raised (integrity false-positive on
            # conformational torsion_difference), one fallback row keeps the
            # candidate visible with its typed failure.
            schedule_rows: list[tuple[ScheduleSpec | None, PrunedCandidate | None]] = (
                [(spec, None) for spec in scheduled.schedules]
                + [(None, record) for record in scheduled.pruned]
                if scheduled is not None
                else [(None, None)]
            )
            baseline_points = feasibility_policy_from_config(
                config if config is not None else {}
            ).point_baseline

            for schedule, pruned in schedule_rows:
                if schedule is not None:
                    mode = _contract_mode(
                        schedule.suggested_mode, len(driver_ids)
                    )
                    n_points = schedule.n_points
                    schedule_kind = schedule.kind
                    requires_smoke = schedule.requires_backend_smoke
                    suggested_mode = schedule.suggested_mode
                elif pruned is not None:
                    mode = _mode_for_drivers(len(driver_ids))
                    n_points = pruned.n_points
                    schedule_kind = pruned.kind
                    requires_smoke = mode != MODE_SINGLE_1D
                    suggested_mode = mode
                else:
                    mode = _mode_for_drivers(len(driver_ids))
                    n_points = (
                        feasibility.n_points
                        if feasibility.n_points is not None
                        else baseline_points
                    )
                    schedule_kind = "linear"
                    requires_smoke = mode != MODE_SINGLE_1D
                    suggested_mode = mode

                check = capability_check(
                    capability, mode, list(driver_kinds), len(driver_kinds)
                )
                failures = list(common_failures)
                if requires_smoke:
                    failures.append(
                        _reason(
                            "BACKEND_SMOKE_REQUIRED",
                            f"mode {mode} requires backend smoke",
                        )
                    )
                if check["status"] == "fail":
                    failures.append(
                        _reason(
                            "BACKEND_CAPABILITY_MISSING",
                            "; ".join(str(item) for item in check.get("missing") or ()),
                        )
                    )
                elif check["status"] == "unknown":
                    failures.append(
                        _reason(
                            "CAPABILITY_UNPROBED",
                            "; ".join(str(item) for item in check.get("missing") or ()),
                        )
                    )
                if pruned is not None:
                    failures.append(
                        _reason(
                            "PRUNED_BY_BUDGET",
                            f"{pruned.schedule_id}:{pruned.reason}",
                        )
                    )

                n_coords = len(driver_ids)
                cost = int(n_points) * n_coords
                rank = RankInputs(
                    has_uncovered_event=bool(pool_candidate.has_uncovered_event),
                    geometry_ok=bool(feasibility.ok),
                    assembly_ready=bool(direction.ready),
                    capability_status=str(check["status"]),
                    constraint_rank=int(pool_candidate.constraint_rank),
                    n_follow_along=sum(
                        1
                        for row in pool_candidate.event_coverage
                        if row.coverage == "coupled_monitor"
                    ),
                    cost=cost,
                    driver_maps_key=driver_maps_key,
                    kinds_key=driver_kinds,
                )
                dedup = _dedup_key(
                    driver_identity=driver_identity,
                    direction=direction.direction,
                    schedule_kind=schedule_kind,
                    assembly_key=assembly_key,
                )
                views.append(
                    _AssembledCandidate(
                        rank=rank,
                        dedup_key=dedup,
                        pool_candidate=pool_candidate,
                        direction=direction,
                        schedule=schedule,
                        pruned_schedule=pruned,
                        scheduled=scheduled,
                        n_points=int(n_points),
                        failure_reasons=tuple(failures),
                        anchor_reason=_anchor_reason_string(direction),
                        assembly_key=assembly_key,
                        capability_status=str(check["status"]),
                        capability_missing=tuple(
                            str(item) for item in check.get("missing") or ()
                        ),
                    )
                )
    return views, skipped


def _candidate_payload(
    view: _AssembledCandidate,
    pool: CoordinatePool,
    capability: EffectiveCapability,
    candidate_id: str,
    route_fallbacks: Sequence[str],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one assembled view into a contracts_v2 proposal candidate."""
    pool_candidate = view.pool_candidate
    driver_ids = pool_candidate.driver_coordinate_ids
    driver_kinds = [pool.coordinate(cid).kind for cid in driver_ids]
    n_drivers = len(driver_ids)
    if view.schedule is not None:
        suggested_mode = view.schedule.suggested_mode
        mode = _contract_mode(suggested_mode, n_drivers)
    else:
        suggested_mode = _mode_for_drivers(n_drivers)
        mode = suggested_mode
    schedule_id = (
        view.schedule.schedule_id
        if view.schedule is not None
        else (
            view.pruned_schedule.schedule_id
            if view.pruned_schedule is not None
            else _FALLBACK_SCHEDULE_ID
        )
    )
    schedule_kind = (
        view.schedule.kind
        if view.schedule is not None
        else (
            view.pruned_schedule.kind
            if view.pruned_schedule is not None
            else "linear"
        )
    )
    lambda_values = (
        list(view.schedule.lambda_values)
        if view.schedule is not None
        else (
            list(view.pruned_schedule.lambda_values)  # type: ignore[union-attr]
            if view.pruned_schedule is not None
            else list(uniform_lambda_grid(max(view.n_points, 2)))
        )
    )
    check = capability_check(
        capability, mode, driver_kinds, len(driver_kinds)
    )
    monitor_ids = pool_candidate.monitor_coordinate_ids
    monitors = [
        _monitor_payload(pool.coordinate(cid), monitor_ids, pool)
        for cid in monitor_ids
    ]
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "mode": mode,
        "start_endpoint": view.direction.start_endpoint,
        "direction": view.direction.direction,
        "assembly_id": view.direction.assembly_id or "asm-not-evaluated",
        "anchor_reason": view.anchor_reason,
        "drivers": _driver_payloads(pool, pool_candidate, view.schedule),
        "monitors": monitors,
        "guards": [],
        "event_coverage": [
            row.to_record() for row in pool_candidate.event_coverage
        ],
        "lambda_values": lambda_values,
        "schedule_id": schedule_id,
        "schedule_kind": schedule_kind,
        "required_capabilities": [mode],
        "capability_check": check,
        "budget": dict(budget),
        "failure_reasons": list(view.failure_reasons),
        "fallback_ids": list(route_fallbacks),
        "expected_cost": view.n_points * len(driver_ids),
        "extensions": {
            "driver_coordinate_ids": list(driver_ids),
            "route_strategy_id": pool_candidate.route_strategy_id,
            "n_points": view.n_points,
            "n_drivers": n_drivers,
            "constraint_rank": pool_candidate.constraint_rank,
            "suggested_mode": suggested_mode,
            "pruned_by_budget": any(
                reason.get("code") == "PRUNED_BY_BUDGET"
                for reason in view.failure_reasons
            ),
            "identity": {
                "drivers": [
                    [kind, list(maps)]
                    for kind, maps in (
                        (
                            pool.coordinate(cid).kind,
                            list(pool.coordinate(cid).atom_maps),
                        )
                        for cid in driver_ids
                    )
                ],
                "direction": view.direction.direction,
                "schedule_kind": schedule_kind,
                "assembly_id": view.assembly_key,
            },
        },
    }
    return payload


def _graph_features(
    edit_graph: ReactionEditGraph,
    context: Any,
    coupling: EventCouplingGraph,
) -> dict[str, Any]:
    coupling_doc = coupling.to_doc()
    context_doc = context.to_doc()
    return {
        "edit_components": [list(comp) for comp in edit_graph.edit_components],
        "connectivity_edit_components": [
            list(comp) for comp in edit_graph.connectivity_edit_components
        ],
        "edit_counts": dict(edit_graph.edit_counts),
        "context_support": {
            "context_radius": getattr(context, "context_radius", None),
            "edits": list(context_doc.get("edits", [])),
        },
        "events": [
            {"event_id": event.event_id, "event_type": event.event_type}
            for event in coupling.events
        ],
        "typed_couplings": {
            "strong_components": [
                list(comp) for comp in coupling.strong_components
            ],
            "weak_links": list(coupling_doc.get("weak_links", [])),
            "hydrogen_events": list(coupling.hydrogen_events),
        },
    }


def _proposal_budget(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-candidate budget object (proposal stage authorizes no compute)."""
    section: Mapping[str, Any] = {}
    if config is not None:
        raw = config.get("scan_strategy")
        if isinstance(raw, Mapping):
            inner = raw.get("proposal_budget")
            if isinstance(inner, Mapping):
                section = inner
    return {
        "max_attempts": _int_policy_key(section, "max_attempts", 1),
        "max_cpu_hours": float(section.get("max_cpu_hours", 0.0) or 0.0),
        "max_wall_seconds": _int_policy_key(section, "max_wall_seconds", 0),
    }


def propose_strategies(
    bundle: EndpointGraphBundle,
    materials: EndpointMaterials | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    reaction_id: str | None = None,
    case_id: str | None = None,
    split: str = "unassigned",
    source_case_sha256: str | None = None,
    capability: EffectiveCapability | None = None,
) -> StrategyProposal:
    """Run the selector pipeline and return one sealed StrategyProposal.

    ``execution_eligible`` is always ``false`` — the contracts factory
    enforces it; freeze/execution eligibility is decided later (todos 23+)
    after review and backend smoke.  Route-level typed refusals
    (``SPECIAL_DOMAIN``, ``NO_REACTION_CHANGE``, conservation failures)
    propagate as whole-case ``blocking_reasons``; per-candidate problems stay
    in each candidate's ``failure_reasons``.
    """
    materials_norm = _normalize_materials(materials)
    electronic = _electronic_from_materials(materials)
    content_sha = str(bundle.content_sha256)
    resolved_reaction_id = reaction_id or f"rxn-{content_sha[:16]}"
    resolved_case_id = case_id or f"case-{content_sha[:16]}"
    if source_case_sha256 is not None:
        resolved_source_sha = source_case_sha256
    else:
        provenance_sha = getattr(bundle, "endpoint_materials_sha256", None)
        resolved_source_sha = (
            str(provenance_sha) if provenance_sha else content_sha
        )

    aromatic = aromatic_regions_from_bundle(bundle)
    edit_graph = build_reaction_edit_graph(
        bundle, aromatic_regions=aromatic if aromatic else None
    )
    context = build_endpoint_context(
        bundle,
        edit_graph,
        r_coordinates=(
            None if materials_norm is None else dict(materials_norm.r_coordinates)
        ),
        p_coordinates=(
            None if materials_norm is None else dict(materials_norm.p_coordinates)
        ),
        endpoint_electronic=electronic,
    )
    coupling = build_event_coupling_graph(bundle, edit_graph, context)

    route = route_strategies(
        bundle=bundle,
        edit_graph=edit_graph,
        context=context,
        coupling=coupling,
        reaction_id=resolved_reaction_id,
        policy=_scan_policy(config),
    )

    blocking_reasons: list[dict[str, Any]] = []
    reasons: list[dict[str, Any]] = [
        _reason(
            REASON_ENDPOINT_HYPOTHESIS,
            "graph selector core; execution_eligible=false (factory-enforced)",
        )
    ]

    if route.outcome in {"typed_rejection", "special_domain_exit"}:
        if route.rejection_code:
            blocking_reasons.append(
                _reason(route.rejection_code, route.rejection_reason or "")
            )
        else:
            blocking_reasons.append(_reason(route.outcome, route.rejection_reason or ""))
        family = str(route.rejection_code or route.outcome)
        status = "rejected"
        candidates_payload: list[dict[str, Any]] = []
        gate_extension: dict[str, Any] = {
            "outcome": route.outcome,
            "n_candidates": 0,
            "n_pruned_by_budget": 0,
        }
    else:
        family_candidates = list(route.strategy_candidates)
        family = (
            str(family_candidates[0].strategy_id)
            if family_candidates
            else "UNROUTED"
        )
        if route.outcome == "needs_review":
            reasons.append(
                _reason(
                    "ROUTE_NEEDS_REVIEW",
                    "|".join(c.strategy_id for c in family_candidates),
                )
            )
        if capability is None:
            capability = effective_capability(
                _DEFAULT_ENGINE_ENTRY,
                _DEFAULT_ADAPTER_ENTRY,
                _DEFAULT_DEPLOYMENT_ENTRY,
            )

        pool = build_coordinate_pool(
            route,
            edit_graph,
            context,
            coupling,
            bundle,
            materials_norm,
        )
        elements = _elements_by_map(bundle)
        edit_atom_set = _edit_atoms(edit_graph)
        direction_inputs = DirectionAssemblyInputs(
            reaction_id=resolved_reaction_id,
            fb_counts=fb_counts_from_edit_graph(edit_graph),
            r_profile=EndpointProfile(
                "R",
                electronic_review_complete=False,
                geometry_review_complete=False,
            ),
            p_profile=EndpointProfile(
                "P",
                electronic_review_complete=False,
                geometry_review_complete=False,
            ),
            driver_sets=_driver_sets(pool),
            r_side=(
                None
                if materials_norm is None
                else _side_material(
                    "R",
                    bundle.r_graph,
                    materials_norm.r_coordinates,
                    elements,
                    edit_atom_set,
                )
            ),
            p_side=(
                None
                if materials_norm is None
                else _side_material(
                    "P",
                    bundle.p_graph,
                    materials_norm.p_coordinates,
                    elements,
                    edit_atom_set,
                )
            ),
            bonded_pairs=_bonded_pairs(bundle),
            elements=elements,
        )
        direction_result = resolve_direction_assembly(
            direction_inputs, config=config
        )
        views, skipped_drivers = _assemble_candidates(
            pool=pool,
            direction_result=direction_result,
            bundle=bundle,
            materials=materials_norm,
            config=config,
            capability=capability,
        )
        for candidate_id, bad_ids in skipped_drivers:
            reasons.append(
                _reason(
                    "COORDINATE_DRIVER_MAPS_NOT_UNIQUE",
                    f"{candidate_id}: {','.join(bad_ids)}",
                )
            )

        # Rank → deduplicate → enforce the config scan budget.
        ranked = rank_rank_inputs([view.rank for view in views])
        rank_to_view = {id(view.rank): view for view in views}
        ordered = [rank_to_view[id(key)] for key in ranked]
        seen: set[tuple[Any, ...]] = set()
        deduped: list[_AssembledCandidate] = []
        for view in ordered:
            if view.dedup_key in seen:
                continue
            seen.add(view.dedup_key)
            deduped.append(view)

        budget_scan = _scan_budget(config)
        route_fallbacks_by_strategy = {
            candidate.strategy_id: tuple(candidate.escalation_fallback)
            for candidate in family_candidates
        }
        candidate_budget = _proposal_budget(config)
        candidates_payload = []
        n_pruned = 0
        for index, view in enumerate(deduped):
            candidate_id = f"cand-{index:04d}"
            route_strategy = view.pool_candidate.route_strategy_id
            fallbacks = route_fallbacks_by_strategy.get(route_strategy or "", ())
            if index >= budget_scan:
                n_pruned += 1
                payload = _candidate_payload(
                    view, pool, capability, candidate_id, fallbacks, candidate_budget
                )
                payload["failure_reasons"] = [
                    *view.failure_reasons,
                    _reason(
                        "PRUNED_BY_BUDGET",
                        f"max_total_candidates.scan={budget_scan}",
                    ),
                ]
                payload["extensions"]["pruned_by_budget"] = True
            else:
                payload = _candidate_payload(
                    view, pool, capability, candidate_id, fallbacks, candidate_budget
                )
            candidates_payload.append(payload)

        if any(
            candidate["capability_check"]["status"] == "unknown"
            for candidate in candidates_payload
        ):
            reasons.append(
                _reason(
                    "CAPABILITY_UNPROBED",
                    "unprobed modes recorded unknown; never silently enabled",
                )
            )
        if n_pruned:
            reasons.append(
                _reason(
                    "CANDIDATES_PRUNED_BY_BUDGET",
                    f"{n_pruned} candidate(s) pruned by scan budget {budget_scan}",
                )
            )
        if not candidates_payload:
            reasons.append(
                _reason(
                    "NO_RUNNABLE_CANDIDATES",
                    "route produced no coordinate candidates",
                )
            )
            status = "needs_review"
        else:
            status = "proposed" if route.outcome == "executable_candidate_set" else "needs_review"
        gate_extension = {
            "outcome": route.outcome,
            "n_candidates": len(candidates_payload),
            "n_pruned_by_budget": n_pruned,
            "capability_engine_entry_id": capability.engine_entry_id,
            "capability_adapter_entry_id": capability.adapter_entry_id,
            "capability_deployment_entry_id": capability.deployment_entry_id,
            "enabled_modes": sorted(capability.enabled_modes),
        }

    graph_input = {
        "endpoint_graph_sha256": content_sha,
        "normalization_version": str(bundle.normalization_version),
        "mapping_equivalence": {
            "map_ids": list(bundle.conservation.map_ids),
            "element_conserved": bool(bundle.conservation.element_conserved),
            "isotope_conserved": bool(bundle.conservation.isotope_conserved),
            "basis": "whitelisted_endpoint_graph_rebuild",
        },
    }
    rule_trace = [{"rule_id": rid} for rid in route.rule_trace]
    rule_trace.append({"rule_id": R_SELECTOR_PIPELINE_V1})

    return make_strategy_proposal(
        resolved_reaction_id,
        status,
        reaction_id=resolved_reaction_id,
        case_id=resolved_case_id,
        split=split,
        source_case_sha256=resolved_source_sha,
        graph_input=graph_input,
        graph_features=_graph_features(edit_graph, context, coupling),
        family=family,
        motif_tags=sorted(route.labels),
        rule_trace=rule_trace,
        epistemic_status="endpoint_hypothesis",
        candidates=candidates_payload,
        reasons=reasons,
        blocking_reasons=blocking_reasons,
        extensions={"selector": gate_extension},
    )
