"""Tests for the todo-14 one-dimensional schedules and finite candidate tree."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_context import build_endpoint_context
from pes2ts_core.g1.endpoint_graph import (
    AtomNode,
    BondEdge,
    ConservationReport,
    EndpointGraphBundle,
    GeometryRef,
    GraphComponent,
    SideGraph,
)
from pes2ts_core.g1.event_coupling import (
    aromatic_regions_from_bundle,
    build_event_coupling_graph,
)
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.contracts_v2 import (
    MODE_COUPLED_1D,
    MODE_SCHEDULED_1D,
    MODE_SINGLE_1D,
    SCHEDULE_KINDS as CONTRACT_SCHEDULE_KINDS,
)
from pes2ts_core.generation.planning.coordinate_pool import (
    GEOM_HH,
    GEOM_PARTNER,
    KIND_A,
    KIND_B,
    PARTNER_ACCEPTOR,
    PARTNER_DONOR,
    ROLE_DRIVER,
    SCHEMA_COORDINATE_POOL,
    UNIT_ANGSTROM,
    UNIT_DEGREE,
    CoordinatePool,
    CoordinateRecord,
    DriverSetCandidate,
    EndpointMaterials,
    OriginRecord,
    build_coordinate_pool,
)
from pes2ts_core.generation.planning.geometry_feasibility import FeasibilityReport
from pes2ts_core.generation.planning.registry import route_strategies
from pes2ts_core.generation.planning.schedules import (
    CODE_DIFFERENCE_COORDINATE_FORBIDDEN,
    CODE_DRIVER_ID_UNRESOLVED,
    CODE_H_DOUBLE_DISTANCE_COLLAPSED,
    CODE_PRUNED_BY_BUDGET,
    INTEGRITY_VIOLATION_PREFIX,
    REFINEMENT_KIND_DOUBLING,
    REFINEMENT_KIND_LOCAL_INTERVAL,
    SCHEDULE_EVENT_A_EARLY,
    SCHEDULE_EVENT_A_LATE,
    SCHEDULE_KIND_ORDER,
    SCHEDULE_LINEAR,
    SCHEDULE_SMOOTHSTEP,
    CandidateWithFeasibility,
    RefinementRule,
    ScheduleSpec,
    assert_schedule_integrity,
    build_schedules,
    double_lambda_grid,
    policy_from_config,
    refine_lambda_interval,
    refinement_options,
    refinement_tree,
    smoothstep_s,
    uniform_lambda_grid,
)

FORBIDDEN = {key.lower() for key in FORBIDDEN_TRUTH_KEYS} | {
    key.lower() for key in FORBIDDEN_EXPORT_KEYS
} | {"endpoint_match", "orientation", "irc_evidence"}


# ---------------------------------------------------------------------------
# Hand-built fixtures (idiom shared with test_geometry_feasibility.py).
# ---------------------------------------------------------------------------
def _node(map_id: int, element: str) -> AtomNode:
    return AtomNode(
        map_id=map_id,
        element=element,
        isotope=0,
        formal_charge=0,
        radical_electrons=0,
        explicit_H_neighbors=0,
        aromatic=False,
        stereo="CHI_UNSPECIFIED",
    )


def _edge(a: int, b: int, order: float = 1.0) -> BondEdge:
    lo, hi = (a, b) if a <= b else (b, a)
    return BondEdge(
        map_a=lo,
        map_b=hi,
        connection_type="SINGLE" if order == 1 else "DOUBLE",
        bond_order=float(order),
        aromatic=False,
        stereo="STEREONONE",
    )


def _components(map_ids: list[int], edges: list[tuple[int, int]]) -> tuple[GraphComponent, ...]:
    parent = {m: m for m in map_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for m in map_ids:
        groups.setdefault(find(m), []).append(m)
    return tuple(
        GraphComponent(component_id=i, map_ids=tuple(sorted(members)))
        for i, members in enumerate(sorted(groups.values()))
    )


def _hand_bundle(
    r_specs: list[tuple[int, int, float]],
    p_specs: list[tuple[int, int, float]],
    *,
    elements: Mapping[int, str] | None = None,
) -> EndpointGraphBundle:
    elements = dict(elements or {})
    maps = sorted({m for spec in r_specs + p_specs for m in spec[:2]} | set(elements))

    def side(specs: list[tuple[int, int, float]]) -> SideGraph:
        nodes = tuple(_node(m, elements.get(m, "C")) for m in maps)
        edges = tuple(
            sorted((_edge(a, b, order) for a, b, order in specs), key=lambda e: (e.map_a, e.map_b))
        )
        comps = _components(maps, [(e.map_a, e.map_b) for e in edges])
        return SideGraph(nodes=nodes, edges=edges, components=comps)

    r_graph = side(r_specs)
    p_graph = side(p_specs)
    geometry = GeometryRef(unit="angstrom", coordinates_sha256="0" * 64, n_points=len(maps))
    conservation = ConservationReport(
        map_ids=tuple(maps),
        element_conserved=True,
        isotope_conserved=True,
        explicit_h_count_r=0,
        explicit_h_count_p=0,
        n_atoms_r=len(maps),
        n_atoms_p=len(maps),
        n_components_r=len(r_graph.components),
        n_components_p=len(p_graph.components),
    )
    return EndpointGraphBundle(
        schema_version="test_handbuilt",
        aromatic_model="rdkit_default_unkekulized",
        rdkit_version="test",
        normalization_version="test",
        r_graph=r_graph,
        p_graph=p_graph,
        r_geometry=geometry,
        p_geometry=geometry,
        conservation=conservation,
        content_sha256="0" * 64,
    )


def _p0(
    r_specs: list[tuple[int, int, float]],
    p_specs: list[tuple[int, int, float]],
    *,
    elements: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    bundle = _hand_bundle(r_specs, p_specs, elements=elements)
    aromatic = aromatic_regions_from_bundle(bundle)
    edit_graph = build_reaction_edit_graph(
        bundle, aromatic_regions=aromatic if aromatic else None
    )
    context = build_endpoint_context(bundle, edit_graph)
    coupling = build_event_coupling_graph(bundle, edit_graph, context)
    route = route_strategies(
        bundle=bundle, edit_graph=edit_graph, context=context, coupling=coupling
    )
    return {
        "bundle": bundle,
        "edit_graph": edit_graph,
        "context": context,
        "coupling": coupling,
        "route": route,
    }


def _h_transfer_pure() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 4, 1)],
        [(1, 2, 1), (1, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H"},
    )


def _pool_from_p0(p0: Mapping[str, Any], materials: Any = None) -> CoordinatePool:
    return build_coordinate_pool(
        p0["route"],
        p0["edit_graph"],
        p0["context"],
        p0["coupling"],
        p0["bundle"],
        materials,
    )


def _coord(
    coordinate_id: str,
    kind: str,
    atom_maps: tuple[int, ...],
    *,
    role: str = ROLE_DRIVER,
    units: str | None = None,
    origin: OriginRecord | None = None,
    event_ids: tuple[str, ...] = (),
) -> CoordinateRecord:
    if units is None:
        units = UNIT_ANGSTROM if kind == KIND_B else UNIT_DEGREE
    if origin is None:
        origin = OriginRecord(source_kind="edit")
    return CoordinateRecord(
        coordinate_id=coordinate_id,
        role=role,
        kind=kind,
        atom_maps=atom_maps,
        units=units,
        event_ids=event_ids,
        origin=origin,
    )


def _candidate(
    driver_ids: tuple[str, ...], candidate_id: str | None = None
) -> DriverSetCandidate:
    cid = candidate_id or "cand:" + "+".join(driver_ids)
    return DriverSetCandidate(
        candidate_id=cid,
        driver_coordinate_ids=driver_ids,
        monitor_coordinate_ids=(),
        guard_coordinate_ids=(),
        target_test_coordinate_ids=(),
        event_coverage=(),
        has_uncovered_event=False,
        default_automatic=True,
        non_default_reason=None,
        route_strategy_id=None,
        route_review_required=False,
        n_drivers=len(driver_ids),
        constraint_rank=len(driver_ids),
        edits_monitored=True,
    )


def _pool(
    records: tuple[CoordinateRecord, ...], candidate: DriverSetCandidate
) -> CoordinatePool:
    return CoordinatePool(
        schema_version=SCHEMA_COORDINATE_POOL,
        reaction_id=None,
        coordinates=records,
        driver_candidates=(candidate,),
        all_edits_monitored=True,
    )


def _feasibility(n_points: int | None = 9, *, ok: bool = True) -> FeasibilityReport:
    return FeasibilityReport(
        schema_version="g1_geometry_feasibility_v1",
        ok=ok,
        candidate_id="cand",
        start_endpoint="R",
        checks=(),
        failures=(),
        n_points=n_points,
        n_points_required=None,
        n_points_baseline=9,
        point_budget_exceeded=False,
        total_variation_by_coord=(),
        jacobian_rank_by_endpoint=(),
        jacobian_condition_max=None,
    )


def _config(**scan_overrides: Any) -> dict[str, Any]:
    scan: dict[str, Any] = {
        "schedule_budget": 3,
        "direction_budget": 2,
        "assembly_candidate_budget": 2,
        "max_total_candidates": {"scan": 6, "path_neb": 1},
        "point_limits": {"baseline": 9, "max": 101},
    }
    scan.update(scan_overrides)
    return {"scan_strategy": scan}


def _two_group_records() -> tuple[CoordinateRecord, ...]:
    """Donor-H + acceptor-H B drivers (EV_H) + a formed-bond driver (EV_B)."""
    return (
        _coord(
            "coord-0001",
            KIND_B,
            (2, 4),
            event_ids=("EV_H",),
            origin=OriginRecord(
                source_kind="hydrogen_event",
                event_id="EV_H",
                hydrogen_map=4,
                partner_map=2,
                partner_role=PARTNER_DONOR,
                geometry_kind=GEOM_PARTNER,
                r_value=1.09,
                p_value=2.40,
            ),
        ),
        _coord(
            "coord-0002",
            KIND_B,
            (1, 4),
            event_ids=("EV_H",),
            origin=OriginRecord(
                source_kind="hydrogen_event",
                event_id="EV_H",
                hydrogen_map=4,
                partner_map=1,
                partner_role=PARTNER_ACCEPTOR,
                geometry_kind=GEOM_PARTNER,
                r_value=2.40,
                p_value=1.09,
            ),
        ),
        _coord(
            "coord-0003",
            KIND_B,
            (5, 6),
            event_ids=("EV_B",),
            origin=OriginRecord(
                source_kind="edit",
                edit_pair=(5, 6),
                edit_kind="formed",
                r_value=2.50,
                p_value=1.50,
            ),
        ),
    )


def _two_group_candidate() -> tuple[CoordinatePool, DriverSetCandidate]:
    records = _two_group_records()
    candidate = _candidate(("coord-0001", "coord-0002", "coord-0003"))
    return _pool(records, candidate), candidate


def _one_group_candidate() -> tuple[CoordinatePool, DriverSetCandidate]:
    records = _two_group_records()
    candidate = _candidate(("coord-0001",))
    return _pool(records, candidate), candidate


def _scan_forbidden(obj: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN:
                hits.append(key)
            hits.extend(_scan_forbidden(value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            hits.extend(_scan_forbidden(item))
    return hits


def _schedule_by_kind(scheduled: Any, kind: str) -> ScheduleSpec:
    for spec in scheduled.schedules:
        if spec.kind == kind:
            return spec
    raise AssertionError(f"missing schedule kind {kind}: {scheduled.schedules}")


# ---------------------------------------------------------------------------
# 1. Schedule-kind math.
# ---------------------------------------------------------------------------
def test_linear_schedule_math_and_shared_lambda() -> None:
    """Given a 2-driver candidate, then linear s=λ and one shared λ list."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    linear = _schedule_by_kind(scheduled, SCHEDULE_LINEAR)
    assert linear.lambda_values == (0.0, 0.25, 0.5, 0.75, 1.0)
    for _cid, values in linear.s_by_driver:
        assert values == linear.lambda_values
    assert linear.s_of("coord-0001") == linear.lambda_values
    assert linear.s_of("coord-0002") == linear.lambda_values
    assert linear.s_of("coord-0003") == linear.lambda_values
    assert all(window is None for _cid, window in linear.driver_windows)


def test_smoothstep_schedule_endpoints_midpoint_and_monotone() -> None:
    """Given the smoothstep kind, then s(0)=0, s(0.5)=0.5, s(1)=1, monotone."""
    assert smoothstep_s(0.0, (0.0, 1.0)) == 0.0
    assert smoothstep_s(1.0, (0.0, 1.0)) == 1.0
    assert smoothstep_s(0.5, (0.0, 1.0)) == 0.5
    values = [smoothstep_s(i / 100, (0.0, 1.0)) for i in range(101)]
    assert all(values[i] <= values[i + 1] + 1e-12 for i in range(100))

    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(schedule_budget=4, max_total_candidates={"scan": 8, "path_neb": 1}),
    )
    smooth = _schedule_by_kind(scheduled, SCHEDULE_SMOOTHSTEP)
    assert smooth.lambda_values == (0.0, 0.25, 0.5, 0.75, 1.0)
    expected = tuple(smoothstep_s(v, (0.0, 1.0)) for v in smooth.lambda_values)
    for _cid, values in smooth.s_by_driver:
        assert values == expected
    assert smooth.window_of("coord-0001") == (0.0, 1.0)


def test_event_a_schedules_require_exactly_two_event_groups() -> None:
    """Given 1 event group, then no event_A; given 2 groups, then both kinds."""
    pool_one, cand_one = _one_group_candidate()
    scheduled_one = build_schedules(
        CandidateWithFeasibility(candidate=cand_one, pool=pool_one, feasibility=_feasibility(5)),
        _config(),
    )
    kinds_one = {spec.kind for spec in scheduled_one.schedules}
    assert SCHEDULE_LINEAR in kinds_one
    assert SCHEDULE_SMOOTHSTEP in kinds_one
    assert SCHEDULE_EVENT_A_EARLY not in kinds_one
    assert SCHEDULE_EVENT_A_LATE not in kinds_one
    assert len(scheduled_one.event_groups) == 1

    pool_two, cand_two = _two_group_candidate()
    scheduled_two = build_schedules(
        CandidateWithFeasibility(candidate=cand_two, pool=pool_two, feasibility=_feasibility(5)),
        _config(),
    )
    kinds_two = {spec.kind for spec in scheduled_two.schedules}
    assert SCHEDULE_EVENT_A_EARLY in kinds_two
    assert SCHEDULE_EVENT_A_LATE in kinds_two
    assert len(scheduled_two.event_groups) == 2
    assert scheduled_two.event_groups[0].event_id == "EV_B"
    assert scheduled_two.event_groups[1].event_id == "EV_H"


def test_event_a_early_frontloads_group_a_and_preserves_endpoints() -> None:
    """Given event_A schedules, then group windows flip and endpoints hold."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    early = _schedule_by_kind(scheduled, SCHEDULE_EVENT_A_EARLY)
    late = _schedule_by_kind(scheduled, SCHEDULE_EVENT_A_LATE)
    for spec in (early, late):
        for _cid, values in spec.s_by_driver:
            assert values[0] == 0.0
            assert values[-1] == 1.0
    group_a = "coord-0003"
    assert early.window_of(group_a) == (0.0, 0.5)
    assert late.window_of(group_a) == (0.5, 1.0)
    mid = 2
    assert early.s_of(group_a)[mid] > late.s_of(group_a)[mid]
    group_h = "coord-0001"
    assert early.window_of(group_h) == (0.5, 1.0)
    assert late.window_of(group_h) == (0.0, 0.5)
    assert early.s_of(group_h)[mid] < late.s_of(group_h)[mid]


def test_suggested_mode_reflects_kind_and_driver_count() -> None:
    """Given linear 1-driver vs multi-driver vs smoothstep, then modes."""
    records = _two_group_records()
    single = _candidate(("coord-0001",))
    pool_single = _pool(records, single)
    scheduled_single = build_schedules(
        CandidateWithFeasibility(candidate=single, pool=pool_single, feasibility=_feasibility(5)),
        _config(),
    )
    linear_single = _schedule_by_kind(scheduled_single, SCHEDULE_LINEAR)
    assert linear_single.suggested_mode == MODE_SINGLE_1D
    assert linear_single.requires_backend_smoke is False

    pool_multi, cand_multi = _two_group_candidate()
    scheduled_multi = build_schedules(
        CandidateWithFeasibility(candidate=cand_multi, pool=pool_multi, feasibility=_feasibility(5)),
        _config(schedule_budget=4, max_total_candidates={"scan": 8, "path_neb": 1}),
    )
    linear_multi = _schedule_by_kind(scheduled_multi, SCHEDULE_LINEAR)
    assert linear_multi.suggested_mode == MODE_COUPLED_1D
    assert linear_multi.requires_backend_smoke is True
    smooth = _schedule_by_kind(scheduled_multi, SCHEDULE_SMOOTHSTEP)
    assert smooth.suggested_mode == MODE_SCHEDULED_1D
    assert smooth.requires_backend_smoke is True


def test_schedule_kind_order_matches_contracts_single_source() -> None:
    """Given the module constants, then they match contracts_v2.SCHEDULE_KINDS."""
    assert SCHEDULE_KIND_ORDER == CONTRACT_SCHEDULE_KINDS
    assert set(SCHEDULE_KIND_ORDER) == {
        SCHEDULE_LINEAR,
        SCHEDULE_EVENT_A_EARLY,
        SCHEDULE_EVENT_A_LATE,
        SCHEDULE_SMOOTHSTEP,
    }


def test_n_points_follows_feasibility_rule_with_baseline_fallback() -> None:
    """Given feasibility n_points / None, then λ grid length follows N."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(12)),
        _config(),
    )
    assert scheduled.n_points == 12
    assert len(scheduled.schedules[0].lambda_values) == 12

    scheduled_none = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(None)),
        _config(),
    )
    assert scheduled_none.n_points == 9
    assert len(scheduled_none.schedules[0].lambda_values) == 9


# ---------------------------------------------------------------------------
# 2. Determinism + purity.
# ---------------------------------------------------------------------------
def test_build_schedules_determinism_identical_serialization() -> None:
    """Given the same input twice, then byte-identical to_json."""
    pool, candidate = _two_group_candidate()
    cw = CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(7))
    first = build_schedules(cw, _config())
    second = build_schedules(cw, _config())
    assert first.to_json() == second.to_json()
    assert first.lambda_by_schedule_id() == second.lambda_by_schedule_id()


def test_to_doc_contains_no_forbidden_truth_keys() -> None:
    """Given a ScheduledCandidate doc, then zero forbidden truth-derived keys."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    assert _scan_forbidden(scheduled.to_doc()) == []


def test_to_doc_carries_no_orca_compilation_syntax() -> None:
    """Given the schedule doc, then no ORCA input-block markers appear."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    text = scheduled.to_json().lower()
    assert "%geom" not in text
    assert "%pal" not in text
    assert "%maxcore" not in text
    assert "scan b " not in text


# ---------------------------------------------------------------------------
# 3. Budget pruning with PRUNED_BY_BUDGET traceability.
# ---------------------------------------------------------------------------
def test_budget_pruning_records_pruned_by_budget_traceably() -> None:
    """Given a tight total budget, then excess schedules are PRUNED_BY_BUDGET."""
    pool, candidate = _two_group_candidate()
    config = _config(max_total_candidates={"scan": 2, "path_neb": 1})
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        config,
    )
    proposed_kinds = {spec.kind for spec in scheduled.schedules} | {
        record.kind for record in scheduled.pruned
    }
    assert proposed_kinds == {
        SCHEDULE_LINEAR,
        SCHEDULE_EVENT_A_EARLY,
        SCHEDULE_EVENT_A_LATE,
        SCHEDULE_SMOOTHSTEP,
    }
    assert scheduled.n_schedules_active == 1
    assert scheduled.n_schedules_pruned == 3
    for record in scheduled.pruned:
        assert record.code == CODE_PRUNED_BY_BUDGET
        assert record.lambda_values
        assert record.reason
        assert "max_total_candidates.scan" in record.reason
    active_ids = {spec.schedule_id for spec in scheduled.schedules}
    for record in scheduled.pruned:
        assert record.schedule_id not in active_ids
    assert scheduled.lambda_by_schedule_id().keys() >= {
        record.schedule_id for record in scheduled.pruned
    }


def test_schedule_budget_cap_prunes_beyond_limit() -> None:
    """Given schedule_budget=2, then at most 2 active + traceable pruned."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(schedule_budget=2),
    )
    assert scheduled.n_schedules_active == 2
    assert scheduled.n_schedules_pruned == 2
    assert scheduled.schedules[0].kind == SCHEDULE_LINEAR
    assert scheduled.schedules[1].kind == SCHEDULE_EVENT_A_EARLY
    for record in scheduled.pruned:
        assert record.code == CODE_PRUNED_BY_BUDGET
        assert "schedule_budget" in record.reason


def test_default_config_keeps_first_schedule_budget_schedules_active() -> None:
    """Given default config (budget 3, 3×2=6 ≤ scan 6), then 3 active + 1 pruned."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    assert scheduled.n_schedules_proposed == 4
    assert scheduled.n_schedules_active == 3
    assert scheduled.n_schedules_pruned == 1
    pruned = scheduled.pruned[0]
    assert pruned.kind == SCHEDULE_SMOOTHSTEP
    assert pruned.code == CODE_PRUNED_BY_BUDGET
    assert "schedule_budget" in pruned.reason


def test_policy_from_config_reads_yaml_style_keys() -> None:
    """Given defaults.yaml-shaped config, then policy values are parsed."""
    policy = policy_from_config(_config())
    assert policy.schedule_budget == 3
    assert policy.direction_budget == 2
    assert policy.assembly_candidate_budget == 2
    assert policy.max_total_candidates_scan == 6
    assert policy.max_total_candidates_path_neb == 1
    assert policy.point_baseline == 9
    assert policy.point_max == 101
    with pytest.raises(ValueError, match="POLICY_INVALID"):
        policy_from_config({"scan_strategy": {"schedule_budget": -1}})


# ---------------------------------------------------------------------------
# 4. Refinement tree closure.
# ---------------------------------------------------------------------------
def _spec_from_scheduled(scheduled: Any, kind: str) -> ScheduleSpec:
    return _schedule_by_kind(scheduled, kind)


def test_refinement_doubling_preserves_endpoints_and_halves_steps() -> None:
    """Given N→2N−1 doubling, then endpoints preserved and midpoints inserted."""
    lam = uniform_lambda_grid(5)
    doubled = double_lambda_grid(lam)
    assert len(doubled) == 2 * len(lam) - 1
    assert doubled[0] == lam[0]
    assert doubled[-1] == lam[-1]
    assert doubled[1] == pytest.approx(0.125)

    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    linear = _spec_from_scheduled(scheduled, SCHEDULE_LINEAR)
    options = refinement_options(linear)
    doubling = next(opt for opt in options if opt.kind == REFINEMENT_KIND_DOUBLING)
    assert doubling.n_points == 2 * linear.n_points - 1
    assert doubling.lambda_values[0] == linear.lambda_values[0]
    assert doubling.lambda_values[-1] == linear.lambda_values[-1]
    assert doubling.depth == 1
    for _cid, values in doubling.s_by_driver:
        assert values[0] == 0.0
        assert values[-1] == 1.0


def test_refinement_local_interval_touches_only_that_interval() -> None:
    """Given a local_interval option, then only that λ interval gains a midpoint."""
    lam = uniform_lambda_grid(5)
    refined = refine_lambda_interval(lam, 1, 2)
    assert len(refined) == len(lam) + 1
    assert refined[0] == lam[0]
    assert refined[-1] == lam[-1]
    assert refined[2] == pytest.approx(0.375)
    assert lam[0] in refined and lam[3] in refined and lam[4] in refined

    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    linear = _spec_from_scheduled(scheduled, SCHEDULE_LINEAR)
    options = refinement_options(linear)
    locals_ = [opt for opt in options if opt.kind == REFINEMENT_KIND_LOCAL_INTERVAL]
    assert len(locals_) == len(linear.lambda_values) - 1
    target = next(opt for opt in locals_ if opt.local_interval == (1, 2))
    assert target.n_points == linear.n_points + 1
    assert target.lambda_values[2] == pytest.approx(0.375)
    assert target.window == (0.25, 0.5)


def test_refinement_options_closed_set_and_max_count() -> None:
    """Given the frozen rule, then closed vocabulary + empty set at max depth."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    linear = _spec_from_scheduled(scheduled, SCHEDULE_LINEAR)
    rule = RefinementRule(max_refinement_count=1)
    options = refinement_options(linear, rule=rule)
    kinds = {opt.kind for opt in options}
    assert kinds <= {REFINEMENT_KIND_DOUBLING, REFINEMENT_KIND_LOCAL_INTERVAL}
    assert len(options) == 1 + (len(linear.lambda_values) - 1)

    depth_one = next(opt for opt in options if opt.kind == REFINEMENT_KIND_DOUBLING)
    child_spec = ScheduleSpec(
        schedule_id=depth_one.option_id,
        kind=linear.kind,
        n_points=depth_one.n_points,
        depth=depth_one.depth,
        lambda_values=depth_one.lambda_values,
        s_by_driver=depth_one.s_by_driver,
        driver_windows=depth_one.driver_windows,
        suggested_mode=linear.suggested_mode,
        requires_backend_smoke=linear.requires_backend_smoke,
        notes=depth_one.notes,
    )
    assert refinement_options(child_spec, rule=rule) == ()

    tree = refinement_tree(linear, rule=rule)
    assert tree.root_schedule_id == linear.schedule_id
    assert tree.options_from(linear) == options


def test_refinement_recomputes_s_on_event_a_schedule() -> None:
    """Given an event_A schedule refinement, then per-driver windows survive."""
    pool, candidate = _two_group_candidate()
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
        _config(),
    )
    early = _spec_from_scheduled(scheduled, SCHEDULE_EVENT_A_EARLY)
    doubling = next(
        opt
        for opt in refinement_options(early)
        if opt.kind == REFINEMENT_KIND_DOUBLING
    )
    assert doubling.driver_windows == early.driver_windows
    group_a_values = dict(doubling.s_by_driver)["coord-0003"]
    assert group_a_values[0] == 0.0
    assert group_a_values[-1] == 1.0
    assert group_a_values == tuple(
        smoothstep_s(v, (0.0, 0.5)) for v in doubling.lambda_values
    )


# ---------------------------------------------------------------------------
# 5. Double-distance integrity (D–H / A–H never a difference single-B).
# ---------------------------------------------------------------------------
def test_double_distance_drivers_survive_into_scheduled_candidate() -> None:
    """Given a real H-transfer pool, then both partner B drivers stay separate."""
    p0 = _h_transfer_pure()
    pool = _pool_from_p0(p0)
    candidates = pool.candidates()
    assert candidates
    h_drivers = [
        record
        for record in pool.coordinates
        if record.role == ROLE_DRIVER
        and record.origin.source_kind == "hydrogen_event"
        and record.origin.geometry_kind == GEOM_PARTNER
    ]
    assert len(h_drivers) >= 2
    donor = next(r for r in h_drivers if r.origin.partner_role == PARTNER_DONOR)
    acceptor = next(r for r in h_drivers if r.origin.partner_role == PARTNER_ACCEPTOR)
    assert donor.coordinate_id != acceptor.coordinate_id
    assert donor.atom_maps != acceptor.atom_maps

    candidate = candidates[0]
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(9)),
        _config(),
    )
    assert scheduled.driver_coordinate_ids == candidate.driver_coordinate_ids
    assert len(set(scheduled.driver_coordinate_ids)) == len(scheduled.driver_coordinate_ids)
    for spec in scheduled.schedules:
        assert set(cid for cid, _v in spec.s_by_driver) == set(candidate.driver_coordinate_ids)


def test_difference_coordinate_in_pool_is_rejected() -> None:
    """Given a pool record claiming a difference coordinate, then hard reject."""
    records = list(_two_group_records())
    records.append(
        _coord(
            "coord-0099",
            KIND_B,
            (1, 2),
            origin=OriginRecord(
                source_kind="hydrogen_event",
                event_id="EV_H",
                hydrogen_map=4,
                geometry_kind="difference",
                note="d1_minus_d2",
            ),
        )
    )
    candidate = _candidate(("coord-0001", "coord-0099"))
    pool = _pool(tuple(records), candidate)
    with pytest.raises(ValueError, match=CODE_DIFFERENCE_COORDINATE_FORBIDDEN):
        build_schedules(
            CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
            _config(),
        )


def test_merged_partner_records_are_rejected() -> None:
    """Given two partner records sharing atom maps, then collapse is rejected."""
    records = list(_two_group_records())
    records[1] = _coord(
        "coord-0002",
        KIND_B,
        (2, 4),
        event_ids=("EV_H",),
        origin=OriginRecord(
            source_kind="hydrogen_event",
            event_id="EV_H",
            hydrogen_map=4,
            partner_map=2,
            partner_role=PARTNER_ACCEPTOR,
            geometry_kind=GEOM_PARTNER,
        ),
    )
    candidate = _candidate(("coord-0001", "coord-0002", "coord-0003"))
    pool = _pool(tuple(records), candidate)
    with pytest.raises(ValueError, match=CODE_H_DOUBLE_DISTANCE_COLLAPSED):
        build_schedules(
            CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(5)),
            _config(),
        )


def test_unresolved_driver_id_is_rejected() -> None:
    """Given a candidate pointing outside the pool, then integrity rejects."""
    pool, _ = _two_group_candidate()
    candidate = _candidate(("coord-0001", "coord-9999"))
    with pytest.raises(ValueError, match=CODE_DRIVER_ID_UNRESOLVED):
        assert_schedule_integrity(pool, candidate)


def test_integrity_notes_record_success_evidence() -> None:
    """Given a clean pool, then integrity notes prove the double-distance rule."""
    pool, candidate = _two_group_candidate()
    notes = assert_schedule_integrity(pool, candidate)
    assert "no_difference_coordinates" in notes
    assert "double_distance_records_distinct" in notes
    assert any(note.startswith("driver_ids_resolved=") for note in notes)


# ---------------------------------------------------------------------------
# 6. Empty-driver guard.
# ---------------------------------------------------------------------------
def test_candidate_without_drivers_yields_empty_schedule_tree() -> None:
    """Given zero drivers, then empty schedules + typed note, no crash."""
    records = _two_group_records()
    candidate = DriverSetCandidate(
        candidate_id="cand:empty",
        driver_coordinate_ids=(),
        monitor_coordinate_ids=("coord-0001",),
        guard_coordinate_ids=(),
        target_test_coordinate_ids=(),
        event_coverage=(),
        has_uncovered_event=False,
        default_automatic=False,
        non_default_reason="NO_DRIVER_COORDINATES",
        route_strategy_id=None,
        route_review_required=False,
        n_drivers=0,
        constraint_rank=0,
        edits_monitored=True,
    )
    pool = _pool(records, candidate)
    scheduled = build_schedules(
        CandidateWithFeasibility(candidate=candidate, pool=pool, feasibility=_feasibility(None)),
        _config(),
    )
    assert scheduled.schedules == ()
    assert scheduled.pruned == ()
    assert "NO_DRIVER_COORDINATES" in scheduled.notes
