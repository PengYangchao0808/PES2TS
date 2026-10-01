"""Tests for the todo-13 geometry feasibility gates (design §6.3 / §8.3)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
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
from pes2ts_core.scan_strategy.coordinate_pool import (
    KIND_A,
    KIND_B,
    KIND_D,
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
from pes2ts_core.scan_strategy.geometry_feasibility import (
    CHECK_ORDER,
    CODE_ANGLE_DEGENERATE,
    CODE_BROKEN_BOND_ALREADY_SEPARATED,
    CODE_CROSS_COMPONENT_OVERLAP,
    CODE_DIHEDRAL_UNDEFINED,
    CODE_DUPLICATE_ATOMS,
    CODE_FORMED_BOND_MISSING_AT_P,
    CODE_INTERPOLATION_INFEASIBLE,
    CODE_JACOBIAN_ILL_CONDITIONED,
    CODE_JACOBIAN_RANK_DEFICIENT,
    CODE_PATH_COLLISION,
    CODE_POINT_BUDGET_EXCEEDED,
    CODE_START_TELEPORT,
    CODE_TRIANGLE_INEQUALITY_VIOLATION,
    DEFAULT_MAX_STEP_ANGLE,
    DEFAULT_MAX_STEP_DIHEDRAL,
    DEFAULT_MAX_STEP_DISTANCE,
    DEFAULT_POINT_BASELINE,
    DEFAULT_POINT_MAX,
    SUGGESTION_POINT_BUDGET,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    assess_geometry_feasibility,
    bond_angle_deg,
    dihedral_deg,
    jacobian_diagnostics,
    policy_from_config,
    shortest_arc_delta,
    unit_scaled_jacobian,
)
from pes2ts_core.scan_strategy.registry import route_strategies

FORBIDDEN = {key.lower() for key in FORBIDDEN_TRUTH_KEYS} | {
    key.lower() for key in FORBIDDEN_EXPORT_KEYS
} | {"endpoint_match", "orientation", "irc_evidence"}


# ---------------------------------------------------------------------------
# Hand-built P0 fixtures (idiom shared with test_coordinate_pool.py).
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
        event_ids=(),
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


def _materials(
    r_coordinates: Mapping[int, tuple[float, float, float]],
    p_coordinates: Mapping[int, tuple[float, float, float]],
) -> EndpointMaterials:
    return EndpointMaterials(
        r_coordinates=dict(r_coordinates), p_coordinates=dict(p_coordinates)
    )


def _codes(report: Any) -> list[str]:
    return [failure.code for failure in report.failures]


def _check_status(report: Any, check_id: str) -> str:
    for check in report.checks:
        if check.check_id == check_id:
            return check.status
    raise AssertionError(f"missing check {check_id}")


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


# ---------------------------------------------------------------------------
# 1. Cross-component overlap detection.
# ---------------------------------------------------------------------------
def test_cross_component_overlap_detected() -> None:
    """Given nonbonded atoms of two components 0.5 Å apart, then rejection."""
    bundle = _hand_bundle([(1, 2, 1), (3, 4, 1)], [(1, 2, 1), (3, 4, 1)])
    record = _coord("coord-0001", KIND_B, (1, 2), origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="order_changed"))
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.5, 0.0, 0.0),
            3: (1.6, 0.0, 0.0),  # 0.5 Å from map 2, different component
            4: (3.1, 0.0, 0.0),
        },
        p_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.5, 0.0, 0.0),
            3: (1.6, 0.0, 0.0),
            4: (3.1, 0.0, 0.0),
        },
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_CROSS_COMPONENT_OVERLAP in _codes(report)
    assert _check_status(report, "cross_component_overlap") == STATUS_FAIL


# ---------------------------------------------------------------------------
# 2. Wrong broken-bond endpoint (atoms already separated at R).
# ---------------------------------------------------------------------------
def test_wrong_broken_bond_endpoint_rejected() -> None:
    """Given an R-graph bond whose R geometry is already separated, then rejection."""
    bundle = _hand_bundle([(1, 2, 1)], [], elements={1: "C", 2: "C"})
    record = _coord(
        "coord-0001",
        KIND_B,
        (1, 2),
        origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="broken"),
    )
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (3.5, 0.0, 0.0)},  # already separated
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (5.0, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_BROKEN_BOND_ALREADY_SEPARATED in _codes(report)


def test_formed_bond_missing_at_p_rejected() -> None:
    """Given a P-graph bond absent in P geometry, then typed rejection."""
    bundle = _hand_bundle([], [(1, 2, 1)], elements={1: "C", 2: "C"})
    record = _coord(
        "coord-0001",
        KIND_B,
        (1, 2),
        origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="formed"),
    )
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (4.0, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (4.5, 0.0, 0.0)},  # still separated
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_FORMED_BOND_MISSING_AT_P in _codes(report)


# ---------------------------------------------------------------------------
# 3. Triangle-inequality failure for multi-B.
# ---------------------------------------------------------------------------
def test_triangle_inequality_failure_multi_b() -> None:
    """Given planned multi-B targets that break triangle inequality, reject.

    Realizable endpoint geometries can never violate Euclidean triangle
    inequalities; the precheck judges the scan's *planned* driver values
    (origin r_value/p_value), which a contradictory candidate can break.
    """
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C", 3: "C"})
    records = (
        _coord(
            "coord-0001",
            KIND_B,
            (1, 2),
            origin=OriginRecord(source_kind="hydrogen_event", r_value=1.5, p_value=1.5),
        ),
        _coord(
            "coord-0002",
            KIND_B,
            (2, 3),
            origin=OriginRecord(source_kind="hydrogen_event", r_value=1.5, p_value=1.5),
        ),
        _coord(
            "coord-0003",
            KIND_B,
            (1, 3),
            origin=OriginRecord(source_kind="hydrogen_event", r_value=3.6, p_value=3.6),
        ),
    )
    candidate = _candidate(("coord-0001", "coord-0002", "coord-0003"))
    # Valid Euclidean geometry — the violation lives in the planned values
    # (1.5 + 1.5 < 3.6), not in the coordinates.
    xyz = {1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (1.0, 1.8, 0.0)}
    materials = _materials(r_coordinates=xyz, p_coordinates=xyz)

    report = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_TRIANGLE_INEQUALITY_VIOLATION in _codes(report)


# ---------------------------------------------------------------------------
# 4. A/D collinear (±180°) rejection.
# ---------------------------------------------------------------------------
def test_angle_collinear_180_rejected() -> None:
    """Given a collinear A triple (angle 180°), then degeneracy rejection."""
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C", 3: "C"})
    record = _coord("coord-0001", KIND_A, (1, 2, 3))
    candidate = _candidate(("coord-0001",))
    collinear = {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (3.0, 0.0, 0.0)}
    materials = _materials(r_coordinates=collinear, p_coordinates=collinear)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_ANGLE_DEGENERATE in _codes(report)


def test_angle_near_180_rejected() -> None:
    """Given an A angle within tolerance of 180°, then degeneracy rejection."""
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C", 3: "C"})
    record = _coord("coord-0001", KIND_A, (1, 2, 3))
    candidate = _candidate(("coord-0001",))
    # 179.2°: atom 3 offset 0.02 Å perpendicular to the 1–2 axis.
    near = {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (3.0, 0.02, 0.0)}
    angle = bond_angle_deg(near[1], near[2], near[3])
    assert 178.5 < angle < 179.5  # within the 1.0° degeneracy window of 180
    materials = _materials(r_coordinates=near, p_coordinates=near)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_ANGLE_DEGENERATE in _codes(report)


def test_dihedral_collinear_rejected() -> None:
    """Given a D whose central-bond triples are collinear, then rejection."""
    bundle = _hand_bundle(
        [], [], elements={1: "C", 2: "C", 3: "C", 4: "C"}
    )
    record = _coord("coord-0001", KIND_D, (1, 2, 3, 4))
    candidate = _candidate(("coord-0001",))
    collinear = {
        1: (0.0, 0.0, 0.0),
        2: (1.5, 0.0, 0.0),
        3: (3.0, 0.0, 0.0),
        4: (4.5, 0.0, 0.0),
    }
    materials = _materials(r_coordinates=collinear, p_coordinates=collinear)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_DIHEDRAL_UNDEFINED in _codes(report)


# ---------------------------------------------------------------------------
# 5. Equivalent-H mapping acceptance (no false positive).
# ---------------------------------------------------------------------------
def test_equivalent_hydrogen_mapping_accepted() -> None:
    """Given symmetry-equivalent H drivers, feasibility must NOT false-positive.

    Methyl-like C2(H4)(H5): both symmetry-equivalent C-H stretches are driven.
    Equivalent-H mapping must not trip overlap, duplicate-atom, or rank gates.
    """
    bundle = _hand_bundle(
        [(1, 2, 1), (2, 4, 1), (2, 5, 1)],
        [(1, 2, 1), (2, 4, 1), (2, 5, 1)],
        elements={1: "C", 2: "C", 4: "H", 5: "H"},
    )
    records = (
        _coord(
            "coord-0001",
            KIND_B,
            (2, 4),
            origin=OriginRecord(
                source_kind="edit", edit_pair=(2, 4), edit_kind="order_changed"
            ),
        ),
        _coord(
            "coord-0002",
            KIND_B,
            (2, 5),
            origin=OriginRecord(
                source_kind="edit", edit_pair=(2, 5), edit_kind="order_changed"
            ),
        ),
    )
    candidate = _candidate(("coord-0001", "coord-0002"))
    # Symmetry-equivalent H4/H5 about the C1-C2 axis.
    materials = _materials(
        r_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.5, 0.0, 0.0),
            4: (1.5, 1.09, 0.0),
            5: (1.5, -0.545, 0.944),
        },
        p_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.5, 0.0, 0.0),
            4: (1.5, 1.09, 0.0),
            5: (1.5, -0.545, 0.944),
        },
    )
    pool = _pool(records, candidate)

    report = assess_geometry_feasibility(candidate, pool, bundle, materials, {})

    assert _check_status(report, "cross_component_overlap") == STATUS_PASS
    assert _check_status(report, "endpoint_degeneracy") == STATUS_PASS
    assert _check_status(report, "jacobian_rank") == STATUS_PASS
    assert _check_status(report, "jacobian_condition") == STATUS_PASS
    assert report.jacobian_rank_by_endpoint
    for _endpoint, rank, n_coords in report.jacobian_rank_by_endpoint:
        assert rank == n_coords == 2
    assert not any(
        code in _codes(report)
        for code in (CODE_DUPLICATE_ATOMS, CODE_CROSS_COMPONENT_OVERLAP)
    )


# ---------------------------------------------------------------------------
# 6. Near-degenerate / contradictory constraints rejected.
# ---------------------------------------------------------------------------
def test_contradictory_duplicate_drivers_rejected() -> None:
    """Given two drivers tracking the same motion, then rank-deficiency rejection."""
    bundle = _hand_bundle([(1, 2, 1)], [(1, 2, 1)], elements={1: "C", 2: "C"})
    records = (
        _coord(
            "coord-0001",
            KIND_B,
            (1, 2),
            origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="order_changed"),
        ),
        _coord(
            "coord-0002",
            KIND_B,
            (1, 2),  # identical motion
            origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="order_changed"),
        ),
    )
    candidate = _candidate(("coord-0001", "coord-0002"))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_JACOBIAN_RANK_DEFICIENT in _codes(report)
    for _endpoint, rank, n_coords in report.jacobian_rank_by_endpoint:
        assert rank < n_coords


def test_h2_three_distance_contradiction_rejected() -> None:
    """Given H2-style planned three distances that violate triangle inequality, reject."""
    bundle = _hand_bundle([], [], elements={1: "H", 2: "H", 3: "H"})
    # Planned H2 set: d(1,2)=3.0, d(1,3)=3.0, d(2,3)=7.0 -> 3+3 < 7.
    records = (
        _coord(
            "coord-0001",
            KIND_B,
            (1, 2),
            origin=OriginRecord(
                source_kind="hydrogen_event",
                geometry_kind="hh_distance",
                r_value=3.0,
                p_value=3.0,
            ),
        ),
        _coord(
            "coord-0002",
            KIND_B,
            (1, 3),
            origin=OriginRecord(
                source_kind="hydrogen_event",
                geometry_kind="partner_distance",
                r_value=3.0,
                p_value=3.0,
            ),
        ),
        _coord(
            "coord-0003",
            KIND_B,
            (2, 3),
            origin=OriginRecord(
                source_kind="hydrogen_event",
                geometry_kind="partner_distance",
                r_value=7.0,
                p_value=7.0,
            ),
        ),
    )
    candidate = _candidate(("coord-0001", "coord-0002", "coord-0003"))
    xyz = {1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (1.0, 1.8, 0.0)}
    materials = _materials(r_coordinates=xyz, p_coordinates=xyz)

    report = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_TRIANGLE_INEQUALITY_VIOLATION in _codes(report)


# ---------------------------------------------------------------------------
# 7. Jacobian rank deficiency (two coordinates, same motion).
# ---------------------------------------------------------------------------
def test_jacobian_rank_deficiency_rejected() -> None:
    """Given duplicated driver motions, rank(J) < n_coords at the endpoint."""
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C"})
    records = (
        _coord("coord-0001", KIND_B, (1, 2)),
        _coord("coord-0002", KIND_B, (1, 2)),
    )
    candidate = _candidate(("coord-0001", "coord-0002"))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (2.5, 0.0, 0.0)},
    )
    report = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, {}
    )
    assert CODE_JACOBIAN_RANK_DEFICIENT in _codes(report)

    # Direct Jacobian diagnostic on the identical rows.
    policy = policy_from_config({})
    j_matrix = unit_scaled_jacobian(records, dict(materials.r_coordinates), policy)
    assert j_matrix is not None
    rank, kappa = jacobian_diagnostics(j_matrix)
    assert rank == 1
    assert math.isinf(kappa)


# ---------------------------------------------------------------------------
# 8. Condition-number gate.
# ---------------------------------------------------------------------------
def test_jacobian_condition_number_gate() -> None:
    """Given full-rank but nearly-parallel rows and a tight kmax, reject."""
    bundle = _hand_bundle([(1, 2, 1)], [], elements={1: "C", 2: "H", 3: "H"})
    records = (
        _coord("coord-0001", KIND_B, (1, 2)),
        _coord("coord-0002", KIND_B, (1, 3)),
    )
    candidate = _candidate(("coord-0001", "coord-0002"))
    # H3 nearly collinear with C1-H2: gradients nearly parallel -> large kappa.
    xyz = {
        1: (0.0, 0.0, 0.0),
        2: (1.5, 0.0, 0.0),
        3: (3.0, 0.01, 0.0),
    }
    materials = _materials(r_coordinates=xyz, p_coordinates=xyz)
    policy = policy_from_config({})
    j_matrix = unit_scaled_jacobian(records, xyz, policy)
    assert j_matrix is not None
    rank, kappa = jacobian_diagnostics(j_matrix)
    assert rank == 2
    # Two bond-length drivers sharing an atom have kappa <= sqrt(3) ~= 1.732;
    # near-collinear placement saturates that bound.
    assert kappa > 1.5

    # Default threshold (1e6) accepts this geometry.
    permissive = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, {}
    )
    assert _check_status(permissive, "jacobian_condition") == STATUS_PASS

    # Tight threshold below the achieved kappa rejects it.
    tight_config = {"scan_strategy": {"jacobian_condition_number_max": 1.5}}
    report = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, tight_config
    )
    assert not report.ok
    assert CODE_JACOBIAN_ILL_CONDITIONED in _codes(report)
    assert report.jacobian_condition_max is not None
    assert report.jacobian_condition_max > 1.5


# ---------------------------------------------------------------------------
# 9. D periodic unwrap (350° → −10° shortest arc).
# ---------------------------------------------------------------------------
def test_dihedral_periodic_unwrap_shortest_arc() -> None:
    """Given D endpoints crossing ±180°/0°, Δ uses the shortest arc."""
    # 350° and −10° are coterminal: shortest arc is 0, never a raw −360° jump.
    assert shortest_arc_delta(350.0, -10.0) == 0.0
    assert shortest_arc_delta(-10.0, 350.0) == 0.0
    # ±180° cut: 170° → −170° is +20°, not −340°.
    assert abs(shortest_arc_delta(170.0, -170.0) - 20.0) < 1e-9
    # 0°/360° boundary: 350° → 10° is +20°.
    assert abs(shortest_arc_delta(350.0, 10.0) - 20.0) < 1e-9

    # Integration: conformational D whose recorded endpoints are 350°/−10°
    # must contribute total variation 0 to the point-count rule.
    bundle = _hand_bundle(
        [], [], elements={1: "C", 2: "C", 3: "C", 4: "C"}
    )
    record = _coord(
        "coord-0001",
        KIND_D,
        (1, 2, 3, 4),
        origin=OriginRecord(
            source_kind="conformational_difference",
            geometry_kind="torsion_difference",
            r_value=350.0,
            p_value=-10.0,
        ),
    )
    candidate = _candidate(("coord-0001",))
    # Geometry realizes dihedrals ≈ −10° (≡350°) on both sides.
    r_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(-10.0)), 1.5 * math.sin(math.radians(-10.0))),
    }
    p_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(-10.0)), 1.5 * math.sin(math.radians(-10.0))),
    }
    q_r = dihedral_deg(r_xyz[1], r_xyz[2], r_xyz[3], r_xyz[4])
    q_p = dihedral_deg(p_xyz[1], p_xyz[2], p_xyz[3], p_xyz[4])
    assert abs(q_r - (-10.0)) < 1e-6
    assert abs(shortest_arc_delta(q_r, q_p)) < 1e-6
    materials = _materials(r_coordinates=r_xyz, p_coordinates=p_xyz)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    # TV = 0 → N falls back to the baseline; no budget issue.
    assert report.n_points == DEFAULT_POINT_BASELINE
    assert not report.point_budget_exceeded
    tv = dict(report.total_variation_by_coord)["coord-0001"]
    assert abs(tv) < 1e-6
    unwrap_check = next(
        c for c in report.checks if c.check_id == "dihedral_periodic_unwrap"
    )
    assert "Δ=0.0000°" in unwrap_check.detail
    assert "naive" in unwrap_check.detail


def test_dihedral_unwrap_crosses_180_in_point_count() -> None:
    """Given D endpoints 170°/−170°, TV=20° via shortest arc, not 340°."""
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C", 3: "C", 4: "C"})
    record = _coord("coord-0001", KIND_D, (1, 2, 3, 4))
    candidate = _candidate(("coord-0001",))
    r_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(170.0)), 1.5 * math.sin(math.radians(170.0))),
    }
    p_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(-170.0)), 1.5 * math.sin(math.radians(-170.0))),
    }
    q_r = dihedral_deg(r_xyz[1], r_xyz[2], r_xyz[3], r_xyz[4])
    q_p = dihedral_deg(p_xyz[1], p_xyz[2], p_xyz[3], p_xyz[4])
    delta = shortest_arc_delta(q_r, q_p)
    assert abs(abs(delta) - 20.0) < 1e-3
    materials = _materials(r_coordinates=r_xyz, p_coordinates=p_xyz)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    tv = dict(report.total_variation_by_coord)["coord-0001"]
    assert abs(tv - 20.0) < 1e-3
    # TV=20°, step 10° → ceil(2)=2 → N_required=3 → N=baseline 9.
    assert report.n_points_required == 3
    assert report.n_points == DEFAULT_POINT_BASELINE


# ---------------------------------------------------------------------------
# 10. Point-count rule math (hand-computed cases).
# ---------------------------------------------------------------------------
def _b_case(r_distance: float, p_distance: float) -> tuple[Any, Any, Any, Any]:
    """Build (bundle, record, candidate, materials) for a single B driver."""
    bundle = _hand_bundle([(1, 2, 1)], [], elements={1: "C", 2: "C"})
    record = _coord(
        "coord-0001",
        KIND_B,
        (1, 2),
        origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="broken"),
    )
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (r_distance, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (p_distance, 0.0, 0.0)},
    )
    return bundle, record, candidate, materials


def test_point_count_rule_math_baseline_wins() -> None:
    """Given TV=0.3 Å with step 0.2, N_required=3 < baseline → N=9."""
    bundle, record, candidate, materials = _b_case(1.8, 2.1)  # TV=0.3
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )
    assert report.ok
    # ceil(0.3/0.2)=2 → N_required=1+2=3; N=max(9,3)=9.
    assert report.n_points_required == 3
    assert report.n_points == DEFAULT_POINT_BASELINE
    assert report.n_points_baseline == 9
    assert dict(report.total_variation_by_coord)["coord-0001"] == pytest.approx(0.3)


def test_point_count_rule_math_required_wins() -> None:
    """Given TV=2.2 Å with step 0.2, N_required=12 > baseline → N=12."""
    bundle, record, candidate, materials = _b_case(1.8, 4.0)  # TV=2.2
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )
    assert report.ok
    # ceil(2.2/0.2)=11 → N_required=12; N=max(9,12)=12.
    assert report.n_points_required == 12
    assert report.n_points == 12


def test_point_count_rule_exact_multiple_no_inflation() -> None:
    """Given TV an exact multiple of the step, ceil must not inflate N."""
    # TV=1.0 Å, step 0.2 → 1.0/0.2=5 exactly → N_required=6 → N=max(9,6)=9.
    bundle, record, candidate, materials = _b_case(1.5, 2.5)
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )
    assert report.n_points_required == 6
    assert report.n_points == DEFAULT_POINT_BASELINE


def test_point_count_rule_angle_and_dihedral_steps() -> None:
    """Given A (60->120 deg) and D (0 deg) drivers, hand-check N_required."""
    bundle = _hand_bundle(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "C"},
    )
    records = (
        _coord("coord-0001", KIND_A, (1, 2, 3)),
        _coord("coord-0002", KIND_D, (1, 2, 3, 4)),
    )
    candidate = _candidate(("coord-0001", "coord-0002"))
    # Chain 1-2-3-4 with bond lengths 1.5 A. Angle at vertex 2 set via atom3;
    # dihedral kept in-plane (0 deg) on both sides by placing atom4 in the
    # plane of 1-2-3.
    def _frame(angle_deg_value: float) -> dict[int, tuple[float, float, float]]:
        sin_a = math.sin(math.radians(angle_deg_value))
        cos_a = math.cos(math.radians(angle_deg_value))
        atom2 = (0.0, 0.0, 0.0)
        atom1 = (0.0, 1.5, 0.0)
        atom3 = (1.5 * sin_a, 1.5 * cos_a, 0.0)
        # Unit direction perpendicular to b1=(atom3-atom2) inside the z=0 plane.
        bx, by, _bz = atom3[0] - atom2[0], atom3[1] - atom2[1], 0.0
        norm = math.hypot(bx, by)
        px, py = -by / norm, bx / norm
        atom4 = (atom3[0] + 1.5 * px, atom3[1] + 1.5 * py, 0.0)
        return {1: atom1, 2: atom2, 3: atom3, 4: atom4}

    r_xyz = _frame(60.0)
    p_xyz = _frame(120.0)
    q_a_r = bond_angle_deg(r_xyz[1], r_xyz[2], r_xyz[3])
    q_a_p = bond_angle_deg(p_xyz[1], p_xyz[2], p_xyz[3])
    assert abs(q_a_r - 60.0) < 1e-6
    assert abs(q_a_p - 120.0) < 1e-6
    q_d_r = dihedral_deg(r_xyz[1], r_xyz[2], r_xyz[3], r_xyz[4])
    q_d_p = dihedral_deg(p_xyz[1], p_xyz[2], p_xyz[3], p_xyz[4])
    assert abs(q_d_r) < 1e-6
    assert abs(q_d_p) < 1e-6
    materials = _materials(r_coordinates=r_xyz, p_coordinates=p_xyz)

    report = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, {}
    )
    assert report.ok, [(f.code, f.detail) for f in report.failures]
    tv = dict(report.total_variation_by_coord)
    # A: TV=60 deg, step 10 -> ceil(6)=6 -> N_req=7. D: TV=0 -> N_req=1.
    assert tv["coord-0001"] == pytest.approx(60.0)
    assert tv["coord-0002"] == pytest.approx(0.0)
    assert report.n_points_required == 7
    assert report.n_points == DEFAULT_POINT_BASELINE

    # Config override: angle step 5 deg -> ceil(60/5)=12 -> N_req=13 > baseline.
    tight = {"scan_strategy": {"max_step_by_kind": {"angle": 5.0}}}
    report_tight = assess_geometry_feasibility(
        candidate, _pool(records, candidate), bundle, materials, tight
    )
    assert report_tight.n_points_required == 13
    assert report_tight.n_points == 13


def test_point_count_rule_config_defaults_match_yaml() -> None:
    """Given config/defaults.yaml, policy keys match the documented constants."""
    from pes2ts_core.config_loader import load_config

    config = load_config()
    policy = policy_from_config(config)
    assert policy.point_baseline == DEFAULT_POINT_BASELINE == 9
    assert policy.point_max == DEFAULT_POINT_MAX == 101
    assert policy.max_step_distance == DEFAULT_MAX_STEP_DISTANCE == 0.2
    assert policy.max_step_angle == DEFAULT_MAX_STEP_ANGLE == 10.0
    assert policy.max_step_dihedral == DEFAULT_MAX_STEP_DIHEDRAL == 10.0
    assert policy.jacobian_condition_number_max == 1.0e6
    assert policy.bond_tolerance == 0.45


# ---------------------------------------------------------------------------
# 11. N > max rejection is typed.
# ---------------------------------------------------------------------------
def test_point_budget_exceeded_typed_rejection() -> None:
    """Given N_required > point_limits.max, typed POINT_BUDGET_EXCEEDED + suggestion."""
    bundle, record, candidate, materials = _b_case(1.5, 25.0)  # TV=23.5 Å
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )
    assert not report.ok
    assert report.point_budget_exceeded
    assert CODE_POINT_BUDGET_EXCEEDED in _codes(report)
    assert report.n_points_required == 119  # 1 + ceil(23.5/0.2)=1+118
    assert report.n_points == 119
    budget_failures = [
        f for f in report.failures if f.code == CODE_POINT_BUDGET_EXCEEDED
    ]
    assert budget_failures
    assert budget_failures[0].suggestion == SUGGESTION_POINT_BUDGET
    assert "todo 17" in budget_failures[0].suggestion
    point_check = next(c for c in report.checks if c.check_id == "point_count")
    assert point_check.status == STATUS_FAIL


def test_point_budget_typed_with_custom_limits() -> None:
    """Given point_limits.max=5, a modest scan still trips the typed verdict."""
    bundle, record, candidate, materials = _b_case(1.5, 4.0)  # TV=2.5 → N_req=14
    config = {
        "scan_strategy": {
            "point_limits": {"baseline": 3, "max": 5},
        }
    }
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, config
    )
    assert not report.ok
    assert CODE_POINT_BUDGET_EXCEEDED in _codes(report)
    assert report.n_points_baseline == 3
    assert report.n_points_required == 14


# ---------------------------------------------------------------------------
# 12. Start-geometry match / no teleporting scan.
# ---------------------------------------------------------------------------
def test_start_geometry_teleport_rejected() -> None:
    """Given a conformational D whose recorded start mismatches geometry, reject."""
    bundle = _hand_bundle(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "C"},
    )
    record = _coord(
        "coord-0001",
        KIND_D,
        (1, 2, 3, 4),
        origin=OriginRecord(
            source_kind="conformational_difference",
            geometry_kind="torsion_difference",
            r_value=10.0,   # recorded start claims 10°
            p_value=170.0,
        ),
    )
    candidate = _candidate(("coord-0001",))
    # R geometry actually realizes ~170° — teleporting-scan mismatch.
    r_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(170.0)), 1.5 * math.sin(math.radians(170.0))),
    }
    p_xyz = dict(r_xyz)
    q_r = dihedral_deg(r_xyz[1], r_xyz[2], r_xyz[3], r_xyz[4])
    assert abs(q_r - 170.0) < 1e-3
    materials = _materials(r_coordinates=r_xyz, p_coordinates=p_xyz)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_START_TELEPORT in _codes(report)
    teleport_check = next(
        c for c in report.checks if c.check_id == "start_geometry_match"
    )
    assert teleport_check.status == STATUS_FAIL


def test_start_geometry_match_passes_when_consistent() -> None:
    """Given a conformational D whose recorded start matches geometry, pass."""
    bundle = _hand_bundle(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "C"},
    )
    record = _coord(
        "coord-0001",
        KIND_D,
        (1, 2, 3, 4),
        origin=OriginRecord(
            source_kind="conformational_difference",
            geometry_kind="torsion_difference",
            r_value=170.0,
            p_value=-170.0,
        ),
    )
    candidate = _candidate(("coord-0001",))
    r_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(170.0)), 1.5 * math.sin(math.radians(170.0))),
    }
    p_xyz = {
        1: (0.0, 1.5, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (1.5, 0.0, 0.0),
        4: (1.5, 1.5 * math.cos(math.radians(-170.0)), 1.5 * math.sin(math.radians(-170.0))),
    }
    materials = _materials(r_coordinates=r_xyz, p_coordinates=p_xyz)

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert _check_status(report, "start_geometry_match") == STATUS_PASS


# ---------------------------------------------------------------------------
# 13. Path precheck: interpolation collision.
# ---------------------------------------------------------------------------
def test_path_precheck_collision_midway_rejected() -> None:
    """Given endpoint geometries whose interpolation collides mid-path, reject."""
    bundle = _hand_bundle(
        [(1, 2, 1), (1, 3, 1)], [(1, 2, 1), (1, 3, 1)], elements={1: "C", 2: "C", 3: "C"}
    )
    record = _coord(
        "coord-0001",
        KIND_B,
        (1, 2),
        origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="order_changed"),
    )
    candidate = _candidate(("coord-0001",))
    # Atoms 2 and 3 swap sides: linear interpolation passes both through the
    # origin at λ=0.5 (severe nonbonded clash).
    materials = _materials(
        r_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (-1.8, 0.0, 0.0),
            3: (1.8, 0.0, 0.0),
        },
        p_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.8, 0.0, 0.0),
            3: (-1.8, 0.0, 0.0),
        },
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    path_codes = {
        CODE_PATH_COLLISION,
        CODE_INTERPOLATION_INFEASIBLE,
    } & set(_codes(report))
    assert path_codes, _codes(report)
    assert _check_status(report, "path_precheck") == STATUS_FAIL


# ---------------------------------------------------------------------------
# 14. Guardrail: 0.45–8 Å crude range is NEVER a qualification gate.
# ---------------------------------------------------------------------------
def test_crude_045_8_range_is_not_a_qualification_gate() -> None:
    """Given a broken-bond scan to 8.5 Å (outside 0.45–8), feasibility still passes.

    planning.py's legacy ContractError range is reference-only (todo spec
    MUST-NOT-DO); bondedness here is Cordero + tolerance, and large
    dissociation distances are legitimate scan endpoints.
    """
    bundle, record, candidate, materials = _b_case(1.5, 8.5)  # P distance 8.5 Å
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )
    assert report.ok, [f.code for f in report.failures]
    assert not any(
        "0.45" in f.detail or "8" == f.detail for f in report.failures
    )
    # The module must not implement the planning.py range as a gate.
    import inspect

    from pes2ts_core.scan_strategy import geometry_feasibility as module

    source = inspect.getsource(module)
    assert "0.45 <= " not in source  # no crude-range comparison gate


def test_short_bond_below_crude_floor_still_vetted_by_cordero() -> None:
    """Given an implausibly short graph-bonded pair, Cordero lower bound rejects."""
    bundle = _hand_bundle([(1, 2, 1)], [(1, 2, 1)], elements={1: "C", 2: "C"})
    record = _coord(
        "coord-0001",
        KIND_B,
        (1, 2),
        origin=OriginRecord(source_kind="edit", edit_pair=(1, 2), edit_kind="order_changed"),
    )
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (0.1, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert any(
        code.startswith("BONDED") or code.startswith("B_DISTANCE")
        for code in _codes(report)
    ), _codes(report)


# ---------------------------------------------------------------------------
# 15. Coordinate definition negatives.
# ---------------------------------------------------------------------------
def test_duplicate_atoms_in_coordinate_rejected() -> None:
    """Given a B coordinate listing the same atom twice, schema rejection."""
    bundle = _hand_bundle([], [], elements={1: "C"})
    record = _coord("coord-0001", KIND_B, (1, 1))
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert CODE_DUPLICATE_ATOMS in _codes(report)


def test_unit_mismatch_rejected() -> None:
    """Given a B coordinate labelled in degrees, schema rejection."""
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C"})
    record = _coord("coord-0001", KIND_B, (1, 2), units=UNIT_DEGREE)
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    assert any(f.code.endswith("UNIT_MISMATCH") or "units" in f.detail for f in report.failures)


def test_b_distance_nonpositive_rejected() -> None:
    """Given coincident atoms for a B driver, non-positive distance rejection."""
    bundle = _hand_bundle([], [], elements={1: "C", 2: "C"})
    record = _coord("coord-0001", KIND_B, (1, 2))
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    from pes2ts_core.scan_strategy.geometry_feasibility import (
        CODE_B_DISTANCE_NONPOSITIVE,
    )

    assert CODE_B_DISTANCE_NONPOSITIVE in _codes(report)


def test_materials_required_typed_when_missing() -> None:
    """Given no endpoint materials, geometry checks skip with MATERIALS_REQUIRED."""
    bundle = _hand_bundle([(1, 2, 1)], [], elements={1: "C", 2: "C"})
    record = _coord("coord-0001", KIND_B, (1, 2))
    candidate = _candidate(("coord-0001",))

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, None, {}
    )

    assert not report.ok
    from pes2ts_core.scan_strategy.geometry_feasibility import CODE_MATERIALS_REQUIRED

    assert CODE_MATERIALS_REQUIRED in _codes(report)
    skipped = [c for c in report.checks if c.status == STATUS_SKIPPED]
    assert skipped
    assert report.n_points is None


# ---------------------------------------------------------------------------
# 16. Happy path through the full P0 pipeline.
# ---------------------------------------------------------------------------
def test_happy_path_single_bond_formation_pipeline() -> None:
    """Given a healthy single-bond formation, all feasibility checks pass."""
    p0 = _p0([(1, 2, 1)], [(1, 2, 1), (1, 3, 1)], elements={1: "C", 2: "C", 3: "C"})
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (3.5, 0.5, 0.0)},
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (1.55, 0.0, 0.0)},
    )
    pool = _pool_from_p0(p0, materials)
    assert pool.candidates()
    candidate = pool.candidates()[0]
    assert candidate.default_automatic

    report = assess_geometry_feasibility(candidate, pool, p0["bundle"], materials, {})

    assert report.ok, [(f.code, f.detail) for f in report.failures]
    assert report.n_points is not None
    assert report.n_points >= DEFAULT_POINT_BASELINE
    status_by_id = {c.check_id: c.status for c in report.checks}
    for check_id in CHECK_ORDER:
        assert status_by_id[check_id] == STATUS_PASS, (check_id, status_by_id[check_id])
    # Formed bond 1–3 is bonded on P (1.55 Å ≤ 0.76+0.76+0.45) and far on R.
    assert dict(report.total_variation_by_coord)


# ---------------------------------------------------------------------------
# 17. Report hygiene: determinism + no truth keys.
# ---------------------------------------------------------------------------
def test_report_deterministic_and_truth_free() -> None:
    """Given identical inputs, two reports serialize identically, no truth keys."""
    bundle, record, candidate, materials = _b_case(1.8, 3.2)
    pool = _pool((record,), candidate)
    first = assess_geometry_feasibility(candidate, pool, bundle, materials, {})
    second = assess_geometry_feasibility(candidate, pool, bundle, materials, {})
    assert first.to_json() == second.to_json()
    assert _scan_forbidden(first.to_doc()) == []


def test_all_checks_recorded_even_on_failure() -> None:
    """Given a failing candidate, every check id in CHECK_ORDER is present."""
    bundle = _hand_bundle([(1, 2, 1)], [], elements={1: "C", 2: "C"})
    record = _coord("coord-0001", KIND_B, (1, 2))
    candidate = _candidate(("coord-0001",))
    materials = _materials(
        r_coordinates={1: (0.0, 0.0, 0.0), 2: (3.5, 0.0, 0.0)},  # broken bond far at R
        p_coordinates={1: (0.0, 0.0, 0.0), 2: (5.0, 0.0, 0.0)},
    )

    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, {}
    )

    assert not report.ok
    seen = {c.check_id for c in report.checks}
    assert seen == set(CHECK_ORDER)
    for failure in report.failures:
        assert failure.check_id in CHECK_ORDER
        assert failure.code


def test_policy_invalid_config_typed() -> None:
    """Given an invalid policy value, typed POLICY_INVALID rejection."""
    bundle, record, candidate, materials = _b_case(1.5, 3.0)
    bad_config = {"scan_strategy": {"max_step_by_kind": {"distance": -0.2}}}
    report = assess_geometry_feasibility(
        candidate, _pool((record,), candidate), bundle, materials, bad_config
    )
    assert not report.ok
    from pes2ts_core.scan_strategy.geometry_feasibility import CODE_POLICY_INVALID

    assert CODE_POLICY_INVALID in _codes(report)
    assert all(c.status == STATUS_SKIPPED for c in report.checks)


def test_start_endpoint_invalid_raises() -> None:
    """Given an unknown start_endpoint, a typed ValueError is raised."""
    bundle, record, candidate, materials = _b_case(1.5, 3.0)
    with pytest.raises(ValueError, match="START_ENDPOINT_INVALID"):
        assess_geometry_feasibility(
            candidate,
            _pool((record,), candidate),
            bundle,
            materials,
            {},
            start_endpoint="side",
        )
