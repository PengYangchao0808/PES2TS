"""Tests for the todo-12 coordinate pool (design §6.1–§6.2)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_context import EndpointElectronic, build_endpoint_context
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
from pes2ts_core.generation.planning.contracts_v2 import COVERAGE_KINDS
from pes2ts_core.generation.planning.coordinate_pool import (
    COVERAGE_COUPLED_MONITOR,
    COVERAGE_DIRECT,
    COVERAGE_UNCOVERED,
    CONFORMATIONAL_DELTA_MIN_DEG,
    EVID_SAME_HYDROGEN,
    EVID_SHARED_ATOM,
    GEOM_ANGLE_DHA,
    GEOM_ATTACK_ANGLE,
    GEOM_ATTACK_TORSION,
    GEOM_DISTANCE_DA,
    GEOM_PARTNER,
    KIND_A,
    KIND_B,
    KIND_D,
    NOTE_ORDER_CHANGE_MONITOR,
    ORIGIN_CONFORMATIONAL,
    ORIGIN_EDIT,
    ORIGIN_HYDROGEN_EVENT,
    ORIGIN_RING_GEOMETRY,
    PARTNER_ACCEPTOR,
    PARTNER_DONOR,
    REASON_ROUTE_REVIEW_REQUIRED,
    REASON_UNCOVERED_EVENT,
    ROLE_DRIVER,
    ROLE_GUARD,
    ROLE_MONITOR,
    ROLE_TARGET_TEST,
    ROLES,
    SCHEMA_COORDINATE_POOL,
    UNIT_ANGSTROM,
    UNIT_DEGREE,
    CoordinatePool,
    EndpointMaterials,
    build_coordinate_pool,
    constraint_rank,
)
from pes2ts_core.generation.planning.registry import route_strategies
from pes2ts_core.utils.hashing import stable_json_dumps

FORBIDDEN = {key.lower() for key in FORBIDDEN_TRUTH_KEYS} | {
    key.lower() for key in FORBIDDEN_EXPORT_KEYS
} | {"endpoint_match", "orientation", "irc_evidence"}


# ---------------------------------------------------------------------------
# Hand-built P0 fixtures (same idiom as test_strategy_registry.py).
# ---------------------------------------------------------------------------
def _node(
    map_id: int, element: str, *, aromatic: bool = False, stereo: str = "CHI_UNSPECIFIED"
) -> AtomNode:
    return AtomNode(
        map_id=map_id,
        element=element,
        isotope=0,
        formal_charge=0,
        radical_electrons=0,
        explicit_H_neighbors=0,
        aromatic=aromatic,
        stereo=stereo,
    )


def _edge(a: int, b: int, order: float, aromatic: bool = False) -> BondEdge:
    lo, hi = (a, b) if a <= b else (b, a)
    if aromatic:
        connection, resolved = "AROMATIC", 1.5
    elif order == 2:
        connection, resolved = "DOUBLE", 2.0
    else:
        connection, resolved = "SINGLE", 1.0
    return BondEdge(
        map_a=lo,
        map_b=hi,
        connection_type=connection,
        bond_order=resolved,
        aromatic=aromatic,
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
    r_aromatic: set[tuple[int, int]] | None = None,
    p_aromatic: set[tuple[int, int]] | None = None,
    r_stereo: Mapping[int, str] | None = None,
    p_stereo: Mapping[int, str] | None = None,
) -> EndpointGraphBundle:
    elements = dict(elements or {})
    r_aromatic = {(min(a, b), max(a, b)) for a, b in (r_aromatic or set())}
    p_aromatic = {(min(a, b), max(a, b)) for a, b in (p_aromatic or set())}
    r_stereo = dict(r_stereo or {})
    p_stereo = dict(p_stereo or {})
    maps = sorted({m for spec in r_specs + p_specs for m in spec[:2]} | set(elements))

    def side(
        specs: list[tuple[int, int, float]],
        aromatic: set[tuple[int, int]],
        stereo: Mapping[int, str],
    ) -> SideGraph:
        arom_atoms = {m for pair in aromatic for m in pair}
        nodes = tuple(
            _node(
                m,
                elements.get(m, "C"),
                aromatic=m in arom_atoms,
                stereo=stereo.get(m, "CHI_UNSPECIFIED"),
            )
            for m in maps
        )
        edges = tuple(
            sorted(
                (
                    _edge(a, b, order, (min(a, b), max(a, b)) in aromatic)
                    for a, b, order in specs
                ),
                key=lambda e: (e.map_a, e.map_b),
            )
        )
        comps = _components(maps, [(e.map_a, e.map_b) for e in edges])
        return SideGraph(nodes=nodes, edges=edges, components=comps)

    r_graph = side(r_specs, r_aromatic, r_stereo)
    p_graph = side(p_specs, p_aromatic, p_stereo)
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
    r_aromatic: set[tuple[int, int]] | None = None,
    p_aromatic: set[tuple[int, int]] | None = None,
    r_stereo: Mapping[int, str] | None = None,
    p_stereo: Mapping[int, str] | None = None,
    endpoint_electronic: Mapping[str, EndpointElectronic] | None = None,
) -> dict[str, Any]:
    bundle = _hand_bundle(
        r_specs,
        p_specs,
        elements=elements,
        r_aromatic=r_aromatic,
        p_aromatic=p_aromatic,
        r_stereo=r_stereo,
        p_stereo=p_stereo,
    )
    aromatic = aromatic_regions_from_bundle(bundle)
    edit_graph = build_reaction_edit_graph(
        bundle, aromatic_regions=aromatic if aromatic else None
    )
    context = build_endpoint_context(
        bundle, edit_graph, endpoint_electronic=endpoint_electronic
    )
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


def _pool(p0: Mapping[str, Any], materials: Any = None) -> CoordinatePool:
    return build_coordinate_pool(
        p0["route"],
        p0["edit_graph"],
        p0["context"],
        p0["coupling"],
        p0["bundle"],
        materials,
    )


def _single_bond_formation() -> dict[str, Any]:
    return _p0([(1, 2, 1)], [(1, 2, 1), (1, 3, 1)], elements={1: "C", 2: "C", 3: "C"})


def _h_transfer_pure() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 4, 1)],
        [(1, 2, 1), (1, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H"},
    )


def _ring_closure_single() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (1, 5, 1)],
    )


def _pure_order_nonaromatic() -> dict[str, Any]:
    return _p0([(1, 2, 1)], [(1, 2, 2)])


def _disjoint_centers() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (3, 4, 1), (5, 6, 1)],
        [(1, 2, 1), (3, 4, 1), (5, 6, 1), (2, 3, 1), (4, 5, 1)],
    )


def _multi_h_coupled() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1)],
        [(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H", 5: "H"},
    )


def _stereo_flip_with_geometry() -> tuple[dict[str, Any], EndpointMaterials]:
    """4-atom chain, stereo flip + materials whose torsion differs R vs P."""
    p0 = _p0(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "C"},
        r_stereo={2: "CHI_TETRAHEDRAL_CW"},
        p_stereo={2: "CHI_TETRAHEDRAL_CCW"},
    )
    # planar vs twisted butane-like torsion on 1-2-3-4.
    materials = EndpointMaterials(
        r_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.5, 0.0, 0.0),
            3: (2.0, 1.5, 0.0),
            4: (3.0, 2.0, 1.4),
        },
        p_coordinates={
            1: (0.0, 0.0, 0.0),
            2: (1.5, 0.0, 0.0),
            3: (2.0, 1.5, 0.0),
            4: (3.0, 2.0, -1.4),
        },
    )
    return p0, materials


def _recs_by_role(pool: CoordinatePool, role: str) -> list[Any]:
    return [c for c in pool.coordinates if c.role == role]


def _find_by_maps(
    pool: CoordinatePool, kind: str, maps: tuple[int, ...], role: str | None = None
) -> list[Any]:
    wanted = set(maps)
    out = []
    for record in pool.coordinates:
        if record.kind != kind or set(record.atom_maps) != wanted:
            continue
        if role is not None and record.role != role:
            continue
        out.append(record)
    return out


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
# Coordinate generation rules.
# ---------------------------------------------------------------------------
def test_fb_edit_yields_b_driver_and_target_test() -> None:
    """Given a single F/B edit, then a B driver + target_test exist on that pair."""
    p0 = _single_bond_formation()
    pool = _pool(p0)

    drivers = _find_by_maps(pool, KIND_B, (1, 3), ROLE_DRIVER)
    assert len(drivers) == 1
    assert drivers[0].units == UNIT_ANGSTROM
    assert drivers[0].origin.source_kind == ORIGIN_EDIT
    assert drivers[0].origin.edit_pair == (1, 3)
    assert drivers[0].origin.edit_kind == "formed"
    assert drivers[0].event_ids  # membership-linked

    tests = _find_by_maps(pool, KIND_B, (1, 3), ROLE_TARGET_TEST)
    assert len(tests) == 1
    assert tests[0].origin.source_kind == ORIGIN_EDIT


def test_h_transfer_yields_separate_partner_distances_and_angle() -> None:
    """H event → donor-H and acceptor-H are SEPARATE B records + ∠DHA + D–A."""
    p0 = _h_transfer_pure()
    pool = _pool(p0)

    donor = _find_by_maps(pool, KIND_B, (2, 4), ROLE_DRIVER)
    acceptor = _find_by_maps(pool, KIND_B, (1, 4), ROLE_DRIVER)
    assert len(donor) == 1
    assert len(acceptor) == 1
    assert donor[0].origin.geometry_kind == GEOM_PARTNER
    assert donor[0].origin.partner_role == PARTNER_DONOR
    assert acceptor[0].origin.geometry_kind == GEOM_PARTNER
    assert acceptor[0].origin.partner_role == PARTNER_ACCEPTOR
    # Distinct records — never one B = d1 − d2.
    assert donor[0].coordinate_id != acceptor[0].coordinate_id

    angles = _find_by_maps(pool, KIND_A, (2, 4, 1), ROLE_DRIVER)
    assert len(angles) == 1
    assert angles[0].units == UNIT_DEGREE
    assert angles[0].origin.geometry_kind == GEOM_ANGLE_DHA

    da = _find_by_maps(pool, KIND_B, (1, 2), ROLE_DRIVER)
    assert any(r.origin.geometry_kind == GEOM_DISTANCE_DA for r in da)

    # No difference-style coordinate exists anywhere in the pool.
    for record in pool.coordinates:
        assert record.origin.geometry_kind != "difference"
        assert "diff" not in (record.origin.geometry_kind or "")


def test_ring_closure_yields_attack_geometry() -> None:
    """Ring closure → B for the closing bond + A or D attack geometry."""
    p0 = _ring_closure_single()
    pool = _pool(p0)

    closing = _find_by_maps(pool, KIND_B, (1, 5), ROLE_DRIVER)
    assert len(closing) == 1
    assert closing[0].origin.edit_kind == "formed"

    attack_d = _find_by_maps(pool, KIND_D, (2, 1, 5, 4), ROLE_DRIVER)
    attack_a = [
        record
        for record in pool.coordinates
        if record.kind == KIND_A
        and record.origin.source_kind == ORIGIN_RING_GEOMETRY
    ]
    assert attack_d or attack_a
    for record in (*attack_d, *attack_a):
        assert record.origin.source_kind == ORIGIN_RING_GEOMETRY
        assert record.origin.geometry_kind in (GEOM_ATTACK_TORSION, GEOM_ATTACK_ANGLE)
        assert record.origin.event_id


def test_pure_order_change_is_monitor_only() -> None:
    """Pure O → monitor-only coordinates; no driver candidates emitted."""
    p0 = _pure_order_nonaromatic()
    pool = _pool(p0)

    assert pool.coordinates
    for record in pool.coordinates:
        assert record.role == ROLE_MONITOR, record
        assert record.kind == KIND_B
        assert record.origin.source_kind == ORIGIN_EDIT
        assert record.origin.edit_kind == "order_changed"
        assert record.origin.note == NOTE_ORDER_CHANGE_MONITOR
        assert record.origin.geometry_kind is None
    assert _recs_by_role(pool, ROLE_DRIVER) == []
    assert _recs_by_role(pool, ROLE_TARGET_TEST) == []
    assert pool.candidates() == ()
    # The O edit still has a monitoring definition.
    assert pool.all_edits_monitored
    monitored_pairs = {
        record.origin.edit_pair
        for record in pool.coordinates
        if record.origin.edit_pair is not None
    }
    assert (1, 2) in monitored_pairs


def test_conformational_difference_yields_dihedral() -> None:
    """Endpoint geometry comparison with Δ torsion → D driver-nominated records."""
    p0, materials = _stereo_flip_with_geometry()
    pool = _pool(p0, materials)

    conformational = [
        record
        for record in pool.coordinates
        if record.origin.source_kind == ORIGIN_CONFORMATIONAL
    ]
    assert conformational
    drivers = [r for r in conformational if r.role == ROLE_DRIVER]
    tests = [r for r in conformational if r.role == ROLE_TARGET_TEST]
    assert drivers
    assert tests  # stereo/config target-test companions (todo 22 surface)
    for record in drivers:
        assert record.kind == KIND_D
        assert record.units == UNIT_DEGREE
        assert record.origin.r_value is not None
        assert record.origin.p_value is not None
        delta = abs(
            (record.origin.r_value - record.origin.p_value + 180.0) % 360.0 - 180.0
        )
        assert delta >= CONFORMATIONAL_DELTA_MIN_DEG


def test_no_materials_yields_no_conformational_records() -> None:
    p0, _materials = _stereo_flip_with_geometry()
    pool = _pool(p0)
    assert not any(
        record.origin.source_kind == ORIGIN_CONFORMATIONAL
        for record in pool.coordinates
    )


# ---------------------------------------------------------------------------
# Event coverage per driver SET.
# ---------------------------------------------------------------------------
def test_uncovered_event_excluded_from_default_but_listed() -> None:
    """Disjoint centres: the remote event is uncovered → non-default, listed."""
    p0 = _disjoint_centers()
    pool = _pool(p0)

    assert pool.candidates()
    uncovered = [
        candidate
        for candidate in pool.candidates()
        if candidate.non_default_reason == REASON_UNCOVERED_EVENT
    ]
    assert uncovered, "expected at least one UNCOVERED_EVENT candidate"
    for candidate in uncovered:
        assert candidate.default_automatic is False
        assert candidate.has_uncovered_event is True
        # Still listed — never silently dropped.
        assert candidate in pool.candidates()
        statuses = {row.coverage for row in candidate.event_coverage}
        assert COVERAGE_UNCOVERED in statuses
    # No default automatic candidate exists for disjoint centres (each driver
    # set leaves the remote event uncovered; no structurally linked combo).
    assert not any(candidate.default_automatic for candidate in pool.candidates())


def test_coupled_monitor_when_shared_centre() -> None:
    """Undriven H event covered via shared centre atom → coupled_monitor."""
    p0 = _multi_h_coupled()
    pool = _pool(p0)

    # Find a candidate whose drivers touch only one of the two H events.
    candidates = pool.candidates()
    assert candidates
    coupled_rows = []
    for candidate in candidates:
        for row in candidate.event_coverage:
            if row.coverage == COVERAGE_COUPLED_MONITOR:
                coupled_rows.append((candidate, row))
    assert coupled_rows, "expected coupled_monitor coverage in multi-H fixture"
    for _candidate, row in coupled_rows:
        assert row.coupling_evidence
        assert any(
            evidence.startswith(EVID_SHARED_ATOM)
            or evidence.startswith(EVID_SAME_HYDROGEN)
            for evidence in row.coupling_evidence
        )


def test_direct_coverage_when_event_driven() -> None:
    p0 = _single_bond_formation()
    pool = _pool(p0)
    default = [c for c in pool.candidates() if c.default_automatic]
    assert default
    for candidate in default:
        assert candidate.n_drivers >= 1
        statuses = {row.coverage for row in candidate.event_coverage}
        assert COVERAGE_DIRECT in statuses
        assert COVERAGE_UNCOVERED not in statuses


def test_coverage_vocabulary_matches_contracts() -> None:
    assert (COVERAGE_DIRECT, COVERAGE_COUPLED_MONITOR, COVERAGE_UNCOVERED) == COVERAGE_KINDS


# ---------------------------------------------------------------------------
# Role discipline.
# ---------------------------------------------------------------------------
def test_all_edits_monitored_in_every_candidate() -> None:
    """Every candidate's driver∪monitor set covers every edit pair."""
    fixtures = [
        _single_bond_formation(),
        _h_transfer_pure(),
        _ring_closure_single(),
        _multi_h_coupled(),
        _disjoint_centers(),
    ]
    for p0 in fixtures:
        pool = _pool(p0)
        edit_pairs = {
            (int(edit.pair[0]), int(edit.pair[1]))
            if int(edit.pair[0]) <= int(edit.pair[1])
            else (int(edit.pair[1]), int(edit.pair[0]))
            for edit in p0["edit_graph"].edits
        }
        assert edit_pairs
        assert pool.all_edits_monitored
        by_id = {record.coordinate_id: record for record in pool.coordinates}
        for candidate in pool.candidates():
            assert candidate.edits_monitored, candidate.candidate_id
            covered: set[tuple[int, int]] = set()
            for cid in (*candidate.driver_coordinate_ids, *candidate.monitor_coordinate_ids):
                record = by_id[cid]
                if record.origin.edit_pair is not None:
                    covered.add(record.origin.edit_pair)
                elif record.kind == KIND_B and len(record.atom_maps) == 2:
                    a, b = record.atom_maps
                    covered.add((a, b) if a <= b else (b, a))
            assert edit_pairs <= covered, (candidate.candidate_id, edit_pairs - covered)


def test_monitor_never_selected_as_driver() -> None:
    """Pool-role monitors never appear in any candidate's driver set."""
    p0 = _pure_order_nonaromatic()
    pool = _pool(p0)
    monitor_ids = {record.coordinate_id for record in pool.coordinates if record.role == ROLE_MONITOR}
    for candidate in pool.candidates():
        assert monitor_ids.isdisjoint(set(candidate.driver_coordinate_ids))


def test_guard_default_empty_and_constraint_rank_counts_guards() -> None:
    """Guards default empty; rank = drivers + guards (monitors never count)."""
    p0 = _single_bond_formation()
    pool = _pool(p0)
    assert _recs_by_role(pool, ROLE_GUARD) == []
    for candidate in pool.candidates():
        assert candidate.guard_coordinate_ids == ()
        assert candidate.constraint_rank == candidate.n_drivers
    assert constraint_rank(1, 0) == 1
    assert constraint_rank(2, 1) == 3
    assert constraint_rank(0, 0) == 0
    with pytest.raises(ValueError, match="CONSTRAINT_RANK_NEGATIVE"):
        constraint_rank(-1, 0)


def test_route_review_blocks_default_automatic() -> None:
    """review_required route candidates never become default automatic."""
    p0 = _multi_h_coupled()  # H_TRANSFER_COUPLED → review_required=True
    pool = _pool(p0)
    assert pool.candidates()
    for candidate in pool.candidates():
        if candidate.route_review_required:
            assert candidate.default_automatic is False
            assert candidate.non_default_reason in (
                REASON_ROUTE_REVIEW_REQUIRED,
                REASON_UNCOVERED_EVENT,
            )


def test_roles_and_kinds_vocabulary() -> None:
    p0 = _ring_closure_single()
    pool = _pool(p0)
    assert pool.schema_version == SCHEMA_COORDINATE_POOL
    for record in pool.coordinates:
        assert record.role in ROLES
        assert record.kind in (KIND_B, KIND_A, KIND_D)
        if record.kind == KIND_B:
            assert record.units == UNIT_ANGSTROM
            assert len(record.atom_maps) == 2
        else:
            assert record.units == UNIT_DEGREE
            assert len(record.atom_maps) == (3 if record.kind == KIND_A else 4)
        assert len(set(record.atom_maps)) == len(record.atom_maps)


# ---------------------------------------------------------------------------
# Determinism + purity.
# ---------------------------------------------------------------------------
def test_determinism_stable_ordering_and_serialization() -> None:
    p0 = _multi_h_coupled()
    first = _pool(p0)
    second = _pool(p0)
    assert first.to_json() == second.to_json()
    assert [c.coordinate_id for c in first.coordinates] == [
        c.coordinate_id for c in second.coordinates
    ]
    assert [c.candidate_id for c in first.candidates()] == [
        c.candidate_id for c in second.candidates()
    ]
    # Ordering: atom_maps ascending, then kind rank B<A<D.
    kind_rank = {KIND_B: 0, KIND_A: 1, KIND_D: 2}
    keys = [(c.atom_maps, kind_rank[c.kind]) for c in first.coordinates]
    assert keys == sorted(keys)
    # Candidate driver-id tuples sorted.
    driver_tuples = [c.driver_coordinate_ids for c in first.candidates()]
    assert driver_tuples == sorted(driver_tuples)


def test_to_doc_has_no_forbidden_truth_keys() -> None:
    p0 = _ring_closure_single()
    pool = _pool(p0)
    assert _scan_forbidden(pool.to_doc()) == []
    # Field-name scan (same discipline as todo-9/11 tests).
    for record in pool.coordinates:
        for field in dataclasses.fields(record):
            assert field.name.lower() not in FORBIDDEN
        for field in dataclasses.fields(record.origin):
            assert field.name.lower() not in FORBIDDEN


def test_hh_event_generates_hh_distance() -> None:
    """H2 event → H–H B distance + related X–H partner distances."""
    p0 = _p0(
        [(1, 3, 1), (2, 4, 1)],
        [(3, 4, 1)],
        elements={1: "C", 2: "C", 3: "H", 4: "H"},
    )
    pool = _pool(p0)
    hh = _find_by_maps(pool, KIND_B, (3, 4), ROLE_DRIVER)
    assert hh
    assert any(r.origin.geometry_kind in ("hh_distance", GEOM_PARTNER) for r in hh)
    xh = _find_by_maps(pool, KIND_B, (1, 3), ROLE_DRIVER)
    assert xh
    xh2 = _find_by_maps(pool, KIND_B, (2, 4), ROLE_DRIVER)
    assert xh2


def test_pool_helpers_surface() -> None:
    p0 = _single_bond_formation()
    pool = _pool(p0)
    assert pool.reaction_id is None
    roles_seen = {record.role for record in pool.coordinates}
    assert roles_seen <= set(ROLES)
    if pool.coordinates:
        record = pool.coordinates[0]
        assert pool.coordinate(record.coordinate_id) == record
    with pytest.raises(KeyError):
        pool.coordinate("coord-9999")
    with pytest.raises(ValueError, match="UNKNOWN_COORDINATE_ROLE"):
        pool.coordinates_by_role("driverz")
    doc = pool.to_doc()
    assert doc["schema_version"] == SCHEMA_COORDINATE_POOL
    assert stable_json_dumps(doc) == pool.to_json()
