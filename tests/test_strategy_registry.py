"""Routing tests for the todo-11 strategy registry (design §5.1/§5.2/§12)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from rdkit import rdBase

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
    EVENT_CONNECTIVITY_EXCHANGE,
    EVENT_DELOCALIZED,
    EVENT_H2,
    EVENT_H_TRANSFER,
    EVENT_RING,
    aromatic_regions_from_bundle,
    build_event_coupling_graph,
)
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.registry import (
    ORGANIC_ELEMENTS,
    OUTCOME_EXECUTABLE_CANDIDATE_SET,
    OUTCOME_NEEDS_REVIEW,
    OUTCOME_SPECIAL_DOMAIN_EXIT,
    OUTCOME_TYPED_REJECTION,
    OUTCOMES,
    R_BRANCH_CONFORMATION_STEREO,
    R_BRANCH_DELOCALIZED_ORDER_ONLY,
    R_BRANCH_O_NETWORK_REVIEW,
    R_HOOK_BUDGET_TODO17_PENDING,
    R_HOOK_DIRECTION_TODO15_PENDING,
    R_HOOK_SCHEDULE_TODO14_PENDING,
    R_MOTIF_AROMATIC_ORDER_ONLY,
    R_MOTIF_CONNECTIVITY_EXCHANGE,
    R_MOTIF_CONFORMATION_STEREO,
    R_MOTIF_H2_EVENT,
    R_MOTIF_H_TRANSFER_COUPLED,
    R_MOTIF_H_TRANSFER_PURE,
    R_MOTIF_LOCAL_CONNECTIVITY_BROKEN,
    R_MOTIF_LOCAL_CONNECTIVITY_FORMED,
    R_MOTIF_RING_CLOSURE,
    R_MOTIF_RING_OPENING,
    R_NO_REACTION_CHANGE,
    R_WATERSHED_SPECIAL_DOMAIN_ELEMENT,
    REJ_ELEMENT_IMBALANCE,
    REJ_NO_REACTION_CHANGE,
    SPEC_DOMAIN,
    SPEC_ELECTRONIC_STATE,
    STRATEGY_AROMATIC_COUPLED,
    STRATEGY_CONNECTIVITY_EXCHANGE,
    STRATEGY_CONFORMATION_STEREO,
    STRATEGY_H2_EVENT,
    STRATEGY_H_TRANSFER,
    STRATEGY_H_TRANSFER_COUPLED,
    STRATEGY_IDS,
    STRATEGY_LOCAL_CONNECTIVITY,
    STRATEGY_MULTI_EVENT_CONNECTED,
    STRATEGY_NETWORK_PATH,
    STRATEGY_REGISTRY,
    STRATEGY_RING_COUPLED,
    STRATUM_NOT_SUPPORTED,
    RouteResult,
    route_strategies,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "p0_demo24"
FORBIDDEN = {key.lower() for key in FORBIDDEN_TRUTH_KEYS} | {
    key.lower() for key in FORBIDDEN_EXPORT_KEYS
} | {"endpoint_match", "orientation", "irc_evidence"}


# ---------------------------------------------------------------------------
# Hand-built P0 fixtures (same idiom as test_event_coupling.py).
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
    return {
        "bundle": bundle,
        "edit_graph": edit_graph,
        "context": context,
        "coupling": coupling,
    }


def _route(p0: Mapping[str, Any], **kwargs: Any) -> RouteResult:
    return route_strategies(
        bundle=p0["bundle"],
        edit_graph=p0["edit_graph"],
        context=p0["context"],
        coupling=p0["coupling"],
        **kwargs,
    )


def _single_bond_formation() -> dict[str, Any]:
    return _p0([(1, 2, 1)], [(1, 2, 1), (1, 3, 1)], elements={1: "C", 2: "C", 3: "C"})


def _single_bond_breakage() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 3, 1)],
        [(1, 2, 1)],
        elements={1: "C", 2: "C", 3: "C"},
    )


def _h_transfer_pure() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 4, 1)],
        [(1, 2, 1), (1, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H"},
    )


def _h2_event() -> dict[str, Any]:
    return _p0(
        [(1, 3, 1), (2, 4, 1)],
        [(3, 4, 1)],
        elements={1: "C", 2: "C", 3: "H", 4: "H"},
    )


def _connectivity_exchange() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 4, 1)],
        [(2, 3, 1), (2, 4, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H"},
    )


def _ring_closure_single() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (1, 5, 1)],
    )


def _ring_opening() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 1, 1)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
    )


def _ring_closure_two_bonds() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (3, 4, 1)],
        [(1, 2, 1), (3, 4, 1), (1, 3, 1), (2, 4, 1)],
    )


def _no_edit_identical() -> dict[str, Any]:
    return _p0([(1, 2, 1), (2, 3, 1)], [(1, 2, 1), (2, 3, 1)])


def _stereo_flip_no_bonds() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 3, 1)],
        [(1, 2, 1), (2, 3, 1)],
        r_stereo={1: "CHI_TETRAHEDRAL_CW"},
        p_stereo={1: "CHI_TETRAHEDRAL_CCW"},
    )


def _pure_order_aromatic() -> dict[str, Any]:
    rings = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
    r_specs = [(a, b, 1.0) for a, b in rings]
    p_specs = [(a, b, 1.0) for a, b in rings]
    p_specs[0] = (1, 2, 2.0)
    return _p0(
        r_specs,
        p_specs,
        r_aromatic=set(rings),
        p_aromatic={(a, b) for a, b in rings if (a, b) != (1, 2)},
    )


def _pure_order_nonaromatic() -> dict[str, Any]:
    return _p0([(1, 2, 1)], [(1, 2, 2)])


def _multi_h_coupled() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1)],
        [(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H", 5: "H"},
    )


def _h_transfer_with_exchange() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (2, 4, 1), (1, 5, 1)],
        [(2, 3, 1), (1, 4, 1), (1, 5, 1)],
        elements={1: "C", 2: "C", 3: "C", 4: "H", 5: "H"},
    )


def _disjoint_centers() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1), (3, 4, 1), (5, 6, 1)],
        [(1, 2, 1), (3, 4, 1), (5, 6, 1), (2, 3, 1), (4, 5, 1)],
    )


def _metal_input() -> dict[str, Any]:
    return _p0([(1, 2, 1)], [(1, 2, 1), (1, 3, 1)], elements={1: "Fe", 2: "C", 3: "C"})


def _electronic_mismatch() -> dict[str, Any]:
    return _p0(
        [(1, 2, 1)],
        [(1, 2, 1), (1, 3, 1)],
        elements={1: "C", 2: "C", 3: "C"},
        endpoint_electronic={
            "reactant": EndpointElectronic(charge=0, multiplicity=1),
            "product": EndpointElectronic(charge=1, multiplicity=2),
        },
    )


# ---------------------------------------------------------------------------
# Registry vocabulary.
# ---------------------------------------------------------------------------
def test_strategy_ids_are_design_5_2_verbatim() -> None:
    assert STRATEGY_IDS == (
        "LOCAL_CONNECTIVITY",
        "H_TRANSFER",
        "CONNECTIVITY_EXCHANGE",
        "H_TRANSFER_COUPLED",
        "RING_COUPLED",
        "H2_EVENT",
        "AROMATIC_COUPLED",
        "MULTI_EVENT_CONNECTED",
        "CONFORMATION_STEREO",
        "NETWORK_PATH",
        "SPECIAL_DOMAIN",
    )
    assert set(STRATEGY_REGISTRY) == set(STRATEGY_IDS)
    for strategy_id in STRATEGY_IDS:
        spec = STRATEGY_REGISTRY[strategy_id]
        assert spec.strategy_id == strategy_id
        assert spec.required_graph_pattern
        assert spec.preferred_drivers
        assert spec.monitoring_targets
        for fallback in spec.escalation_fallback:
            assert fallback in STRATEGY_IDS


def test_organic_element_policy_constant_is_conservative() -> None:
    assert "C" in ORGANIC_ELEMENTS and "H" in ORGANIC_ELEMENTS
    assert "Fe" not in ORGANIC_ELEMENTS
    assert "Cl" in ORGANIC_ELEMENTS


# ---------------------------------------------------------------------------
# §5.1 routing coverage.
# ---------------------------------------------------------------------------
def test_single_bond_formation_routes_local_connectivity() -> None:
    result = _route(_single_bond_formation())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    assert result.strategy_candidates
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_LOCAL_CONNECTIVITY
    assert R_MOTIF_LOCAL_CONNECTIVITY_FORMED in primary.rule_trace
    assert primary.n_drivers_planned == 1
    assert primary.review_required is False


def test_single_bond_breakage_routes_local_connectivity() -> None:
    result = _route(_single_bond_breakage())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_LOCAL_CONNECTIVITY
    assert R_MOTIF_LOCAL_CONNECTIVITY_BROKEN in primary.rule_trace


def test_h_transfer_pure_routes_h_transfer() -> None:
    result = _route(_h_transfer_pure())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_H_TRANSFER
    assert R_MOTIF_H_TRANSFER_PURE in primary.rule_trace
    assert EVENT_H_TRANSFER in primary.labels
    assert primary.n_drivers_planned == 1


def test_h2_event_routes_h2_event() -> None:
    result = _route(_h2_event())
    assert result.outcome in (OUTCOME_EXECUTABLE_CANDIDATE_SET, OUTCOME_NEEDS_REVIEW)
    assert result.strategy_candidates
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_H2_EVENT
    assert R_MOTIF_H2_EVENT in primary.rule_trace
    assert EVENT_H2 in primary.labels
    assert primary.n_drivers_planned == 3


def test_connectivity_exchange_routes_exchange() -> None:
    result = _route(_connectivity_exchange())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_CONNECTIVITY_EXCHANGE
    assert R_MOTIF_CONNECTIVITY_EXCHANGE in primary.rule_trace
    assert primary.n_drivers_planned == 2


def test_ring_closure_single_bond_routes_local_connectivity_with_ring_rule() -> None:
    result = _route(_ring_closure_single())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_LOCAL_CONNECTIVITY
    assert R_MOTIF_RING_CLOSURE in primary.rule_trace
    assert EVENT_RING in primary.labels


def test_ring_opening_routes_local_connectivity() -> None:
    result = _route(_ring_opening())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_LOCAL_CONNECTIVITY
    assert R_MOTIF_RING_OPENING in primary.rule_trace


def test_ring_closure_two_bonds_routes_ring_coupled() -> None:
    result = _route(_ring_closure_two_bonds())
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_RING_COUPLED
    assert R_MOTIF_RING_CLOSURE in primary.rule_trace
    assert EVENT_RING in primary.labels


def test_no_edit_identical_endpoints_rejected_no_reaction_change() -> None:
    result = _route(_no_edit_identical())
    assert result.outcome == OUTCOME_TYPED_REJECTION
    assert result.rejection_code == REJ_NO_REACTION_CHANGE
    assert R_NO_REACTION_CHANGE in result.rule_trace
    assert result.strategy_candidates == ()


def test_stereo_change_without_bond_edit_routes_conformation_stereo() -> None:
    result = _route(_stereo_flip_no_bonds())
    assert result.outcome == OUTCOME_NEEDS_REVIEW
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_CONFORMATION_STEREO
    assert R_BRANCH_CONFORMATION_STEREO in primary.rule_trace
    assert R_MOTIF_CONFORMATION_STEREO in primary.rule_trace


def test_pure_order_change_aromatic_routes_delocalized_strategy() -> None:
    result = _route(_pure_order_aromatic())
    assert result.outcome == OUTCOME_NEEDS_REVIEW
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_AROMATIC_COUPLED
    assert R_BRANCH_DELOCALIZED_ORDER_ONLY in primary.rule_trace
    assert R_MOTIF_AROMATIC_ORDER_ONLY in primary.rule_trace
    assert EVENT_DELOCALIZED in primary.labels


def test_pure_order_change_nonaromatic_routes_review_branch_not_aromatic() -> None:
    result = _route(_pure_order_nonaromatic())
    assert result.outcome == OUTCOME_NEEDS_REVIEW
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_NETWORK_PATH
    assert STRATEGY_AROMATIC_COUPLED not in {
        candidate.strategy_id for candidate in result.strategy_candidates
    }
    assert R_BRANCH_O_NETWORK_REVIEW in primary.rule_trace


def test_multi_h_transfers_coupled_routes_h_transfer_coupled() -> None:
    result = _route(_multi_h_coupled())
    assert result.outcome in (OUTCOME_EXECUTABLE_CANDIDATE_SET, OUTCOME_NEEDS_REVIEW)
    primary = result.strategy_candidates[0]
    assert primary.strategy_id == STRATEGY_H_TRANSFER_COUPLED
    assert R_MOTIF_H_TRANSFER_COUPLED in primary.rule_trace
    assert len(primary.component_events) >= 2
    assert EVENT_H_TRANSFER in primary.labels


def test_h_transfer_with_exchange_preserves_both_labels() -> None:
    result = _route(_h_transfer_with_exchange())
    primary = result.strategy_candidates[0]
    assert primary.strategy_id in {
        STRATEGY_H_TRANSFER_COUPLED,
        STRATEGY_MULTI_EVENT_CONNECTED,
    }
    assert EVENT_H_TRANSFER in primary.labels
    assert EVENT_CONNECTIVITY_EXCHANGE in primary.labels
    assert "R_EVENT_H_TRANSFER" in primary.rule_trace
    assert "R_EVENT_CONNECTIVITY_EXCHANGE" in primary.rule_trace
    assert EVENT_H_TRANSFER in result.labels
    assert EVENT_CONNECTIVITY_EXCHANGE in result.labels


def test_multiple_disjoint_centers_yield_per_component_candidates() -> None:
    result = _route(_disjoint_centers())
    strategies = [candidate.strategy_id for candidate in result.strategy_candidates]
    assert strategies.count(STRATEGY_LOCAL_CONNECTIVITY) == 2
    assert STRATEGY_NETWORK_PATH in strategies
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET


def test_metal_input_special_domain_exit() -> None:
    result = _route(_metal_input())
    assert result.outcome == OUTCOME_SPECIAL_DOMAIN_EXIT
    assert result.rejection_code == SPEC_DOMAIN
    assert result.strategy_candidates == ()
    assert result.special_domain is not None
    assert result.special_domain["code"] == SPEC_DOMAIN
    assert result.special_domain["required_representations"]
    assert result.special_domain["required_methods"]
    assert R_WATERSHED_SPECIAL_DOMAIN_ELEMENT in result.rule_trace


def test_electronic_state_mismatch_special_exit() -> None:
    result = _route(_electronic_mismatch())
    assert result.outcome == OUTCOME_SPECIAL_DOMAIN_EXIT
    assert result.rejection_code == SPEC_ELECTRONIC_STATE
    assert result.special_domain is not None
    assert result.special_domain["code"] == SPEC_ELECTRONIC_STATE


def test_element_imbalance_typed_rejection() -> None:
    p0 = _single_bond_formation()
    conservation = dataclasses.replace(
        p0["bundle"].conservation, element_conserved=False
    )
    bundle = dataclasses.replace(p0["bundle"], conservation=conservation)
    result = route_strategies(
        bundle=bundle,
        edit_graph=p0["edit_graph"],
        context=p0["context"],
        coupling=p0["coupling"],
    )
    assert result.outcome == OUTCOME_TYPED_REJECTION
    assert result.rejection_code == REJ_ELEMENT_IMBALANCE


# ---------------------------------------------------------------------------
# Hard rules: stratum rejection, multi-label, determinism, no silent empty.
# ---------------------------------------------------------------------------
def test_stratum_kwarg_rejected_with_typed_error() -> None:
    p0 = _single_bond_formation()
    with pytest.raises(ValueError, match=STRATUM_NOT_SUPPORTED):
        route_strategies(
            bundle=p0["bundle"],
            edit_graph=p0["edit_graph"],
            context=p0["context"],
            coupling=p0["coupling"],
            stratum="C,H|1|small",
        )


def test_determinism_same_input_identical_serialization() -> None:
    p0 = _h_transfer_with_exchange()
    first = _route(p0, reaction_id="RXN_TEST").to_json()
    second = _route(p0, reaction_id="RXN_TEST").to_json()
    assert first == second
    p0_exchange = _connectivity_exchange()
    assert _route(p0_exchange).to_json() == _route(p0_exchange).to_json()


def test_extension_hooks_recorded_in_trace() -> None:
    result = _route(_single_bond_formation())
    for candidate in result.strategy_candidates:
        assert R_HOOK_DIRECTION_TODO15_PENDING in candidate.rule_trace
        assert R_HOOK_SCHEDULE_TODO14_PENDING in candidate.rule_trace
        assert R_HOOK_BUDGET_TODO17_PENDING in candidate.rule_trace


def test_route_result_to_doc_carries_no_forbidden_keys() -> None:
    for fixture in (
        _single_bond_formation(),
        _h_transfer_with_exchange(),
        _pure_order_aromatic(),
        _stereo_flip_no_bonds(),
    ):
        doc = _route(fixture).to_doc()
        found: set[str] = set()

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    found.add(str(key).lower())
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(doc)
        assert not (found & FORBIDDEN), sorted(found & FORBIDDEN)


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        ("single_bond_formation", _single_bond_formation),
        ("single_bond_breakage", _single_bond_breakage),
        ("h_transfer_pure", _h_transfer_pure),
        ("h2_event", _h2_event),
        ("connectivity_exchange", _connectivity_exchange),
        ("ring_closure_single", _ring_closure_single),
        ("ring_opening", _ring_opening),
        ("ring_closure_two_bonds", _ring_closure_two_bonds),
        ("no_edit_identical", _no_edit_identical),
        ("stereo_flip_no_bonds", _stereo_flip_no_bonds),
        ("pure_order_aromatic", _pure_order_aromatic),
        ("pure_order_nonaromatic", _pure_order_nonaromatic),
        ("multi_h_coupled", _multi_h_coupled),
        ("h_transfer_with_exchange", _h_transfer_with_exchange),
        ("disjoint_centers", _disjoint_centers),
        ("metal_input", _metal_input),
        ("electronic_mismatch", _electronic_mismatch),
    ],
)
def test_every_fixture_gets_exactly_one_disposition(
    name: str, builder: Callable[[], dict[str, Any]]
) -> None:
    result = _route(builder(), reaction_id=name)
    assert result.outcome in OUTCOMES
    if result.outcome in (OUTCOME_EXECUTABLE_CANDIDATE_SET, OUTCOME_NEEDS_REVIEW):
        assert result.strategy_candidates, name
        for candidate in result.strategy_candidates:
            assert candidate.strategy_id in STRATEGY_REGISTRY
            assert candidate.rule_trace
    else:
        assert result.strategy_candidates == (), name
        assert result.rejection_code, name
        if result.outcome == OUTCOME_SPECIAL_DOMAIN_EXIT:
            assert result.special_domain is not None


# ---------------------------------------------------------------------------
# Demo24 integration (optional): every fixture record routes somewhere.
# ---------------------------------------------------------------------------
def _load_manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _rebuild_p0(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    from pes2ts_core.g1.endpoint_context import EndpointElectronic as _EE
    from pes2ts_core.scan_strategy.graph_rebuild import (
        load_endpoint_materials_from_export,
        rebuild_endpoint_graphs,
    )

    materials = load_endpoint_materials_from_export(snapshot)
    bundle = rebuild_endpoint_graphs(str(snapshot["reaction_smiles"]), materials)
    aromatic = aromatic_regions_from_bundle(bundle)
    edit_graph = build_reaction_edit_graph(
        bundle, aromatic_regions=aromatic if aromatic else None
    )
    maps = [int(m) for m in snapshot["maps"]]
    r_coordinates = {
        map_id: tuple(float(x) for x in snapshot["r_coordinates"][i])
        for i, map_id in enumerate(maps)
    }
    p_coordinates = {
        map_id: tuple(float(x) for x in snapshot["p_coordinates"][i])
        for i, map_id in enumerate(maps)
    }
    electronic_raw = snapshot["endpoint_electronic"]
    endpoint_electronic = {
        "reactant": _EE(
            charge=int(electronic_raw["reactant"]["charge"]),
            multiplicity=int(electronic_raw["reactant"]["multiplicity"]),
        ),
        "product": _EE(
            charge=int(electronic_raw["product"]["charge"]),
            multiplicity=int(electronic_raw["product"]["multiplicity"]),
        ),
    }
    context = build_endpoint_context(
        bundle,
        edit_graph,
        r_coordinates=r_coordinates,
        p_coordinates=p_coordinates,
        endpoint_electronic=endpoint_electronic,
    )
    coupling = build_event_coupling_graph(bundle, edit_graph, context)
    return {
        "bundle": bundle,
        "edit_graph": edit_graph,
        "context": context,
        "coupling": coupling,
    }


def test_demo24_every_record_routes_to_exactly_one_disposition() -> None:
    manifest = _load_manifest()
    pinned = str(manifest["rdkit_version"])
    if rdBase.rdkitVersion != pinned:
        pytest.fail(
            f"rdkit {rdBase.rdkitVersion} != fixture pin {pinned} (never-skip gate)"
        )
    records_dir = FIXTURE_ROOT / "records"
    paths = sorted(records_dir.glob("RXN_*.json"))
    assert len(paths) == 24
    for path in paths:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        p0 = _rebuild_p0(snapshot)
        result = _route(p0, reaction_id=str(snapshot.get("reaction_id", path.stem)))
        assert result.outcome in OUTCOMES, path.name
        if result.outcome in (OUTCOME_EXECUTABLE_CANDIDATE_SET, OUTCOME_NEEDS_REVIEW):
            assert result.strategy_candidates, path.name
        else:
            assert result.rejection_code, path.name
