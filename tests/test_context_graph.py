"""Offline tests for the endpoint context graph (g1/endpoint_context.py).

Fixtures are synthetic mapped reaction SMILES plus whitelisted map-keyed
materials (same trust boundary as test_endpoint_graph.py).  Support-path and
cross-component expectations are the exact build_evidence.py semantics; the
evidence cross-check compares against
``outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json``
for six Demo24 records (skip-if-data-missing) and never reads
``data/ground_truth`` or ``g1_v2/edits``.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.g1.endpoint_context import (
    DEFAULT_CONTEXT_RADIUS,
    EndpointElectronic,
    analyze_side_graph,
    build_endpoint_context,
    context_radius_from_config,
    evidence_edit_fields,
)
from pes2ts_core.g1.endpoint_graph import (
    AtomNode,
    BondEdge,
    GraphComponent,
    SideGraph,
    build_endpoint_graph_bundle,
)
from pes2ts_core.g1.parse import parse_reaction
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json"
CANDIDATES_CSV = ROOT / "data/manifests/pes2ts_demo24_candidates_v1.csv"
CASES_ROOT = ROOT / "data/interim/g1_v2/reaction_cases_review_v1"

#: Cyclopropane closure: formed (1,3), R-side support path 1-2-3.
RING_CLOSE = "[C:1][C:2][C:3]>>[C:1]1[C:2][C:3]1"
#: Cyclopropane opening: broken (1,3), P-side support path 1-2-3.
RING_OPEN = "[C:1]1[C:2][C:3]1>>[C:1][C:2][C:3]"
#: Two disjoint new bonds close ONE cyclobutane ring.
TWO_BONDS_ONE_RING = "[C:1][C:2].[C:3][C:4]>>[C:1]1[C:2][C:3][C:4]1"
#: Two disjoint new bonds close TWO independent cyclopropane rings.
TWO_INDEPENDENT_RINGS = (
    "[C:1][C:2][C:3].[C:4][C:5][C:6]>>[C:1]1[C:2][C:3]1.[C:4]1[C:5][C:6]1"
)
#: Two disjoint new bonds on one chain; R-support paths overlap on 3-4-5.
PATH_OVERLAP = (
    "[C:1][C:2][C:3][C:4][C:5][C:6][C:7][C:8]"
    ">>[C:1]1[C:2][C:3]2[C:4][C:5]1[C:6][C:7]2[C:8]"
)
#: Cross-component formed bond (two ethyl fragments → butane).
CROSS_FORMED = "[C:1][C:2].[C:3][C:4]>>[C:1][C:2][C:3][C:4]"
#: Cross-component broken bond (butane → two ethyl fragments).
BROKEN_CROSS = "[C:1][C:2][C:3][C:4]>>[C:1][C:2].[C:3][C:4]"
#: Order change on the organic component; HCl is a graph-invariant spectator.
SPECTATOR = (
    "[C:1][C:2][C:3][C:4].[H:5][Cl:6]>>[C:1]=[C:2][C:3][C:4].[H:5][Cl:6]"
)
#: Toluene (no edits): aromatic region = ring atoms, boundary = methyl carbon.
TOLUENE = (
    "[CH3:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    ">>[CH3:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
)
#: Benzene topology fixture.
BENZENE = (
    "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1"
    ">>[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1"
)
#: Two cyclohexanes joined by one bridge bond (6,7).
BICYCLIC = (
    "[C:1]1[C:2][C:3][C:4][C:5][C:6]1[C:7]2[C:8][C:9][C:10][C:11][C:12]2"
    ">>[C:1]1[C:2][C:3][C:4][C:5][C:6]1[C:7]2[C:8][C:9][C:10][C:11][C:12]2"
)
#: Charged-atom endpoints (methylammonium → methylamine).
CHARGED = "[CH3:1][NH3+:2]>>[CH3:1][NH2:2]"
#: Long-chain cross-component formed bond (seeds sparse enough for shells).
LONG_CROSS = (
    "[C:1][C:2][C:3][C:4].[C:5][C:6][C:7][C:8]"
    ">>[C:1][C:2][C:3][C:4][C:5][C:6][C:7][C:8]"
)
EVIDENCE_RECORD_IDS = (
    "RXN_0000007104",
    "RXN_0000161724",
    "RXN_0000077619",
    "RXN_0000026256",
    "RXN_0000102998",
    "RXN_0000100736",
)


@dataclass(frozen=True, slots=True)
class FakeEdit:
    """Duck-typed todo-7 edit record (attribute access)."""

    pair: tuple[int, int]
    edit_kind: str
    r_bond_order: float | None
    p_bond_order: float | None


@dataclass(frozen=True, slots=True)
class FakeEditGraph:
    """Duck-typed todo-7 EditGraph (.edits + .atom_events)."""

    edits: tuple[FakeEdit, ...]
    atom_events: tuple[Any, ...] = ()


def _materials_for(smiles: str) -> dict[str, dict[int, dict[str, object]]]:
    """Map-keyed materials whose coordinates depend only on map id."""
    reactants, products = parse_reaction(smiles)
    materials: dict[str, dict[int, dict[str, object]]] = {"r": {}, "p": {}}
    for side_key, side in (("r", reactants), ("p", products)):
        for mol, map_list in zip(side.mols, side.map_lists, strict=True):
            for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True):
                map_id = int(map_number)
                materials[side_key][map_id] = {
                    "element": atom.GetSymbol(),
                    "coordinates": [
                        float(map_id),
                        0.5 * map_id,
                        -0.25 * map_id,
                    ],
                }
    return materials


def _bond_orders(graph: SideGraph) -> dict[tuple[int, int], float]:
    """Unordered map-pair → bond order of one side graph."""
    return {(e.map_a, e.map_b): e.bond_order for e in graph.edges}


def _diff_edits(bundle: Any) -> tuple[FakeEdit, ...]:
    """Exclusive F/B/O edits from the bundle graphs (build_evidence diffs)."""
    r_orders = _bond_orders(bundle.r_graph)
    p_orders = _bond_orders(bundle.p_graph)
    edits: list[FakeEdit] = []
    for pair in sorted(set(r_orders) | set(p_orders)):
        r_order = r_orders.get(pair)
        p_order = p_orders.get(pair)
        if r_order is None and p_order is not None:
            kind = "formed"
        elif r_order is not None and p_order is None:
            kind = "broken"
        elif r_order != p_order:
            kind = "order_changed"
        else:
            continue
        edits.append(FakeEdit(pair=pair, edit_kind=kind, r_bond_order=r_order, p_bond_order=p_order))
    return tuple(edits)


def _coords(smiles: str) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]]]:
    """Deterministic R/P coordinates from the materials pattern."""
    materials = _materials_for(smiles)
    r_coords = {m: tuple(row["coordinates"]) for m, row in materials["r"].items()}  # type: ignore[arg-type]
    p_coords = {m: tuple(row["coordinates"]) for m, row in materials["p"].items()}  # type: ignore[arg-type]
    return r_coords, p_coords


def _bundle(smiles: str) -> Any:
    """Build the endpoint graph bundle for a synthetic fixture."""
    return build_endpoint_graph_bundle(smiles, _materials_for(smiles))


def _context(
    smiles: str,
    edits: tuple[FakeEdit, ...] | None = None,
    *,
    radius: int | None = None,
    electronic: Mapping[str, EndpointElectronic] | None = None,
    with_coords: bool = True,
) -> Any:
    """Build bundle + context for a fixture."""
    bundle = _bundle(smiles)
    edit_graph = FakeEditGraph(edits=edits if edits is not None else _diff_edits(bundle))
    r_coords, p_coords = _coords(smiles)
    return build_endpoint_context(
        bundle,
        edit_graph,
        r_coordinates=r_coords if with_coords else None,
        p_coordinates=p_coords if with_coords else None,
        endpoint_electronic=electronic,
        context_radius=radius,
    )


def _manual_side_graph(
    node_ids: list[int], edge_pairs: list[tuple[int, int]]
) -> SideGraph:
    """Hand-built SideGraph for exact topology math assertions."""
    nodes = tuple(
        AtomNode(
            map_id=m,
            element="C",
            isotope=0,
            formal_charge=0,
            radical_electrons=0,
            explicit_H_neighbors=0,
            aromatic=False,
            stereo="",
        )
        for m in sorted(node_ids)
    )
    edges = tuple(
        BondEdge(
            map_a=min(a, b),
            map_b=max(a, b),
            connection_type="SINGLE",
            bond_order=1.0,
            aromatic=False,
            stereo="",
        )
        for a, b in sorted((min(a, b), max(a, b)) for a, b in edge_pairs)
    )
    parent = {m: m for m in node_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in ((e.map_a, e.map_b) for e in edges):
        parent[find(b)] = find(a)
    groups: dict[int, list[int]] = {}
    for m in node_ids:
        groups.setdefault(find(m), []).append(m)
    components = tuple(
        GraphComponent(i, tuple(sorted(g)))
        for i, g in enumerate(sorted(groups.values()))
    )
    return SideGraph(nodes=nodes, edges=edges, components=components)


def test_support_path_R_when_formed_edit_closes_ring() -> None:
    # Given the cyclopropane closure fixture with its formed edit
    ctx = _context(RING_CLOSE, (FakeEdit((1, 3), "formed", None, 1.0),))

    # When reading the R-side support path of the formed edit
    edit = ctx.edits[0]

    # Then the unchanged R path between the endpoints is recorded exactly
    assert edit.edit_kind == "formed"
    assert edit.support_path_R == (1, 2, 3)
    assert edit.support_path_P is None
    assert edit.cross_component_R is False
    assert edit.cross_component_P is False
    assert evidence_edit_fields(edit)["support_path_R"] == [1, 2, 3]
    assert "support_path_P" not in evidence_edit_fields(edit)


def test_support_path_P_when_broken_edit_opens_ring() -> None:
    # Given the cyclopropane opening fixture with its broken edit
    ctx = _context(RING_OPEN, (FakeEdit((1, 3), "broken", 1.0, None),))

    # When reading the P-side remaining path of the broken edit
    edit = ctx.edits[0]

    # Then the P path between the two atoms is recorded exactly
    assert edit.edit_kind == "broken"
    assert edit.support_path_P == (1, 2, 3)
    assert edit.support_path_R is None
    fields = evidence_edit_fields(edit)
    assert fields["support_path_P"] == [1, 2, 3]
    assert "support_path_R" not in fields


def test_order_changed_edit_carries_neither_support_path() -> None:
    # Given an order-change-only edit (SPECTATOR fixture organic bond)
    ctx = _context(SPECTATOR)

    # When projecting evidence fields
    fields = evidence_edit_fields(ctx.edits[0])

    # Then neither support-path key is present (build_evidence semantics)
    assert ctx.edits[0].edit_kind == "order_changed"
    assert "support_path_R" not in fields
    assert "support_path_P" not in fields
    assert fields["cross_component_R"] is False


def test_two_disjoint_new_bonds_share_one_ring_group() -> None:
    # Given two disjoint formed bonds that jointly close one cyclobutane
    ctx = _context(TWO_BONDS_ONE_RING)

    # When grouping formed edits by ring evidence
    # Then both bonds land in ONE ring group (plan failure scenario)
    assert len(ctx.edits) == 2
    assert all(edit.edit_kind == "formed" for edit in ctx.edits)
    assert all(edit.ring_group_id is not None for edit in ctx.edits)
    assert ctx.edits[0].ring_group_id == ctx.edits[1].ring_group_id
    assert len(ctx.ring_groups) == 1
    assert ctx.ring_groups[0].basis == "shared_ring_region"
    assert ctx.ring_groups[0].edit_pairs == ((1, 4), (2, 3))


def test_two_independent_ring_closures_get_separate_groups() -> None:
    # Given two disjoint new bonds closing two independent rings
    ctx = _context(TWO_INDEPENDENT_RINGS)

    # When grouping by ring evidence
    # Then the two closures stay in distinct ring groups
    assert len(ctx.edits) == 2
    assert ctx.edits[0].ring_group_id != ctx.edits[1].ring_group_id
    assert len(ctx.ring_groups) == 2


def test_support_path_overlap_links_disjoint_formed_bonds() -> None:
    # Given two disjoint formed bonds whose R-support paths share atoms 3-4-5
    ctx = _context(PATH_OVERLAP)

    # When reading support paths and ring grouping
    # Then the shared path links them into one ring event
    by_pair = {edit.pair: edit for edit in ctx.edits}
    assert by_pair[(1, 5)].support_path_R == (1, 2, 3, 4, 5)
    assert by_pair[(3, 7)].support_path_R == (3, 4, 5, 6, 7)
    assert by_pair[(1, 5)].ring_group_id == by_pair[(3, 7)].ring_group_id
    assert len(ctx.ring_groups) == 1
    assert "support_path_overlap" in ctx.ring_groups[0].basis


def test_cross_component_flags_when_bond_crosses_components() -> None:
    # Given a formed bond whose R-side atoms sit in different components
    ctx = _context(CROSS_FORMED)

    # Then the R flag is true, the P flag false, and the R path is null
    edit = ctx.edits[0]
    assert edit.pair == (2, 3)
    assert edit.cross_component_R is True
    assert edit.cross_component_P is False
    assert edit.same_component_R is False
    assert edit.same_component_P is True
    assert edit.support_path_R is None
    fields = evidence_edit_fields(edit)
    assert fields["support_path_R"] is None
    assert fields["cross_component_R"] is True
    assert fields["cross_component_P"] is False


def test_cross_component_flag_when_bond_breaks_across_components() -> None:
    # Given a broken bond whose P-side atoms sit in different components
    ctx = _context(BROKEN_CROSS)

    # Then the P flag is true and the P support path is null
    edit = ctx.edits[0]
    assert edit.pair == (2, 3)
    assert edit.cross_component_P is True
    assert edit.cross_component_R is False
    assert edit.support_path_P is None


def test_cycle_rank_and_biconnected_math_on_manual_graph() -> None:
    # Given a triangle (1-2-3) with a pendant atom 4 on atom 1
    graph = _manual_side_graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 1), (1, 4)])

    # When analyzing the side topology
    topology = analyze_side_graph(graph)

    # Then cycle rank, bridges, blocks, and ring membership are exact
    assert topology.n_atoms == 4
    assert topology.n_edges == 4
    assert topology.n_components == 1
    assert topology.cycle_rank == 1
    assert topology.cycle_rank_per_component == ((0, 1),)
    assert topology.bridges == ((1, 4),)
    assert topology.biconnected_blocks == ((1, 2, 3), (1, 4))
    assert topology.two_edge_components == ((1, 2, 3), (4,))
    by_map = {a.map_id: a for a in topology.atoms}
    assert by_map[1].in_cycle is True
    assert by_map[2].in_cycle is True
    assert by_map[3].in_cycle is True
    assert by_map[4].in_cycle is False
    assert by_map[2].ring_sizes == (3,)
    assert by_map[4].ring_sizes == ()
    assert by_map[1].biconnected_block_ids == (0, 1)


def test_cycle_rank_on_bicyclic_fixture() -> None:
    # Given two cyclohexanes joined by a single bridge bond
    ctx = _context(BICYCLIC, ())

    # Then total cycle rank is 2; bridge (6,7) is its own size-2 block
    assert ctx.r_topology.cycle_rank == 2
    assert ctx.r_topology.bridges == ((6, 7),)
    assert ctx.r_topology.biconnected_blocks == (
        (1, 2, 3, 4, 5, 6),
        (6, 7),
        (7, 8, 9, 10, 11, 12),
    )
    assert ctx.r_topology.two_edge_components == (
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12),
    )
    by_map = {a.map_id: a for a in ctx.r_topology.atoms}
    assert by_map[6].in_cycle is True
    assert by_map[7].in_cycle is True
    assert by_map[1].ring_sizes == (6,)


def test_benzene_ring_sizes_and_no_bridges() -> None:
    # Given benzene on both sides with no edits
    ctx = _context(BENZENE, ())

    # Then every ring atom has ring size 6, no bridges, rank 1
    topology = ctx.r_topology
    assert topology.cycle_rank == 1
    assert topology.bridges == ()
    assert topology.biconnected_blocks == ((1, 2, 3, 4, 5, 6),)
    for atom in topology.atoms:
        assert atom.in_cycle is True
        assert atom.ring_sizes == (6,)
        assert atom.aromatic is True


def test_cycle_rank_change_when_ring_forms() -> None:
    # Given a ring-closing reaction (chain → cyclopropane)
    ctx = _context(RING_CLOSE)

    # Then cycle rank rises from 0 to 1 (change R-P = -1)
    assert ctx.r_topology.cycle_rank == 0
    assert ctx.p_topology.cycle_rank == 1
    assert ctx.endpoints[0].cycle_rank == 0
    assert ctx.endpoints[1].cycle_rank == 1


def test_context_radius_configurable_without_changing_strong_outputs() -> None:
    # Given the same long-chain cross-formed reaction at radius 1 and 3
    ctx1 = _context(LONG_CROSS, radius=1)
    ctx3 = _context(LONG_CROSS, radius=3)

    # Then strong-coupling-relevant outputs are identical; shells only grow
    assert [evidence_edit_fields(e) for e in ctx1.edits] == [
        evidence_edit_fields(e) for e in ctx3.edits
    ]
    assert [g.edit_pairs for g in ctx1.ring_groups] == [
        g.edit_pairs for g in ctx3.ring_groups
    ]
    assert ctx1.endpoints_evidence_block() == ctx3.endpoints_evidence_block()
    assert ctx1.r_topology.cycle_rank == ctx3.r_topology.cycle_rank
    assert ctx1.bridge_edges_R == ctx3.bridge_edges_R
    shell1 = set(ctx1.shells_R.map_ids)
    shell3 = set(ctx3.shells_R.map_ids)
    assert shell1 == {3, 4, 5, 6}
    assert shell1 < shell3
    assert shell3 == {1, 2, 3, 4, 5, 6, 7, 8}
    assert ctx1.shells_R.radius == 1
    assert ctx3.shells_R.radius == 3


def test_context_radius_from_config_reads_defaults() -> None:
    # Given the shipped defaults.yaml and override dicts
    import yaml

    defaults = yaml.safe_load((ROOT / "config/defaults.yaml").read_text(encoding="utf-8"))

    # Then scan_strategy.context_radius is honored with safe fallbacks
    assert context_radius_from_config(defaults) == DEFAULT_CONTEXT_RADIUS == 2
    assert context_radius_from_config({"scan_strategy": {"context_radius": 3}}) == 3
    assert context_radius_from_config({}) == 2
    assert context_radius_from_config({"scan_strategy": {"context_radius": 0}}) == 2
    assert context_radius_from_config({"scan_strategy": {"context_radius": "x"}}) == 2


def test_aromatic_region_and_boundary_on_toluene() -> None:
    # Given toluene with no edits
    ctx = _context(TOLUENE, ())

    # Then the aromatic region is the ring, the boundary is the methyl carbon
    topology = ctx.r_topology
    assert topology.aromatic_regions == ((2, 3, 4, 5, 6, 7),)
    assert topology.aromatic_atoms == (2, 3, 4, 5, 6, 7)
    assert topology.aromatic_boundary == (1,)
    assert topology.aromatic_periphery == (2,)
    by_map = {a.map_id: a for a in topology.atoms}
    assert by_map[2].aromatic_region_id == 0
    assert by_map[1].aromatic_region_id is None
    assert by_map[1].aromatic is False


def test_spectator_components_when_one_component_has_no_edits() -> None:
    # Given an order-change reaction with an untouched HCl component
    ctx = _context(SPECTATOR)

    # Then HCl is listed as a spectator on both sides and the organic is not
    assert ctx.edit_atoms == (1, 2)
    assert ctx.spectator_components_R == ((5, 6),)
    assert ctx.spectator_components_P == ((5, 6),)
    assert (1, 2, 3, 4) not in ctx.spectator_components_R
    edit = ctx.edits[0]
    assert edit.same_component_R is True
    assert edit.same_component_P is True


def test_distances_from_endpoint_coordinates() -> None:
    # Given hand-placed coordinates for the ring-close fixture
    bundle = _bundle(RING_CLOSE)
    r_coords = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (3.0, 4.0, 0.0)}
    p_coords = {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (0.0, 1.0, 0.0)}
    edit_graph = FakeEditGraph(edits=(FakeEdit((1, 3), "formed", None, 1.0),))

    # When the context computes per-edit endpoint distances
    ctx = build_endpoint_context(
        bundle, edit_graph, r_coordinates=r_coords, p_coordinates=p_coords
    )

    # Then distances match build_evidence.py's round(math.dist, 6)
    expected_r = round(math.dist(r_coords[1], r_coords[3]), 6)
    expected_p = round(math.dist(p_coords[1], p_coords[3]), 6)
    assert expected_r == 5.0
    assert expected_p == 1.0
    assert ctx.edits[0].distance_R_A == expected_r
    assert ctx.edits[0].distance_P_A == expected_p
    fields = evidence_edit_fields(ctx.edits[0])
    assert fields["distance_R_A"] == expected_r
    assert fields["distance_P_A"] == expected_p


def test_endpoints_features_block_matches_evidence_keys() -> None:
    # Given charged endpoints with case-level charge/multiplicity
    electronic = {
        "reactant": EndpointElectronic(charge=1, multiplicity=1),
        "product": EndpointElectronic(charge=0, multiplicity=1),
    }
    ctx = _context(CHARGED, (), electronic=electronic)

    # Then the evidence endpoints block carries all seven keys per side
    block = ctx.endpoints_evidence_block()
    assert set(block) == {"reactant", "product"}
    expected_keys = {
        "n_components",
        "cycle_rank",
        "charge",
        "multiplicity",
        "rdkit_formal_charge_sum",
        "rdkit_charged_atom_count",
        "rdkit_radical_electron_count",
    }
    assert set(block["reactant"]) == expected_keys
    assert block["reactant"]["rdkit_formal_charge_sum"] == 1
    assert block["reactant"]["rdkit_charged_atom_count"] == 1
    assert block["product"]["rdkit_formal_charge_sum"] == 0
    assert block["product"]["rdkit_charged_atom_count"] == 0
    assert block["reactant"]["charge"] == 1
    assert block["product"]["multiplicity"] == 1


def test_bridge_edges_flag_cross_side_component_structure() -> None:
    # Given the broken-across-components fixture
    ctx = _context(BROKEN_CROSS)

    # Then the R edge (2,3) connects atoms that differ in P components
    assert ctx.edits[0].pair == (2, 3)
    assert ctx.bridge_edges_R == ((2, 3),)
    # P graph has no edge (2,3) anymore, so bridge_edges_P is empty
    assert ctx.bridge_edges_P == ()


def test_build_is_deterministic() -> None:
    # Given the same inputs built twice
    first = _context(TWO_BONDS_ONE_RING)
    second = _context(TWO_BONDS_ONE_RING)

    # Then every exposed projection is byte-identical
    assert first.to_doc() == second.to_doc()


def test_output_contains_no_truth_derived_keys() -> None:
    # Given a fully populated context document
    doc = _context(SPECTATOR).to_doc()
    forbidden = {k.lower() for k in FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS}

    # When recursively scanning keys
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                found.add(str(key).lower())
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(doc)

    # Then no truth-derived key appears
    assert found & forbidden == set()


def _load_case_materials(case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]], dict[str, EndpointElectronic]]:
    """Whitelist-load materials/coords/electronic from a review case doc."""
    atoms = case["atoms"]
    materials: dict[str, Any] = {"r": {}, "p": {}}
    coords: dict[str, dict[int, tuple[float, float, float]]] = {"r": {}, "p": {}}
    for side_key, case_key in (("r", "reactant"), ("p", "product")):
        geometry = case[case_key]["geometry"]
        for index, atom in enumerate(atoms):
            map_id = int(atom["atom_map_id"])
            row = (float(geometry[index][0]), float(geometry[index][1]), float(geometry[index][2]))
            materials[side_key][map_id] = {"element": atom["element"], "coordinates": list(row)}
            coords[side_key][map_id] = row
    electronic = {
        "reactant": EndpointElectronic(
            charge=int(case["reactant"]["charge"]),
            multiplicity=int(case["reactant"]["multiplicity"]),
        ),
        "product": EndpointElectronic(
            charge=int(case["product"]["charge"]),
            multiplicity=int(case["product"]["multiplicity"]),
        ),
    }
    return materials, coords["r"], coords["p"], electronic


def test_evidence_cross_check_support_paths_cross_components_endpoints() -> None:
    # Given the frozen Demo24 evidence + whitelisted case inputs
    if not EVIDENCE_PATH.exists() or not CANDIDATES_CSV.exists():
        pytest.skip("demo24 evidence or candidates CSV missing")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    records = {r["reaction_id"]: r for r in evidence["records"]}
    rows = {
        r["reaction_id"]: r
        for r in csv.DictReader(CANDIDATES_CSV.open(encoding="utf-8-sig", newline=""))
    }

    # When rebuilding context for each available target record
    checked = 0
    for rid in EVIDENCE_RECORD_IDS:
        record = records.get(rid)
        row = rows.get(rid)
        if record is None or row is None:
            continue
        case_path = CASES_ROOT / row["split"] / f"{rid}.json"
        if not case_path.exists():
            continue
        case = json.loads(case_path.read_text(encoding="utf-8"))
        materials, r_coords, p_coords, electronic = _load_case_materials(case)
        bundle = build_endpoint_graph_bundle(row["reaction_smiles"], materials)
        fake_edits = tuple(
            FakeEdit(
                pair=(int(e["pair"][0]), int(e["pair"][1])),
                edit_kind=str(e["edit_kind"]),
                r_bond_order=None if e.get("r_bond_order") is None else float(e["r_bond_order"]),
                p_bond_order=None if e.get("p_bond_order") is None else float(e["p_bond_order"]),
            )
            for e in record["edits"]
        )
        ctx = build_endpoint_context(
            bundle,
            FakeEditGraph(edits=fake_edits),
            r_coordinates=r_coords,
            p_coordinates=p_coords,
            endpoint_electronic=electronic,
        )

        # Then support paths, cross-component flags, distances, and the
        # endpoints block reproduce the evidence fields exactly
        assert len(ctx.edits) == len(record["edits"])
        for mine, theirs in zip(ctx.edits, record["edits"], strict=True):
            fields = evidence_edit_fields(mine)
            assert fields["cross_component_R"] == theirs["cross_component_R"], rid
            assert fields["cross_component_P"] == theirs["cross_component_P"], rid
            assert list(mine.pair) == list(theirs["pair"]), rid
            assert mine.edit_kind == theirs["edit_kind"], rid
            if mine.edit_kind == "formed":
                expected = theirs.get("support_path_R")
                actual = fields.get("support_path_R")
                assert actual == (None if expected is None else list(expected)), rid
            if mine.edit_kind == "broken":
                expected = theirs.get("support_path_P")
                actual = fields.get("support_path_P")
                assert actual == (None if expected is None else list(expected)), rid
            if "distance_R_A" in theirs:
                assert fields["distance_R_A"] == theirs["distance_R_A"], rid
            if "distance_P_A" in theirs:
                assert fields["distance_P_A"] == theirs["distance_P_A"], rid
        block = ctx.endpoints_evidence_block()
        expected_endpoints = record["features"]["endpoints"]
        for side in ("reactant", "product"):
            for key, value in expected_endpoints[side].items():
                assert block[side][key] == value, f"{rid} {side}.{key}"
        checked += 1

    # Then at least three records were actually cross-checked
    if checked < 3:
        pytest.skip(f"only {checked} evidence records available with full inputs")
    assert checked >= 3


def test_evidence_cross_check_ring_case_rxn_102998() -> None:
    # Given the RING_COUPLED record with two disjoint formed bonds
    if not EVIDENCE_PATH.exists() or not CANDIDATES_CSV.exists():
        pytest.skip("demo24 evidence or candidates CSV missing")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    records = {r["reaction_id"]: r for r in evidence["records"]}
    rid = "RXN_0000102998"
    record = records.get(rid)
    rows = {
        r["reaction_id"]: r
        for r in csv.DictReader(CANDIDATES_CSV.open(encoding="utf-8-sig", newline=""))
    }
    row = rows.get(rid)
    if record is None or row is None:
        pytest.skip("RXN_0000102998 missing from evidence or CSV")
    case_path = CASES_ROOT / row["split"] / f"{rid}.json"
    if not case_path.exists():
        pytest.skip("RXN_0000102998 case doc missing")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    materials, r_coords, p_coords, electronic = _load_case_materials(case)
    bundle = build_endpoint_graph_bundle(row["reaction_smiles"], materials)
    fake_edits = tuple(
        FakeEdit(
            pair=(int(e["pair"][0]), int(e["pair"][1])),
            edit_kind=str(e["edit_kind"]),
            r_bond_order=None if e.get("r_bond_order") is None else float(e["r_bond_order"]),
            p_bond_order=None if e.get("p_bond_order") is None else float(e["p_bond_order"]),
        )
        for e in record["edits"]
    )
    ctx = build_endpoint_context(
        bundle,
        FakeEditGraph(edits=fake_edits),
        r_coordinates=r_coords,
        p_coordinates=p_coords,
        endpoint_electronic=electronic,
    )

    # Then the two disjoint formed bonds reproduce their evidence support
    # paths and the broken bond its P-side path
    by_pair = {edit.pair: edit for edit in ctx.edits}
    assert by_pair[(2, 7)].support_path_R == (2, 4, 7)
    assert by_pair[(4, 8)].support_path_R == (4, 7, 8)
    assert by_pair[(2, 4)].support_path_P == (2, 7, 4)
    for mine, theirs in zip(ctx.edits, record["edits"], strict=True):
        fields = evidence_edit_fields(mine)
        assert fields["cross_component_R"] == theirs["cross_component_R"]
        assert fields["cross_component_P"] == theirs["cross_component_P"]
