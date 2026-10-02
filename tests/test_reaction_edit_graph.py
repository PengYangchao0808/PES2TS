"""Offline tests for the atom-level reaction edit graph (g1/reaction_edit_graph.py).

Synthetic fixtures only in the primary suite: mapped reaction SMILES plus
whitelisted materials, or hand-built EndpointGraphBundle graphs with exactly
chosen edit topology.  Locks F/B/O mutual exclusivity, component/cycle-rank
math, R/P swap symmetry, atom attribute events, aromatic-region carry-through,
determinism, and output purity.  One gated cross-check re-derives the Demo24
evidence features from the real export tree when it is present (skip-if-missing;
todo 10 owns the frozen-fixture golden).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_graph import (
    AtomNode,
    BondEdge,
    ConservationReport,
    EndpointGraphBundle,
    GeometryRef,
    SideGraph,
    build_endpoint_graph_bundle,
)
from pes2ts_core.g1.parse import parse_reaction
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.graph_rebuild import (
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.utils.hashing import stable_json_dumps

#: Butane -> 2-butene-style order change (bond-order-only edit).
ORDER_CHANGE = "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH2:1]=[CH:2][CH2:3][CH3:4]"
#: H3 migrates O2 -> C1: one broken + one formed bond edit on the H pairs.
H_MIGRATION = "[CH3:1][O:2][H:3]>>[CH2:1]([H:3])[O:2]"
#: Charge-only change, zero bond edits.
ATTR_ONLY = "[NH4+:1]>>[NH3:1]"

FORBIDDEN_KEYS: frozenset[str] = frozenset(FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS)
GOLDEN_EDIT_FIELDS: tuple[str, ...] = (
    "pair",
    "edit_kind",
    "elements",
    "r_bond_order",
    "p_bond_order",
    "aromatic_region",
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json"
CSV_PATH = ROOT / "data/manifests/pes2ts_demo24_candidates_v1.csv"
EXPORT_TREES: tuple[Path, ...] = (
    ROOT / "data/interim/g1_v2/export_contracts_v1",
    ROOT / "data/interim/g1_v2/export",
)


def _materials_for(smiles: str) -> dict[str, dict[int, dict[str, object]]]:
    """Build map-keyed materials whose coordinates depend only on map id."""
    reactants, products = parse_reaction(smiles)
    materials: dict[str, dict[int, dict[str, object]]] = {"r": {}, "p": {}}
    for side_key, side in (("r", reactants), ("p", products)):
        for mol, map_list in zip(side.mols, side.map_lists, strict=True):
            for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True):
                map_id = int(map_number)
                materials[side_key][map_id] = {
                    "element": atom.GetSymbol(),
                    "coordinates": [float(map_id), 0.5 * map_id, -0.25 * map_id],
                }
    return materials


def _bundle_from_smiles(smiles: str) -> EndpointGraphBundle:
    return build_endpoint_graph_bundle(smiles, _materials_for(smiles))


def _node(
    map_id: int,
    element: str = "C",
    *,
    charge: int = 0,
    radical: int = 0,
    aromatic: bool = False,
    stereo: str = "CHI_UNSPECIFIED",
) -> AtomNode:
    return AtomNode(
        map_id=map_id,
        element=element,
        isotope=0,
        formal_charge=charge,
        radical_electrons=radical,
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


def _hand_bundle(
    r_specs: list[tuple[int, int, float]],
    p_specs: list[tuple[int, int, float]],
    *,
    elements: Mapping[int, str] | None = None,
    r_attrs: Mapping[int, Mapping[str, object]] | None = None,
    p_attrs: Mapping[int, Mapping[str, object]] | None = None,
    r_aromatic: set[tuple[int, int]] | None = None,
    p_aromatic: set[tuple[int, int]] | None = None,
) -> EndpointGraphBundle:
    """Assemble a minimal EndpointGraphBundle from chosen R/P edge specs.

    Edge specs are ``(a, b, order)`` unordered pairs; ``r_aromatic``/``p_aromatic``
    mark pairs rendered aromatic (order 1.5).  Per-side attribute overrides let
    atom-only changes be injected without any bond edit.
    """
    elements = dict(elements or {})
    r_attrs = dict(r_attrs or {})
    p_attrs = dict(p_attrs or {})
    r_aromatic = set(r_aromatic or set())
    p_aromatic = set(p_aromatic or set())
    maps = sorted(
        {m for spec in r_specs + p_specs for m in spec[:2]}
        | set(elements)
        | set(r_attrs)
        | set(p_attrs)
    )

    def _side(
        specs: list[tuple[int, int, float]],
        attrs: Mapping[int, Mapping[str, object]],
        aromatic: set[tuple[int, int]],
    ) -> SideGraph:
        nodes = []
        for m in maps:
            override = attrs.get(m, {})
            nodes.append(
                _node(
                    m,
                    str(elements.get(m, "C")),
                    charge=int(override.get("charge", 0)),
                    radical=int(override.get("radical", 0)),
                    stereo=str(override.get("stereo", "CHI_UNSPECIFIED")),
                )
            )
        edges = tuple(
            sorted(
                (_edge(a, b, order, (min(a, b), max(a, b)) in aromatic) for a, b, order in specs),
                key=lambda e: (e.map_a, e.map_b),
            )
        )
        return SideGraph(nodes=tuple(nodes), edges=edges, components=())

    r_graph = _side(r_specs, r_attrs, r_aromatic)
    p_graph = _side(p_specs, p_attrs, p_aromatic)
    geometry = GeometryRef(unit="angstrom", coordinates_sha256="0" * 64, n_points=len(maps))
    conservation = ConservationReport(
        map_ids=tuple(maps),
        element_conserved=True,
        isotope_conserved=True,
        explicit_h_count_r=0,
        explicit_h_count_p=0,
        n_atoms_r=len(maps),
        n_atoms_p=len(maps),
        n_components_r=1,
        n_components_p=1,
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


def _swap_bundle(bundle: EndpointGraphBundle) -> EndpointGraphBundle:
    """Return the bundle with R/P graphs and geometry bindings exchanged."""
    return dataclasses.replace(
        bundle,
        r_graph=bundle.p_graph,
        p_graph=bundle.r_graph,
        r_geometry=bundle.p_geometry,
        p_geometry=bundle.r_geometry,
    )


def _collect_keys(value: object, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            _collect_keys(child, found)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, found)


def _find_export_doc(reaction_id: str) -> Path | None:
    shard = f"{int(reaction_id[4:]) // 1000:05d}"
    for tree in EXPORT_TREES:
        candidate = tree / shard / f"{reaction_id}.json"
        if candidate.exists():
            return candidate
    return None


def test_order_change_only_yields_no_formed_or_broken_records() -> None:
    # Given a reaction whose only edit is a bond-order change
    bundle = _bundle_from_smiles(ORDER_CHANGE)

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then exactly one order_changed record exists and F/B are absent entirely
    assert len(graph.edits) == 1
    edit = graph.edits[0]
    assert edit.pair == (1, 2)
    assert edit.edit_kind == "order_changed"
    assert edit.r_bond_order == 1.0
    assert edit.p_bond_order == 2.0
    assert "formed" not in graph.edit_counts
    assert "broken" not in graph.edit_counts
    assert graph.edit_counts == {"order_changed": 1}
    assert all(e.edit_kind != "formed" and e.edit_kind != "broken" for e in graph.edits)
    assert graph.connectivity_edit_components == ()
    assert graph.edit_components == ((1, 2),)
    assert graph.edit_cycle_rank == 0


def test_h_migration_surfaces_as_broken_and_formed_bond_edits() -> None:
    # Given an explicit-H transfer O2 -> C1
    bundle = _bundle_from_smiles(H_MIGRATION)

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then both partner bonds appear as exclusive F/B edits on the H map
    assert graph.edit_counts == {"formed": 1, "broken": 1}
    by_pair = {e.pair: e for e in graph.edits}
    assert set(by_pair) == {(1, 3), (2, 3)}
    assert by_pair[(1, 3)].edit_kind == "formed"
    assert by_pair[(1, 3)].elements == ("C", "H")
    assert by_pair[(2, 3)].edit_kind == "broken"
    assert by_pair[(2, 3)].elements == ("O", "H")
    # H3's partner set changed, so it is an atom event
    event_maps = {event["atom_map_id"] for event in graph.atom_events}
    assert 3 in event_maps
    h_event = next(e for e in graph.atom_events if e["atom_map_id"] == 3)
    assert h_event["partner_set_changed"] is True
    assert h_event["partners_r"] == [2]
    assert h_event["partners_p"] == [1]
    # Heavy partners 1 and 2 also changed partner sets
    assert {1, 2} <= event_maps


def test_attribute_only_change_produces_no_bond_edits() -> None:
    # Given a charge-only change with identical connectivity
    bundle = _bundle_from_smiles(ATTR_ONLY)

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then no bond edits exist but the attribute change is recorded
    assert graph.edits == ()
    assert graph.edit_counts == {}
    assert graph.edit_components == ()
    assert graph.connectivity_edit_components == ()
    assert graph.edit_cycle_rank == 0
    assert len(graph.atom_attribute_changes) == 1
    change = graph.atom_attribute_changes[0]
    assert change["atom_map_id"] == 1
    assert change["R"]["formal_charge"] == 1
    assert change["P"]["formal_charge"] == 0
    assert change["R"]["radical_electrons"] == change["P"]["radical_electrons"]
    event = graph.atom_events[0]
    assert event["atom_map_id"] == 1
    assert event["partner_set_changed"] is False
    assert event["attributes_changed"] is True


def test_edit_components_and_cycle_rank_on_disjoint_formed_bonds() -> None:
    # Given two disjoint formed bonds (no shared atoms)
    bundle = _hand_bundle([], [(1, 2, 1.0), (3, 4, 1.0)])

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then each bond is its own component and the cycle rank is zero
    assert graph.edit_counts == {"formed": 2}
    assert graph.edit_components == ((1, 2), (3, 4))
    assert graph.connectivity_edit_components == ((1, 2), (3, 4))
    # |E| - |V| + C = 2 - 4 + 2 = 0
    assert graph.edit_cycle_rank == 0
    assert graph.edit_degree == {1: 1, 2: 1, 3: 1, 4: 1}
    assert graph.center_atoms == ()


def test_edit_cycle_rank_counts_triangle_of_order_changes() -> None:
    # Given three order_changed bonds forming a triangle on maps 1-2-3
    bundle = _hand_bundle(
        [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 1.0)],
        [(1, 2, 2.0), (2, 3, 2.0), (1, 3, 2.0)],
    )

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then the single component carries cycle rank |E|-|V|+C = 3-3+1 = 1
    assert graph.edit_counts == {"order_changed": 3}
    assert graph.edit_components == ((1, 2, 3),)
    assert graph.edit_cycle_rank == 1
    # No connectivity (F/B) edits at all
    assert graph.connectivity_edit_components == ()
    # Every triangle atom is a multi-edit centre
    assert graph.center_atoms == (1, 2, 3)
    assert graph.edit_degree == {1: 2, 2: 2, 3: 2}


def test_connectivity_components_split_when_only_order_change_bridges() -> None:
    # Given formed(1,2) + order_changed(2,3) + broken(3,4): one F/B/O component
    # but two F/B-only components bridged solely by the order change
    bundle = _hand_bundle(
        [(2, 3, 1.0), (3, 4, 1.0)],
        [(1, 2, 1.0), (2, 3, 2.0)],
    )

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then F/B/O connectivity merges all four atoms; F/B-only does not
    kinds = {e.pair: e.edit_kind for e in graph.edits}
    assert kinds == {(1, 2): "formed", (2, 3): "order_changed", (3, 4): "broken"}
    assert graph.edit_components == ((1, 2, 3, 4),)
    assert graph.connectivity_edit_components == ((1, 2), (3, 4))
    # |E|-|V|+C over F/B/O = 3 - 4 + 1 = 0
    assert graph.edit_cycle_rank == 0
    # Shared edit atoms 2 and 3 are centres
    assert graph.center_atoms == (2, 3)


def test_shared_center_atoms_and_edit_degree() -> None:
    # Given three formed bonds all incident on atom 1
    bundle = _hand_bundle([], [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)])

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then atom 1 is the unique centre and degrees follow incident edits
    assert graph.edit_degree == {1: 3, 2: 1, 3: 1, 4: 1}
    assert graph.center_atoms == (1,)
    assert graph.edit_components == ((1, 2, 3, 4),)
    assert graph.edit_cycle_rank == 0


def test_rp_swap_exchanges_formed_and_broken_on_hand_built_graph() -> None:
    # Given R carries bond (1,2) and P carries bond (3,4) on disjoint pairs
    bundle = _hand_bundle([(1, 2, 1.0)], [(3, 4, 1.0)])
    forward = build_reaction_edit_graph(bundle)

    # When the R/P graphs are swapped
    backward = build_reaction_edit_graph(_swap_bundle(bundle))

    # Then F/B records exchange pairs while counts and topology stay fixed
    assert forward.edit_counts == {"formed": 1, "broken": 1}
    assert backward.edit_counts == {"formed": 1, "broken": 1}
    forward_kinds = {e.pair: e.edit_kind for e in forward.edits}
    backward_kinds = {e.pair: e.edit_kind for e in backward.edits}
    # R-only bond (1,2) is broken; P-only bond (3,4) is formed — swap mirrors them
    assert forward_kinds == {(1, 2): "broken", (3, 4): "formed"}
    assert backward_kinds == {(1, 2): "formed", (3, 4): "broken"}
    assert forward.edit_components == backward.edit_components == ((1, 2), (3, 4))
    assert forward.edit_cycle_rank == backward.edit_cycle_rank == 0
    # Order-bearing records exchange r/p orders under the swap
    fwd_o = next(e for e in forward.edits if e.pair == (1, 2))
    bwd_o = next(e for e in backward.edits if e.pair == (1, 2))
    assert (fwd_o.r_bond_order, fwd_o.p_bond_order) == (1.0, None)
    assert (bwd_o.r_bond_order, bwd_o.p_bond_order) == (None, 1.0)


def test_rp_swap_on_smiles_fixtures_preserves_order_changed_and_swaps_fb() -> None:
    # Given the order-change fixture and the H-migration fixture
    order_graph = build_reaction_edit_graph(_bundle_from_smiles(ORDER_CHANGE))
    h_graph = build_reaction_edit_graph(_bundle_from_smiles(H_MIGRATION))

    # When each reaction is written in the opposite direction
    order_rev = build_reaction_edit_graph(_bundle_from_smiles(_reverse_smiles(ORDER_CHANGE)))
    h_rev = build_reaction_edit_graph(_bundle_from_smiles(_reverse_smiles(H_MIGRATION)))

    # Then order_changed survives direction reversal untouched
    assert order_graph.edit_counts == order_rev.edit_counts == {"order_changed": 1}
    assert [e.pair for e in order_graph.edits] == [e.pair for e in order_rev.edits]
    fwd = order_graph.edits[0]
    rev = order_rev.edits[0]
    assert (fwd.r_bond_order, fwd.p_bond_order) == (rev.p_bond_order, rev.r_bond_order)
    # And F/B counts mirror while the edited pair set is identical
    assert h_graph.edit_counts == h_rev.edit_counts == {"formed": 1, "broken": 1}
    assert {e.pair for e in h_graph.edits} == {e.pair for e in h_rev.edits}
    h_fwd = {e.pair: e for e in h_graph.edits}
    h_bwd = {e.pair: e for e in h_rev.edits}
    for pair, edit in h_fwd.items():
        other = h_bwd[pair]
        if edit.edit_kind == "formed":
            assert other.edit_kind == "broken"
        elif edit.edit_kind == "broken":
            assert other.edit_kind == "formed"
        assert (edit.r_bond_order, edit.p_bond_order) == (
            other.p_bond_order,
            other.r_bond_order,
        )
    assert h_graph.edit_components == h_rev.edit_components
    assert h_graph.edit_cycle_rank == h_rev.edit_cycle_rank


def _reverse_smiles(smiles: str) -> str:
    left, right = smiles.split(">>")
    return f"{right}>>{left}"


def test_atom_attribute_change_on_heavy_atom_without_bond_edit() -> None:
    # Given a hand-built bundle whose O atom changes charge but keeps bonds
    bundle = _hand_bundle(
        [(1, 2, 1.0)],
        [(1, 2, 1.0)],
        elements={1: "C", 2: "O"},
        r_attrs={2: {"charge": 1}},
        p_attrs={2: {"charge": 0}},
    )

    # When the edit graph is built
    graph = build_reaction_edit_graph(bundle)

    # Then connectivity is unchanged but the attribute event is recorded
    assert graph.edits == ()
    assert len(graph.atom_attribute_changes) == 1
    change = graph.atom_attribute_changes[0]
    assert change["atom_map_id"] == 2
    assert change["R"]["formal_charge"] == 1
    assert change["P"]["formal_charge"] == 0
    # Evidence-schema keys only: charge/radical/aromatic/chiral_tag
    assert set(change["R"]) == {
        "formal_charge",
        "radical_electrons",
        "aromatic",
        "chiral_tag",
    }
    event = next(e for e in graph.atom_events if e["atom_map_id"] == 2)
    assert event["attributes_changed"] is True
    assert event["partner_set_changed"] is False


def test_aromatic_region_carried_through_when_provided() -> None:
    # Given an aromatic order change (single 1.0 -> aromatic 1.5)
    bundle = _hand_bundle(
        [(1, 2, 1.0)],
        [(1, 2, 1.5)],
        r_aromatic=set(),
        p_aromatic={(1, 2)},
    )

    # When regions are supplied in export-doc shape and in pair shape
    export_shape = build_reaction_edit_graph(
        bundle, aromatic_regions={"abc123def456": [[1, 2]]}
    )
    pair_shape = build_reaction_edit_graph(bundle, aromatic_regions={(1, 2): "xyz789"})
    absent = build_reaction_edit_graph(bundle)

    # Then the region id is carried through verbatim, else null
    assert export_shape.edits[0].aromatic_region == "abc123def456"
    assert pair_shape.edits[0].aromatic_region == "xyz789"
    assert absent.edits[0].aromatic_region is None
    # The record still classifies as order_changed with aromatic 1.5 on P
    assert export_shape.edits[0].edit_kind == "order_changed"
    assert export_shape.edits[0].p_bond_order == 1.5


def test_determinism_same_input_identical_serialization() -> None:
    # Given the same SMILES fixture built twice
    first = build_reaction_edit_graph(_bundle_from_smiles(H_MIGRATION))
    second = build_reaction_edit_graph(_bundle_from_smiles(H_MIGRATION))

    # When both documents are serialized stably
    payload_first = stable_json_dumps(first.to_doc())
    payload_second = stable_json_dumps(second.to_doc())

    # Then the serializations are byte-identical
    assert payload_first == payload_second
    # And the hand-built path is deterministic too
    hand = _hand_bundle([(1, 2, 1.0)], [(2, 3, 1.0), (3, 4, 2.0)])
    again = _hand_bundle([(1, 2, 1.0)], [(2, 3, 1.0), (3, 4, 2.0)])
    assert stable_json_dumps(build_reaction_edit_graph(hand).to_doc()) == stable_json_dumps(
        build_reaction_edit_graph(again).to_doc()
    )


def test_edit_records_expose_todo8_extension_seam() -> None:
    # Given a graph with mixed edit kinds
    graph = build_reaction_edit_graph(_bundle_from_smiles(H_MIGRATION))

    # When per-edit records are projected
    records = graph.edit_records()

    # Then each record carries exactly the six golden fields todo 8 extends
    assert len(records) == 2
    for record in records:
        assert set(record) == set(GOLDEN_EDIT_FIELDS)
        assert isinstance(record["pair"], list)
        assert len(record["pair"]) == 2
        assert record["pair"][0] <= record["pair"][1]
        assert isinstance(record["elements"], list)
        assert len(record["elements"]) == 2


def test_to_doc_carries_no_forbidden_truth_keys() -> None:
    # Given edit graphs over every synthetic fixture
    graphs = [
        build_reaction_edit_graph(_bundle_from_smiles(smiles))
        for smiles in (ORDER_CHANGE, H_MIGRATION, ATTR_ONLY)
    ]

    # When the documents are key-scanned
    found: set[str] = set()
    for graph in graphs:
        _collect_keys(graph.to_doc(), found)

    # Then no truth-derived key appears anywhere
    assert found & FORBIDDEN_KEYS == set()


def test_demo24_evidence_crosscheck_edit_graph_features() -> None:
    # Given the real Demo24 evidence and export tree (skip when absent)
    if not EVIDENCE_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("demo24 evidence/candidates tree not present in this checkout")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        smiles_by_id = {
            row["reaction_id"]: row["reaction_smiles"]
            for row in csv.DictReader(handle)
        }

    # When each record is re-derived from its whitelisted SMILES + export doc
    compared = 0
    for record in evidence["records"]:
        reaction_id = record["reaction_id"]
        export_path = _find_export_doc(reaction_id)
        if export_path is None or reaction_id not in smiles_by_id:
            continue
        export_bytes = export_path.read_bytes()
        expected_sha = record.get("export_sha256")
        if expected_sha and hashlib.sha256(export_bytes).hexdigest() != expected_sha:
            continue
        export = json.loads(export_bytes.decode("utf-8"))
        materials = load_endpoint_materials_from_export(export)
        bundle = rebuild_endpoint_graphs(smiles_by_id[reaction_id], materials)
        graph = build_reaction_edit_graph(
            bundle, aromatic_regions=export.get("aromatic_regions")
        )
        features = record["features"]

        # Then the golden edit-graph features match byte-field-for-byte
        assert graph.edit_counts == features["edit_counts"], reaction_id
        assert [list(c) for c in graph.edit_components] == features["edit_components"], (
            reaction_id
        )
        assert [list(c) for c in graph.connectivity_edit_components] == features[
            "connectivity_edit_components"
        ], reaction_id
        assert graph.edit_cycle_rank == features["edit_cycle_rank"], reaction_id
        assert list(graph.atom_attribute_changes) == features["atom_attribute_changes"], (
            reaction_id
        )
        assert len(graph.edits) == len(record["edits"]), reaction_id
        for ours, theirs in zip(graph.edits, record["edits"], strict=True):
            projected = ours.to_record()
            for field in GOLDEN_EDIT_FIELDS:
                assert projected[field] == theirs[field], (reaction_id, field)
        compared += 1

    # Then at least three records were available for the cross-check
    if compared < 3:
        pytest.skip(f"only {compared} demo24 export records available for cross-check")
    assert compared >= 3
