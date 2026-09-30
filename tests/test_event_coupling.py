"""Offline tests for the typed event coupling graph H (g1/event_coupling.py).

Primary suite is synthetic: hand-built EndpointGraphBundle graphs with exactly
chosen edit topology (same trust boundary as test_reaction_edit_graph.py).
Locks genuinely independent strong components, weak CONTEXT_NEAR never
merging events (the plan failure scenario: remote edits are not
auto-synchronised), H2 formation never labelled H_TRANSFER, one aromatic
region = one DELOCALIZED_REGION event, shared-centre justification,
every-edit-exactly-once-or-shared membership, and determinism.  One gated
cross-check re-derives ``hydrogen_events`` + ``aromatic_regions`` for every
Demo24 evidence record from the real export tree when present
(skip-if-data-missing; todo 10 owns the frozen-fixture golden).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

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
    EVIDENCE_STRONG,
    EVIDENCE_WEAK,
    EVENT_CONNECTIVITY_EXCHANGE,
    EVENT_CONTEXT_NEAR,
    EVENT_DELOCALIZED,
    EVENT_H2,
    EVENT_H_TRANSFER,
    EVENT_RING,
    EVENT_SHARED_CENTER,
    EVENT_TYPES,
    RULE_CONTEXT_NEAR,
    RULE_H2_EVENT,
    RULE_H_TRANSFER,
    RULE_SHARED_CENTER,
    RULE_SHARED_EDIT,
    aromatic_regions_from_bundle,
    build_event_coupling_graph,
    hydrogen_events_from_bundle,
)
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.graph_rebuild import (
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.utils.hashing import stable_json_dumps

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json"
CSV_PATH = ROOT / "data/manifests/pes2ts_demo24_candidates_v1.csv"
EXPORT_TREES: tuple[Path, ...] = (
    ROOT / "data/interim/g1_v2/export_contracts_v1",
    ROOT / "data/interim/g1_v2/export",
)

FORBIDDEN_KEYS: frozenset[str] = frozenset(FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS)

#: Two disjoint H transfers (separate molecular components).
TWO_H_TRANSFERS_ELEMENTS: dict[int, str] = {1: "C", 2: "O", 3: "H", 4: "C", 5: "O", 6: "H"}
TWO_H_TRANSFERS_R: list[tuple[int, int, float]] = [(1, 2, 1), (2, 3, 1), (4, 5, 1), (5, 6, 1)]
TWO_H_TRANSFERS_P: list[tuple[int, int, float]] = [(1, 2, 1), (1, 3, 1), (4, 5, 1), (4, 6, 1)]

#: H2 formation: C1-H3 + C4-H5 on R, H3-H5 on P (to_hh, never transfer).
H2_FORMATION_ELEMENTS: dict[int, str] = {1: "C", 3: "H", 4: "C", 5: "H"}
H2_FORMATION_R: list[tuple[int, int, float]] = [(1, 3, 1), (4, 5, 1)]
H2_FORMATION_P: list[tuple[int, int, float]] = [(3, 5, 1)]

#: Benzene R aromatic -> kekulized P: six order_changed edits, one region.
BENZENE_ELEMENTS: dict[int, str] = {m: "C" for m in range(1, 7)}
BENZENE_AROMATIC: set[tuple[int, int]] = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)}
BENZENE_R: list[tuple[int, int, float]] = [(a, b, 1.5) for a, b in sorted(BENZENE_AROMATIC)]
BENZENE_P: list[tuple[int, int, float]] = [
    (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 6, 2), (1, 6, 1),
]

#: Nearby-but-disjoint H transfers on one chain (shells overlap at radius 2).
NEAR_H_ELEMENTS: dict[int, str] = {m: ("H" if m in (7, 8) else "C") for m in range(1, 9)}
NEAR_H_R: list[tuple[int, int, float]] = [
    (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (2, 7, 1), (4, 8, 1),
]
NEAR_H_P: list[tuple[int, int, float]] = [
    (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (1, 7, 1), (5, 8, 1),
]

#: Ring closure (1,5) + H6 transfer 3->1 sharing centre atom 1.
SHARED_CENTER_ELEMENTS: dict[int, str] = {m: ("H" if m == 6 else "C") for m in range(1, 7)}
SHARED_CENTER_R: list[tuple[int, int, float]] = [
    (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (3, 6, 1),
]
SHARED_CENTER_P: list[tuple[int, int, float]] = [
    (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (1, 5, 1), (1, 6, 1),
]

#: Substitution pattern: A-B (1,2) broken + B-C (2,3) formed sharing B=2.
EXCHANGE_ELEMENTS: dict[int, str] = {1: "C", 2: "C", 3: "C", 4: "H"}
EXCHANGE_R: list[tuple[int, int, float]] = [(1, 2, 1), (2, 4, 1)]
EXCHANGE_P: list[tuple[int, int, float]] = [(2, 3, 1), (2, 4, 1)]


def _node(map_id: int, element: str, *, aromatic: bool = False) -> AtomNode:
    return AtomNode(
        map_id=map_id,
        element=element,
        isotope=0,
        formal_charge=0,
        radical_electrons=0,
        explicit_H_neighbors=0,
        aromatic=aromatic,
        stereo="CHI_UNSPECIFIED",
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
    """Connected components ordered by smallest map (bundle numbering)."""
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
) -> EndpointGraphBundle:
    """Assemble a minimal EndpointGraphBundle with populated components."""
    elements = dict(elements or {})
    r_aromatic = {(min(a, b), max(a, b)) for a, b in (r_aromatic or set())}
    p_aromatic = {(min(a, b), max(a, b)) for a, b in (p_aromatic or set())}
    maps = sorted({m for spec in r_specs + p_specs for m in spec[:2]} | set(elements))

    def side(
        specs: list[tuple[int, int, float]], aromatic: set[tuple[int, int]]
    ) -> SideGraph:
        arom_atoms = {m for pair in aromatic for m in pair}
        nodes = tuple(
            _node(m, elements.get(m, "C"), aromatic=m in arom_atoms) for m in maps
        )
        edges = tuple(
            sorted(
                (_edge(a, b, order, (min(a, b), max(a, b)) in aromatic) for a, b, order in specs),
                key=lambda e: (e.map_a, e.map_b),
            )
        )
        comps = _components(maps, [(e.map_a, e.map_b) for e in edges])
        return SideGraph(nodes=nodes, edges=edges, components=comps)

    r_graph = side(r_specs, r_aromatic)
    p_graph = side(p_specs, p_aromatic)
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


def _coupling(
    r_specs: list[tuple[int, int, float]],
    p_specs: list[tuple[int, int, float]],
    *,
    elements: Mapping[int, str] | None = None,
    r_aromatic: set[tuple[int, int]] | None = None,
    p_aromatic: set[tuple[int, int]] | None = None,
    context_radius: int = 2,
) -> tuple[EndpointGraphBundle, object, object]:
    """Build bundle + edit graph + context + coupling graph for one fixture."""
    bundle = _hand_bundle(
        r_specs, p_specs, elements=elements, r_aromatic=r_aromatic, p_aromatic=p_aromatic
    )
    aromatic = aromatic_regions_from_bundle(bundle)
    edit_graph = build_reaction_edit_graph(
        bundle, aromatic_regions=aromatic if aromatic else None
    )
    context = build_endpoint_context(bundle, edit_graph, context_radius=context_radius)
    coupling = build_event_coupling_graph(bundle, edit_graph, context)
    return bundle, edit_graph, coupling


def _find_export_doc(reaction_id: str) -> Path | None:
    shard = f"{int(reaction_id[4:]) // 1000:05d}"
    for tree in EXPORT_TREES:
        candidate = tree / shard / f"{reaction_id}.json"
        if candidate.exists():
            return candidate
    return None


def _collect_keys(value: object, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            _collect_keys(child, found)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, found)


# ---------------------------------------------------------------------------
# Synthetic behaviour locks.
# ---------------------------------------------------------------------------


def test_two_disjoint_h_transfers_form_independent_strong_components() -> None:
    # Given two H transfers on disjoint molecular components
    _, edit_graph, coupling = _coupling(
        TWO_H_TRANSFERS_R, TWO_H_TRANSFERS_P, elements=TWO_H_TRANSFERS_ELEMENTS
    )

    # Then exactly two H_TRANSFER events exist, one per transfer
    transfers = coupling.events_by_type(EVENT_H_TRANSFER)
    assert len(transfers) == 2
    assert all(e.rule_id == RULE_H_TRANSFER for e in transfers)
    assert [e.event_id for e in transfers] == [
        "H_TRANSFER:h3:2-1",
        "H_TRANSFER:h6:5-4",
    ]

    # And the two transfers sit in genuinely different strong components
    component_a = coupling.component_of("H_TRANSFER:h3:2-1")
    component_b = coupling.component_of("H_TRANSFER:h6:5-4")
    assert component_a != component_b
    assert "H_TRANSFER:h3:2-1" not in component_b
    assert "H_TRANSFER:h6:5-4" not in component_a

    # And no coupling record links the two transfer events at all
    pair = ("H_TRANSFER:h3:2-1", "H_TRANSFER:h6:5-4")
    linked = [
        rec
        for rec in coupling.coupling_records
        if len(rec.event_ids) == 2 and tuple(rec.event_ids) == pair
    ]
    assert linked == []

    # And every edit is registered exactly-once-or-shared
    edit_pairs = {e.pair for e in edit_graph.edits}
    registry = coupling.membership_map()
    assert set(registry) == edit_pairs
    assert all(len(ids) >= 1 for ids in registry.values())


def test_weak_context_near_does_not_merge_strong_components() -> None:
    # Given two nearby-but-disjoint H transfers whose shells overlap at radius 2
    _, _, coupling = _coupling(
        NEAR_H_R, NEAR_H_P, elements=NEAR_H_ELEMENTS, context_radius=2
    )
    transfers = coupling.events_by_type(EVENT_H_TRANSFER)
    assert len(transfers) == 2
    id_a, id_b = transfers[0].event_id, transfers[1].event_id

    # Then a CONTEXT_NEAR weak link records the shell overlap
    pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
    weak = [w for w in coupling.weak_links if tuple(w.event_ids) == pair]
    assert weak, coupling.weak_links
    assert all(w.rule_id == RULE_CONTEXT_NEAR for w in weak)
    assert all(w.evidence_level == EVIDENCE_WEAK for w in weak)
    assert all(w.support_atom_maps for w in weak)

    # And the weak link NEVER merges the two strong components
    assert coupling.component_of(id_a) != coupling.component_of(id_b)
    assert len(coupling.strong_components) >= 2

    # And no strong rule fires between the two transfers
    strong_between = [
        rec
        for rec in coupling.coupling_records
        if len(rec.event_ids) == 2
        and tuple(rec.event_ids) == pair
        and rec.evidence_level == EVIDENCE_STRONG
    ]
    assert strong_between == []


def test_h2_formation_is_not_labelled_h_transfer() -> None:
    # Given H2 formation: two hydrogens leave heavy partners and bond to each other
    bundle, _, coupling = _coupling(
        H2_FORMATION_R, H2_FORMATION_P, elements=H2_FORMATION_ELEMENTS
    )

    # Then the golden hydrogen events use the H2 taxonomy, not transfer
    h_events = hydrogen_events_from_bundle(bundle)
    assert [c["kind"] for c in h_events] == ["to_hh", "to_hh"]
    assert all(c["kind"] != RULE_H_TRANSFER for c in h_events)

    # And exactly one H2_EVENT owns the H-H pair; zero H_TRANSFER events
    h2_events = coupling.events_by_type(EVENT_H2)
    assert len(h2_events) == 1
    assert h2_events[0].event_id == "H2_EVENT:hh3_5"
    assert h2_events[0].rule_id == RULE_H2_EVENT
    assert set(h2_events[0].edit_pairs) == {(1, 3), (3, 5), (4, 5)}
    assert coupling.events_by_type(EVENT_H_TRANSFER) == ()


def test_aromatic_region_is_exactly_one_delocalized_event() -> None:
    # Given benzene re-kekulization: six aromatic bond-order edits in one region
    bundle, edit_graph, coupling = _coupling(
        BENZENE_R,
        BENZENE_P,
        elements=BENZENE_ELEMENTS,
        r_aromatic=BENZENE_AROMATIC,
    )

    # Then the golden aromatic region map has exactly one region of six pairs
    regions = aromatic_regions_from_bundle(bundle)
    assert coupling.aromatic_regions == regions
    assert len(regions) == 1
    members = next(iter(regions.values()))
    assert len(members) == 6

    # And all six order_changed edits belong to ONE DELOCALIZED_REGION event
    deloc = coupling.events_by_type(EVENT_DELOCALIZED)
    assert len(deloc) == 1
    assert len(deloc[0].edit_pairs) == 6
    assert {e.edit_kind for e in edit_graph.edits} == {"order_changed"}
    assert len(edit_graph.edits) == 6

    # And the single event forms a single strong component (never six scans)
    assert coupling.strong_components == ((deloc[0].event_id,),)
    registry = coupling.membership_map()
    assert all(ids == (deloc[0].event_id,) for ids in registry.values())


def test_shared_center_justification_couples_events() -> None:
    # Given a ring closure (1,5) and an H transfer 3->1 sharing centre atom 1
    _, _, coupling = _coupling(
        SHARED_CENTER_R, SHARED_CENTER_P, elements=SHARED_CENTER_ELEMENTS
    )
    transfers = coupling.events_by_type(EVENT_H_TRANSFER)
    rings = coupling.events_by_type(EVENT_RING)
    assert len(transfers) == 1 and len(rings) >= 1

    # Then a SHARED_CENTER strong link names the shared centre atom
    shared_links = [
        rec
        for rec in coupling.coupling_records
        if rec.rule_id == RULE_SHARED_CENTER and len(rec.event_ids) == 2
    ]
    assert shared_links, coupling.coupling_records
    assert all(1 in rec.support_atom_maps for rec in shared_links)
    assert all(rec.evidence_level == EVIDENCE_STRONG for rec in shared_links)

    # And shared-edit membership also couples transfer to ring events
    edit_links = [
        rec
        for rec in coupling.coupling_records
        if rec.rule_id == RULE_SHARED_EDIT and len(rec.event_ids) == 2
    ]
    assert edit_links

    # And all involved events land in ONE strong component
    transfer_id = transfers[0].event_id
    for ring in rings:
        assert coupling.component_of(transfer_id) == coupling.component_of(ring.event_id)

    # And every multi-membership row carries a typed justification
    multi = [row for row in coupling.membership if len(row.event_ids) >= 2]
    assert multi
    for row in multi:
        assert row.justifications, row
        assert all(j.rule_id and j.event_id and j.detail for j in row.justifications)


def test_connectivity_exchange_groups_edits_sharing_a_center() -> None:
    # Given A-B broken + B-C formed sharing centre atom B=2
    _, _, coupling = _coupling(
        EXCHANGE_R, EXCHANGE_P, elements=EXCHANGE_ELEMENTS
    )

    # Then one CONNECTIVITY_EXCHANGE event owns both edits
    exchanges = coupling.events_by_type(EVENT_CONNECTIVITY_EXCHANGE)
    assert len(exchanges) == 1
    assert exchanges[0].event_id == "CONNECTIVITY_EXCHANGE:c2"
    assert set(exchanges[0].edit_pairs) == {(1, 2), (2, 3)}
    assert exchanges[0].support_atom_maps  # centre atoms recorded

    # And every event multi-sharing an exchange edit sits in the same component
    registry = coupling.membership_map()
    component = coupling.component_of(exchanges[0].event_id)
    assert exchanges[0].event_id in component
    partner_ids: set[str] = set()
    for pair in exchanges[0].edit_pairs:
        partner_ids.update(registry[pair])
    for event_id in partner_ids:
        assert coupling.component_of(event_id) == component


def test_membership_registry_covers_every_edit_exactly_once_or_shared() -> None:
    # Given a fixture with mixed H, ring, and exchange activity
    _, edit_graph, coupling = _coupling(
        SHARED_CENTER_R, SHARED_CENTER_P, elements=SHARED_CENTER_ELEMENTS
    )

    # When the registry is inspected
    registry = coupling.membership_map()
    edit_pairs = {e.pair for e in edit_graph.edits}

    # Then every edit appears with at least one owning event, no extras
    assert set(registry) == edit_pairs
    assert all(len(ids) >= 1 for ids in registry.values())

    # And strong components partition the event set exactly
    all_ids = {e.event_id for e in coupling.events}
    flat = [eid for c in coupling.strong_components for eid in c]
    assert sorted(flat) == sorted(all_ids)
    assert len(flat) == len(set(flat))


def test_determinism_same_input_identical_serialization() -> None:
    # Given the same synthetic fixture built twice
    _, _, first = _coupling(
        SHARED_CENTER_R, SHARED_CENTER_P, elements=SHARED_CENTER_ELEMENTS
    )
    _, _, second = _coupling(
        SHARED_CENTER_R, SHARED_CENTER_P, elements=SHARED_CENTER_ELEMENTS
    )

    # When both documents are stably serialized
    payload_first = stable_json_dumps(first.to_doc())
    payload_second = stable_json_dumps(second.to_doc())

    # Then the serializations are byte-identical
    assert payload_first == payload_second


def test_event_type_vocabulary_matches_design() -> None:
    # Given the design §4.3 vocabulary
    # When the module constants are read
    # Then all seven §4.3 names are present in documentation order
    assert EVENT_TYPES == (
        EVENT_H_TRANSFER,
        EVENT_H2,
        EVENT_CONNECTIVITY_EXCHANGE,
        EVENT_RING,
        EVENT_DELOCALIZED,
        EVENT_SHARED_CENTER,
        EVENT_CONTEXT_NEAR,
    )
    assert all(isinstance(name, str) and name for name in EVENT_TYPES)


def test_to_doc_carries_no_forbidden_truth_keys() -> None:
    # Given coupling graphs over representative synthetic fixtures
    graphs = []
    for r_specs, p_specs, elements in (
        (TWO_H_TRANSFERS_R, TWO_H_TRANSFERS_P, TWO_H_TRANSFERS_ELEMENTS),
        (H2_FORMATION_R, H2_FORMATION_P, H2_FORMATION_ELEMENTS),
        (BENZENE_R, BENZENE_P, BENZENE_ELEMENTS),
    ):
        _, _, coupling = _coupling(r_specs, p_specs, elements=elements)
        graphs.append(coupling)

    # When the documents are key-scanned
    found: set[str] = set()
    for coupling in graphs:
        _collect_keys(coupling.to_doc(), found)

    # Then no truth-derived key appears anywhere
    assert found & FORBIDDEN_KEYS == set()


def test_coupling_records_expose_required_rule_fields() -> None:
    # Given any coupling graph
    _, _, coupling = _coupling(
        TWO_H_TRANSFERS_R, TWO_H_TRANSFERS_P, elements=TWO_H_TRANSFERS_ELEMENTS
    )

    # When every coupling record is projected
    records = coupling.coupling_records

    # Then each record carries the four §4.3-required fields with typed values
    assert records
    for record in records:
        projected = record.to_record()
        assert set(projected) == {
            "rule_id",
            "evidence_level",
            "event_ids",
            "support_atom_maps",
            "support_edges",
        }
        assert projected["rule_id"]
        assert projected["evidence_level"] in (EVIDENCE_STRONG, EVIDENCE_WEAK)
        assert projected["event_ids"]
        assert isinstance(projected["support_atom_maps"], list)
        assert isinstance(projected["support_edges"], list)


# ---------------------------------------------------------------------------
# Demo24 evidence cross-check (skip-if-data-missing).
# ---------------------------------------------------------------------------


def test_demo24_evidence_crosscheck_hydrogen_and_aromatic() -> None:
    # Given the real Demo24 evidence and export tree (skip when absent)
    if not EVIDENCE_PATH.exists() or not CSV_PATH.exists():
        pytest.skip("demo24 evidence/candidates tree not present in this checkout")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        smiles_by_id = {
            row["reaction_id"]: row["reaction_smiles"] for row in csv.DictReader(handle)
        }

    # When each record is re-derived from whitelisted SMILES + export doc
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
        aromatic = aromatic_regions_from_bundle(bundle)
        h_events = hydrogen_events_from_bundle(bundle)
        edit_graph = build_reaction_edit_graph(
            bundle, aromatic_regions=export.get("aromatic_regions")
        )
        maps = export["maps"]
        index = {m: i for i, m in enumerate(maps)}
        r_coords = {m: tuple(export["r_coordinates"][index[m]]) for m in maps}
        p_coords = {m: tuple(export["p_coordinates"][index[m]]) for m in maps}
        context = build_endpoint_context(
            bundle, edit_graph, r_coordinates=r_coords, p_coordinates=p_coords
        )
        coupling = build_event_coupling_graph(bundle, edit_graph, context)
        features = record["features"]

        # Then the golden hydrogen_events and aromatic_regions match exactly
        assert list(coupling.hydrogen_events) == features["hydrogen_events"], reaction_id
        assert coupling.aromatic_regions == features["aromatic_regions"], reaction_id

        # And the partition of edits into events covers every edit exactly-once-or-shared
        edit_pairs = {e.pair for e in edit_graph.edits}
        registry = coupling.membership_map()
        assert set(registry) == edit_pairs, reaction_id
        for row in coupling.membership:
            assert len(row.event_ids) >= 1, (reaction_id, row)
            if len(row.event_ids) >= 2:
                assert row.justifications, (reaction_id, row)

        # And strong components partition the event set
        all_ids = {e.event_id for e in coupling.events}
        flat = [eid for c in coupling.strong_components for eid in c]
        assert sorted(flat) == sorted(all_ids), reaction_id
        assert len(flat) == len(set(flat)), reaction_id

        compared += 1

    # Then the real data tree yields all 24 records; absent tree skips loudly
    if compared == 0:
        pytest.skip("demo24 export tree not present in this checkout")
    assert compared == 24, f"expected all 24 evidence records, compared {compared}"
