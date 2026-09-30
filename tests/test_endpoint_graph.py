"""Offline tests for EndpointGraphBundle (g1/endpoint_graph.py).

Every fixture is a synthetic mapped reaction SMILES plus whitelisted
map-keyed endpoint materials.  Both sides go through the strict G1 parse path
(``removeHs=False``), so explicit hydrogens stay in the graph and aromatic
bonds stay unkekulized at 1.5.  The suite locks conservation, template
atom-order invariance, rigid-translation invariance, typed map/conservation
errors, and the zero-truth-key output contract.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest
import rdkit

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_graph import (
    AROMATIC_MODEL,
    CODE_ELEMENT_IMBALANCE,
    CODE_H_INVENTORY,
    CODE_ISOTOPE_IMBALANCE,
    CODE_MAP_AMBIGUOUS,
    CODE_MAP_INVALID,
    CODE_MATERIALS_SCHEMA,
    NORMALIZATION_VERSION,
    SCHEMA_VERSION,
    EndpointGraphBundle,
    EndpointGraphError,
    build_endpoint_graph_bundle,
)
from pes2ts_core.g1.parse import parse_reaction
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS

#: Butane -> 2-butene-style order change (all heavy-atom maps conserved).
ORDER_CHANGE = "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH2:1]=[CH:2][CH2:3][CH3:4]"
#: Same reaction written with reversed template atom order on both sides.
ORDER_CHANGE_SHUFFLED = "[CH3:4][CH2:3][CH2:2][CH3:1]>>[CH3:4][CH2:3][CH:2]=[CH2:1]"
#: Methanol -> methanethiol: map 2 changes element O -> S.
ELEMENT_SWAP = "[CH3:1][OH:2]>>[CH3:1][SH:2]"
#: 13C-labelled methanol -> unlabelled methanol: map 1 isotope changes.
ISOTOPE_SWAP = "[13CH3:1][OH:2]>>[CH3:1][OH:2]"
#: One explicit hydrogen map present on R but deleted from the product SMILES.
H_DELETION_SMILES = "[C:1]([H:2])([H:3])[H:4]>>[C:1]([H:2])[H:3]"
#: Benzene + H2 -> 1,3-cyclohexadiene (aromatic ring bonds stay 1.5).
AROMATIC = (
    "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[H:7][H:8]"
    ">>[C:1]([H:7])([H:8])1[CH:2]=[CH:3][CH:4]=[CH:5][CH2:6]1"
)

FORBIDDEN_KEYS: frozenset[str] = frozenset(FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS)


def _materials_for(
    smiles: str, shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> dict[str, dict[int, dict[str, object]]]:
    """Build map-keyed materials whose coordinates depend only on map id."""
    reactants, products = parse_reaction(smiles)
    materials: dict[str, dict[int, dict[str, object]]] = {"r": {}, "p": {}}
    for side_key, side in (("r", reactants), ("p", products)):
        for mol, map_list in zip(side.mols, side.map_lists, strict=True):
            for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True):
                map_id = int(map_number)
                materials[side_key][map_id] = {
                    "element": atom.GetSymbol(),
                    "coordinates": [
                        float(map_id) + shift[0],
                        0.5 * map_id + shift[1],
                        -0.25 * map_id + shift[2],
                    ],
                }
    return materials


def _collect_keys(value: object, found: set[str]) -> None:
    """Recursively collect every mapping key as a string."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            _collect_keys(child, found)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, found)


def test_conservation_passes_for_consistent_reaction() -> None:
    # Given a consistent order-change reaction and its materials
    materials = _materials_for(ORDER_CHANGE)

    # When the bundle is built twice from the same inputs
    first = build_endpoint_graph_bundle(ORDER_CHANGE, materials)
    second = build_endpoint_graph_bundle(ORDER_CHANGE, materials)

    # Then conservation holds, the identity hash is stable, and versions record
    assert first.conservation.element_conserved is True
    assert first.conservation.isotope_conserved is True
    assert first.conservation.explicit_h_count_r == first.conservation.explicit_h_count_p
    assert first.conservation.n_atoms_r == first.conservation.n_atoms_p == 4
    assert first.conservation.n_components_r == first.conservation.n_components_p == 1
    assert first.content_sha256 == second.content_sha256
    assert len(first.content_sha256) == 64
    assert first.schema_version == SCHEMA_VERSION
    assert first.aromatic_model == AROMATIC_MODEL
    assert first.normalization_version == NORMALIZATION_VERSION
    assert first.rdkit_version == rdkit.__version__


def test_template_atom_order_invariance_yields_same_hash() -> None:
    # Given the same reaction written with reversed atom order
    base = build_endpoint_graph_bundle(ORDER_CHANGE, _materials_for(ORDER_CHANGE))
    shuffled = build_endpoint_graph_bundle(
        ORDER_CHANGE_SHUFFLED, _materials_for(ORDER_CHANGE_SHUFFLED)
    )

    # When comparing the frozen identity
    # Then the hash is invariant to template atom ordering
    assert base.content_sha256 == shuffled.content_sha256
    assert [node.map_id for node in base.r_graph.nodes] == [
        node.map_id for node in shuffled.r_graph.nodes
    ]
    assert base.r_graph.edges == shuffled.r_graph.edges


def test_rigid_translation_invariance_yields_same_hash() -> None:
    # Given endpoint materials rigidly translated by a constant vector
    base = build_endpoint_graph_bundle(ORDER_CHANGE, _materials_for(ORDER_CHANGE))
    shifted = build_endpoint_graph_bundle(
        ORDER_CHANGE, _materials_for(ORDER_CHANGE, shift=(5.0, -2.0, 1.5))
    )

    # When comparing the frozen identity
    # Then both the bundle hash and the geometry binding are translation-invariant
    assert base.content_sha256 == shifted.content_sha256
    assert base.r_geometry.coordinates_sha256 == shifted.r_geometry.coordinates_sha256
    assert base.p_geometry.coordinates_sha256 == shifted.p_geometry.coordinates_sha256


def test_element_swap_raises_typed_element_imbalance() -> None:
    # Given an SMILES whose product changes map 2 from O to S
    materials = _materials_for(ELEMENT_SWAP)

    # When the bundle is built
    # Then a typed ELEMENT_IMBALANCE error names the offending map
    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(ELEMENT_SWAP, materials)
    assert excinfo.value.code == CODE_ELEMENT_IMBALANCE
    assert any("map=2" in reason for reason in excinfo.value.reasons)


def test_isotope_swap_raises_typed_isotope_imbalance() -> None:
    # Given an SMILES whose product drops the 13C label on map 1
    materials = _materials_for(ISOTOPE_SWAP)

    # When the bundle is built
    # Then a typed ISOTOPE_IMBALANCE error names the offending map
    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(ISOTOPE_SWAP, materials)
    assert excinfo.value.code == CODE_ISOTOPE_IMBALANCE
    assert any("map=1" in reason for reason in excinfo.value.reasons)


def test_h_deletion_in_smiles_raises_typed_error_not_silent() -> None:
    # Given a product SMILES that drops explicit hydrogen map 4
    materials = _materials_for(H_DELETION_SMILES)

    # When the bundle is built
    # Then a typed H_INVENTORY_MISMATCH error is raised, never a silent pass
    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(H_DELETION_SMILES, materials)
    assert excinfo.value.code == CODE_H_INVENTORY
    assert any("map=4" in reason for reason in excinfo.value.reasons)


def test_h_deletion_in_materials_raises_typed_error_not_silent() -> None:
    # Given valid SMILES with all H maps present, but P materials missing map 4
    smiles = "[C:1]([H:2])([H:3])[H:4]>>[C:1]([H:2])([H:3])[H:4]"
    materials = _materials_for(smiles)
    del materials["p"][4]

    # When the bundle is built
    # Then the missing-H materials raise a typed error, never a silent accept
    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(smiles, materials)
    assert excinfo.value.code == CODE_H_INVENTORY
    assert any("map=4" in reason for reason in excinfo.value.reasons)


def test_map_invalid_on_non_bijection_extra_and_missing_maps() -> None:
    # Given materials carrying an extra map and materials missing a heavy-atom map
    extra = _materials_for(ORDER_CHANGE)
    extra["r"][99] = {"element": "H", "coordinates": [9.9, 9.9, 9.9]}
    missing = _materials_for(ORDER_CHANGE)
    del missing["r"][2]

    # When each bundle is built
    # Then both raise MAP_INVALID (non-bijection), not a silent accept
    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(ORDER_CHANGE, extra)
    assert excinfo.value.code == CODE_MAP_INVALID
    assert any("99" in reason for reason in excinfo.value.reasons)

    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(ORDER_CHANGE, missing)
    assert excinfo.value.code == CODE_MAP_INVALID
    assert any("map=2" in reason for reason in excinfo.value.reasons)


def test_map_ambiguous_on_order_based_binding() -> None:
    # Given list-form materials that omit per-entry map ids
    materials: dict[str, list[dict[str, object]]] = {
        "r": [
            {"element": "C", "coordinates": [0.0, 0.0, 0.0]},
            {"element": "C", "coordinates": [1.0, 0.0, 0.0]},
        ],
        "p": [
            {"element": "C", "coordinates": [0.0, 0.0, 0.0]},
            {"element": "C", "coordinates": [1.0, 0.0, 0.0]},
        ],
    }

    # When the bundle is built
    # Then a typed MAP_AMBIGUOUS error with the order-binding reason is raised
    with pytest.raises(EndpointGraphError) as excinfo:
        build_endpoint_graph_bundle(ORDER_CHANGE, materials)
    assert excinfo.value.code == CODE_MAP_AMBIGUOUS
    assert any("order_based_binding_forbidden" in reason for reason in excinfo.value.reasons)


def test_materials_whitelist_rejects_unknown_keys() -> None:
    # Given materials with a non-whitelisted top-level key and an unknown atom key
    bad_top = _materials_for(ORDER_CHANGE)
    bad_top["reactant"] = {}
    bad_atom = _materials_for(ORDER_CHANGE)
    bad_atom["r"][1]["charge"] = 0

    # When each bundle is built
    # Then both raise MATERIALS_SCHEMA_INVALID
    for materials in (bad_top, bad_atom):
        with pytest.raises(EndpointGraphError) as excinfo:
            build_endpoint_graph_bundle(ORDER_CHANGE, materials)
        assert excinfo.value.code == CODE_MATERIALS_SCHEMA


def test_output_doc_contains_no_forbidden_truth_keys() -> None:
    # Given a fully built aromatic bundle
    bundle = build_endpoint_graph_bundle(AROMATIC, _materials_for(AROMATIC))
    assert isinstance(bundle, EndpointGraphBundle)

    # When the document and dataclass field names are scanned
    doc_keys: set[str] = set()
    _collect_keys(bundle.to_doc(), doc_keys)
    doc_keys.update(field.name for field in dataclasses.fields(bundle))

    # Then no forbidden truth-derived key appears anywhere
    hits = sorted(doc_keys & FORBIDDEN_KEYS)
    assert hits == []


def test_aromatic_bonds_stay_unkekulized_and_versions_recorded() -> None:
    # Given an aromatic-ring reaction parsed through the fixed model
    bundle = build_endpoint_graph_bundle(AROMATIC, _materials_for(AROMATIC))

    # When the reactant ring bonds are inspected
    aromatic_edges = [edge for edge in bundle.r_graph.edges if edge.aromatic]
    ring_orders = {edge.bond_order for edge in aromatic_edges}

    # Then aromatic bonds carry order 1.5 on both the graph and the doc
    assert len(aromatic_edges) == 6
    assert ring_orders == {1.5}
    doc = bundle.to_doc()
    assert doc["aromatic_model"] == "rdkit_default_unkekulized"
    assert doc["rdkit_version"] == rdkit.__version__
    assert doc["r_geometry"]["unit"] == "angstrom"
    assert len(doc["r_geometry"]["coordinates_sha256"]) == 64


def test_node_and_edge_property_sets_are_complete() -> None:
    # Given bundles built from a fully explicit-H methane and the order-change reaction
    explicit_h_smiles = "[C:1]([H:2])([H:3])([H:4])[H:5]>>[C:1]([H:2])([H:3])([H:4])[H:5]"
    explicit_bundle = build_endpoint_graph_bundle(
        explicit_h_smiles, _materials_for(explicit_h_smiles)
    )
    bundle = build_endpoint_graph_bundle(ORDER_CHANGE, _materials_for(ORDER_CHANGE))

    # When the node and edge records are inspected
    # Then every design §3.1 property is present with the expected values
    node = next(n for n in explicit_bundle.r_graph.nodes if n.map_id == 1)
    assert node.element == "C"
    assert node.isotope == 0
    assert node.formal_charge == 0
    assert node.radical_electrons == 0
    assert node.explicit_H_neighbors == 4
    assert node.aromatic is False
    assert isinstance(node.stereo, str)
    h_node = next(n for n in explicit_bundle.r_graph.nodes if n.map_id == 2)
    assert h_node.element == "H"
    edge = bundle.r_graph.edges[0]
    assert (edge.map_a, edge.map_b) == (1, 2)
    assert edge.connection_type == "SINGLE"
    assert edge.bond_order == 1.0
    assert edge.aromatic is False
    assert isinstance(edge.stereo, str)
    doc_nodes = bundle.to_doc()["r_graph"]["nodes"]
    assert all(
        set(row) >= {
            "map_id", "element", "isotope", "formal_charge",
            "radical_electrons", "explicit_H_neighbors", "aromatic", "stereo",
        }
        for row in doc_nodes
    )
