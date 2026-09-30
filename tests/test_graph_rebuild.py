"""Offline tests for whitelisted R/P graph rebuild (scan_strategy/graph_rebuild.py).

Every fixture is synthetic and in-memory: a mapped reaction SMILES plus
map-keyed endpoint materials, or a minimal export-contract-like document
constructed inline.  No test reads the real data tree.  The suite locks:
bundle conservation and stable identity, source-hash provenance binding,
output purity (no truth-derived keys anywhere), a zero-read guarantee for
the internal v2 data tree (armed path-open guard), the export-doc loader
roundtrip, typed loader rejections, and batch API capability.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

import pes2ts_core.scan_strategy.graph_rebuild as graph_rebuild_module
from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.endpoint_materials import CODE_MATERIALS_SCHEMA
from pes2ts_core.g1.graph_payload import graph_payload
from pes2ts_core.g1.parse import parse_reaction
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.graph_rebuild import (
    CODE_EXPORT_SCHEMA,
    CODE_SMILES_INVALID,
    PROVENANCE_KEY,
    REBUILD_SOURCE_KIND,
    RebuildInputError,
    RebuiltEndpointGraphBundle,
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.utils.hashing import stable_json_dumps

#: Butane -> 2-butene-style order change (all heavy-atom maps conserved).
ORDER_CHANGE = "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH2:1]=[CH:2][CH2:3][CH3:4]"
#: Same reaction written with reversed template atom order on both sides.
ORDER_CHANGE_SHUFFLED = "[CH3:4][CH2:3][CH2:2][CH3:1]>>[CH3:4][CH2:3][CH:2]=[CH2:1]"
#: Benzene + H2 -> 1,3-cyclohexadiene (aromatic ring bonds stay 1.5).
AROMATIC = (
    "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[H:7][H:8]"
    ">>[C:1]([H:7])([H:8])1[CH:2]=[CH:3][CH:4]=[CH:5][CH2:6]1"
)

FORBIDDEN_KEYS: frozenset[str] = frozenset(FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS)
NAMED_PURITY_KEYS: frozenset[str] = frozenset(
    {"endpoint_match", "orientation", "irc_evidence"}
)


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


def _synthetic_export_doc(smiles: str) -> dict[str, object]:
    """Minimal inline export-contract-like doc for *smiles* (no real data reads).

    Carries the whitelisted coordinate/element fields the loader consumes plus
    representative extra fields real contract exports carry, so loader
    schema-generality is exercised.
    """
    materials = _materials_for(smiles)
    map_ids = sorted(materials["r"])
    reactants, _products = parse_reaction(smiles)
    side_symbols: list[str] = []
    for mol, map_list in zip(reactants.mols, reactants.map_lists, strict=True):
        for atom, _map_number in zip(mol.GetAtoms(), map_list, strict=True):
            side_symbols.append(str(atom.GetSymbol()))
    element_by_map = {
        int(map_number): str(atom.GetSymbol())
        for mol, map_list in zip(reactants.mols, reactants.map_lists, strict=True)
        for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True)
    }
    atomic_number_by_symbol = {"C": 6, "H": 1, "O": 8, "N": 7, "S": 16, "Cl": 17, "F": 9}
    return {
        "schema_version": "g1_v2_export_v1",
        "reaction_id": "RXN_SYNTHETIC_0000000001",
        "dataset_version": "zenodo-18551029-rev1",
        "atom_order": "map ascending",
        "maps": map_ids,
        "elements": [element_by_map[m] for m in map_ids],
        "r_coordinates": [list(materials["r"][m]["coordinates"]) for m in map_ids],
        "p_coordinates": [list(materials["p"][m]["coordinates"]) for m in map_ids],
        "r_atomic_numbers": [
            atomic_number_by_symbol[element_by_map[m]] for m in map_ids
        ],
        "p_atomic_numbers": [
            atomic_number_by_symbol[element_by_map[m]] for m in map_ids
        ],
        "atom_rows": [
            {
                "element": element_by_map[m],
                "map": m,
                "r_component": "R0",
                "r_local_index": index,
                "p_component": "P0",
                "p_local_index": index,
            }
            for index, m in enumerate(map_ids)
        ],
        # Real contract exports carry their own whitelisted-derived reaction
        # record under this key; the loader must ignore it entirely.
        "edits": [{"pair": [1, 2], "edit_kind": "order_changed"}],
        "status_summary": {"audit_status": "clean", "p1_status": "resolved_unique"},
        "mapping_provenance": "truth_assisted_p1",
    }


def _expected_materials_sha256(
    materials: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> str:
    """Documented canonical binding: sorted (side, map, element, coordinates) rows."""
    canonical: list[object] = []
    for side in sorted(materials):
        for map_id in sorted(materials[side]):
            row = materials[side][map_id]
            assert isinstance(row, Mapping)
            canonical.append([side, map_id, row["element"], list(row["coordinates"])])
    return hashlib.sha256(stable_json_dumps(canonical).encode("utf-8")).hexdigest()  # pyright: ignore[reportArgumentType]


def _expected_payload_sha256(smiles: str) -> str:
    """Documented canonical map-space payload binding (string-keyed projection)."""
    payload = graph_payload(smiles)
    assert payload is not None
    canonical = {
        "elements": {str(k): v for k, v in payload["elements"].items()},
        "r_bonds": [list(bond) for bond in payload["r_bonds"]],
        "p_bonds": [list(bond) for bond in payload["p_bonds"]],
        "r_components": {str(k): v for k, v in payload["r_components"].items()},
        "p_components": {str(k): v for k, v in payload["p_components"].items()},
    }
    return hashlib.sha256(stable_json_dumps(canonical).encode("utf-8")).hexdigest()


def _collect_keys(value: object, found: set[str]) -> None:
    """Recursively collect every mapping key as a string."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            _collect_keys(child, found)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, found)


def test_rebuild_returns_endpoint_graph_bundle_with_conservation_and_stable_hash() -> None:
    # Given a consistent order-change reaction and its whitelisted materials
    materials = _materials_for(ORDER_CHANGE)

    # When the graph is rebuilt twice from the same inputs
    first = rebuild_endpoint_graphs(ORDER_CHANGE, materials)
    second = rebuild_endpoint_graphs(ORDER_CHANGE, materials)

    # Then both results are endpoint graph bundles with conserved chemistry
    # and a stable identity hash
    assert isinstance(first, EndpointGraphBundle)
    assert isinstance(first, RebuiltEndpointGraphBundle)
    assert first.conservation.element_conserved is True
    assert first.conservation.isotope_conserved is True
    assert first.conservation.n_atoms_r == first.conservation.n_atoms_p == 4
    assert first.conservation.n_components_r == first.conservation.n_components_p == 1
    assert first.conservation.explicit_h_count_r == first.conservation.explicit_h_count_p == 0
    assert first.content_sha256 == second.content_sha256
    assert len(first.content_sha256) == 64
    assert first.to_doc()["schema_version"] == "g1_endpoint_graph_v1"
    assert first.r_graph.edges[0].bond_order == 1.0
    assert any(edge.bond_order == 2.0 for edge in first.p_graph.edges)


def test_rebuild_provenance_binds_source_hashes() -> None:
    # Given a rebuild from a known SMILES and known materials
    materials = _materials_for(ORDER_CHANGE)
    bundle = rebuild_endpoint_graphs(ORDER_CHANGE, materials)

    # When the bundle document is inspected
    doc = bundle.to_doc()
    provenance = doc[PROVENANCE_KEY]

    # Then the provenance block binds the SMILES, materials, and map-space
    # payload digests exactly as documented
    assert provenance["source_kind"] == REBUILD_SOURCE_KIND
    assert provenance["reaction_smiles_sha256"] == hashlib.sha256(
        ORDER_CHANGE.encode("utf-8")
    ).hexdigest()
    assert provenance["endpoint_materials_sha256"] == _expected_materials_sha256(materials)
    assert provenance["graph_payload_sha256"] == _expected_payload_sha256(ORDER_CHANGE)
    for digest in (
        provenance["reaction_smiles_sha256"],
        provenance["endpoint_materials_sha256"],
        provenance["graph_payload_sha256"],
    ):
        assert isinstance(digest, str) and len(digest) == 64


def test_rebuild_content_hash_invariant_to_template_atom_order() -> None:
    # Given the same reaction written with reversed template atom order
    base = rebuild_endpoint_graphs(ORDER_CHANGE, _materials_for(ORDER_CHANGE))
    shuffled = rebuild_endpoint_graphs(
        ORDER_CHANGE_SHUFFLED, _materials_for(ORDER_CHANGE_SHUFFLED)
    )

    # When the frozen identities are compared
    # Then content and materials bindings are template-order invariant while
    # the SMILES digest reflects the literal source string
    assert base.content_sha256 == shuffled.content_sha256
    assert (
        base.to_doc()[PROVENANCE_KEY]["endpoint_materials_sha256"]
        == shuffled.to_doc()[PROVENANCE_KEY]["endpoint_materials_sha256"]
    )
    assert (
        base.to_doc()[PROVENANCE_KEY]["graph_payload_sha256"]
        == shuffled.to_doc()[PROVENANCE_KEY]["graph_payload_sha256"]
    )
    assert (
        base.to_doc()[PROVENANCE_KEY]["reaction_smiles_sha256"]
        != shuffled.to_doc()[PROVENANCE_KEY]["reaction_smiles_sha256"]
    )


def test_rebuild_materials_hash_binds_raw_coordinates() -> None:
    # Given materials rigidly translated by a constant vector
    base = rebuild_endpoint_graphs(ORDER_CHANGE, _materials_for(ORDER_CHANGE))
    shifted = rebuild_endpoint_graphs(
        ORDER_CHANGE, _materials_for(ORDER_CHANGE, shift=(5.0, -2.0, 1.5))
    )

    # When provenance and content are compared
    # Then the chemistry identity is translation-invariant while the source
    # materials digest binds the literal coordinates
    assert base.content_sha256 == shifted.content_sha256
    assert base.r_geometry.coordinates_sha256 == shifted.r_geometry.coordinates_sha256
    base_doc = base.to_doc()
    shifted_doc = shifted.to_doc()
    assert (
        base_doc[PROVENANCE_KEY]["endpoint_materials_sha256"]
        != shifted_doc[PROVENANCE_KEY]["endpoint_materials_sha256"]
    )
    assert (
        base_doc[PROVENANCE_KEY]["reaction_smiles_sha256"]
        == shifted_doc[PROVENANCE_KEY]["reaction_smiles_sha256"]
    )
    assert (
        base_doc[PROVENANCE_KEY]["graph_payload_sha256"]
        == shifted_doc[PROVENANCE_KEY]["graph_payload_sha256"]
    )


def test_rebuilt_bundle_doc_contains_no_truth_derived_keys() -> None:
    # Given a rebuilt aromatic bundle (rich graph plus provenance block)
    bundle = rebuild_endpoint_graphs(AROMATIC, _materials_for(AROMATIC))

    # When every key in the document and the dataclass fields is scanned
    doc_keys: set[str] = set()
    _collect_keys(bundle.to_doc(), doc_keys)
    doc_keys.update(field.name for field in dataclasses.fields(bundle))

    # Then no truth-derived key appears anywhere, named keys included
    assert sorted(doc_keys & FORBIDDEN_KEYS) == []
    assert sorted(doc_keys & NAMED_PURITY_KEYS) == []


def test_rebuild_module_source_has_no_internal_tree_references() -> None:
    # Given the shipped rebuild module
    module_path = Path(graph_rebuild_module.__file__)
    assert module_path is not None

    # When its full source text is scanned
    source = module_path.read_text(encoding="utf-8")

    # Then no reference to the internal v2 data tree appears anywhere
    assert "edits" not in source


def test_rebuild_never_opens_paths_under_internal_v2_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given armed guards that fail any open whose path mentions the internal
    # v2 data tree
    real_path_open = Path.open
    real_builtin_open = builtins.open

    def guarded_path_open(self: Path, *args: object, **kwargs: object) -> object:
        resolved = str(self.resolve()) if self.exists() else str(self)
        if "edits" in resolved:
            raise AssertionError(f"forbidden path open: {resolved}")
        return real_path_open(self, *args, **kwargs)  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]

    def guarded_builtin_open(file: object, *args: object, **kwargs: object) -> object:
        if isinstance(file, str | bytes | os.PathLike):
            if "edits" in str(file):
                raise AssertionError(f"forbidden builtin open: {file}")
        return real_builtin_open(file, *args, **kwargs)  # pyright: ignore[reportGeneralTypeIssues]

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)

    # And a probe file whose path mentions the internal tree
    probe_dir = tmp_path / "g1_v2" / "edits"
    probe_dir.mkdir(parents=True)
    probe = probe_dir / "RXN_0000000001.json"

    # When a full rebuild runs from SMILES + materials only
    bundle = rebuild_endpoint_graphs(ORDER_CHANGE, _materials_for(ORDER_CHANGE))

    # Then the rebuild succeeds without touching any guarded path
    assert len(bundle.content_sha256) == 64
    assert bundle.conservation.element_conserved is True

    # And the guard itself is armed: an injected open fails loudly
    with pytest.raises(AssertionError, match="forbidden"):
        probe.open("w")  # noqa: PTH123 — deliberate guard probe
    with pytest.raises(AssertionError, match="forbidden"):
        builtins.open(probe, "w")


def test_load_endpoint_materials_from_export_roundtrip_rebuilds_same_bundle() -> None:
    # Given a minimal synthetic export-like document for the order-change reaction
    doc = _synthetic_export_doc(ORDER_CHANGE)

    # When materials are loaded from the parsed document
    materials = load_endpoint_materials_from_export(doc)

    # Then the materials carry every map with doc-consistent elements and
    # coordinates, in the mapping form the bundle builder accepts
    assert set(materials) == {"r", "p"}
    assert sorted(materials["r"]) == [1, 2, 3, 4]
    assert sorted(materials["p"]) == [1, 2, 3, 4]
    for map_id in (1, 2, 3, 4):
        assert materials["r"][map_id]["element"] == "C"
        assert materials["p"][map_id]["element"] == "C"
        assert materials["r"][map_id]["coordinates"] == doc["r_coordinates"][map_id - 1]
        assert materials["p"][map_id]["coordinates"] == doc["p_coordinates"][map_id - 1]

    # And a rebuild from the loaded materials equals the rebuild from the
    # hand-built fixture materials (same chemistry, same source binding)
    from_loaded = rebuild_endpoint_graphs(ORDER_CHANGE, materials)
    from_fixture = rebuild_endpoint_graphs(ORDER_CHANGE, _materials_for(ORDER_CHANGE))
    assert from_loaded.content_sha256 == from_fixture.content_sha256
    assert (
        from_loaded.to_doc()[PROVENANCE_KEY]["endpoint_materials_sha256"]
        == from_fixture.to_doc()[PROVENANCE_KEY]["endpoint_materials_sha256"]
    )


def test_load_endpoint_materials_from_export_accepts_elements_list_fallback() -> None:
    # Given a doc whose atom_rows are absent but the aligned elements list is present
    doc = _synthetic_export_doc(ORDER_CHANGE)
    doc.pop("atom_rows")

    # When materials are loaded
    materials = load_endpoint_materials_from_export(doc)

    # Then the elements list supplies the map -> element table
    assert materials["r"][2]["element"] == "C"
    assert materials["p"][4]["element"] == "C"


@pytest.mark.parametrize(
    ("mutation", "expected_reason_fragment"),
    [
        (lambda doc: doc.pop("maps"), "maps_invalid"),
        (lambda doc: doc.pop("r_coordinates"), "coordinates_shape_invalid"),
        (
            lambda doc: doc.__setitem__("p_coordinates", doc["p_coordinates"][:-1]),
            "coordinates_shape_invalid",
        ),
        (
            lambda doc: doc.__setitem__(
                "r_coordinates",
                [row[:2] for row in doc["r_coordinates"]],  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]
            ),
            "coordinates_shape_invalid",
        ),
        (
            lambda doc: doc.__setitem__(
                "r_coordinates",
                [
                    ["x", "y", "z"] if index == 0 else row  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]
                    for index, row in enumerate(doc["r_coordinates"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                ],
            ),
            "coordinate_value_invalid",
        ),
        (
            lambda doc: doc.__setitem__("maps", [1, 1, 2, 3]),
            "maps_invalid",
        ),
        (
            lambda doc: doc.__setitem__("maps", [1, 2, 3, 0]),
            "maps_invalid",
        ),
        (
            lambda doc: (doc.pop("elements"), doc.pop("atom_rows")),
            "element_source_missing",
        ),
        (
            lambda doc: (
                doc.pop("atom_rows"),
                doc.__setitem__("elements", ["C", "C", "C"]),
            ),
            "element_source_missing",
        ),
    ],
)
def test_load_endpoint_materials_from_export_rejects_malformed_docs(
    mutation: object, expected_reason_fragment: str
) -> None:
    # Given a malformed export-like document
    doc = _synthetic_export_doc(ORDER_CHANGE)
    assert callable(mutation)
    mutation(doc)  # pyright: ignore[reportGeneralTypeIssues]

    # When materials are loaded
    # Then a typed EXPORT_SCHEMA_INVALID error names the malformed field
    with pytest.raises(RebuildInputError) as excinfo:
        load_endpoint_materials_from_export(doc)
    assert excinfo.value.code == CODE_EXPORT_SCHEMA
    assert any(expected_reason_fragment in reason for reason in excinfo.value.reasons)


def test_rebuild_rejects_unparseable_smiles_with_typed_error() -> None:
    # Given a reaction SMILES that does not parse under the strict G1 rules
    materials = _materials_for(ORDER_CHANGE)

    # When the rebuild is attempted
    # Then a typed error is raised before any bundle is produced
    with pytest.raises(RebuildInputError) as excinfo:
        rebuild_endpoint_graphs("not-a-reaction", materials)
    assert excinfo.value.code == CODE_SMILES_INVALID


def test_rebuild_rejects_unknown_materials_top_key_with_typed_error() -> None:
    # Given materials carrying a non-whitelisted top-level key
    materials = _materials_for(ORDER_CHANGE)
    materials["products"] = {}

    # When the rebuild is attempted
    # Then the materials trust boundary raises its typed schema error
    with pytest.raises(Exception) as excinfo:
        rebuild_endpoint_graphs(ORDER_CHANGE, materials)
    assert getattr(excinfo.value, "code", None) == CODE_MATERIALS_SCHEMA


def test_rebuild_api_supports_batch_of_synthetic_reactions() -> None:
    # Given two distinct synthetic reactions with their whitelisted materials
    batch = [
        (ORDER_CHANGE, _materials_for(ORDER_CHANGE)),
        (AROMATIC, _materials_for(AROMATIC)),
    ]

    # When each pair is rebuilt through the public API
    bundles = [
        rebuild_endpoint_graphs(smiles, materials) for smiles, materials in batch
    ]

    # Then every reaction yields its own stable, conserved bundle — the API
    # is batch-capable for the later demo-set golden validation
    assert len(bundles) == 2
    hashes = [bundle.content_sha256 for bundle in bundles]
    assert len(set(hashes)) == 2
    for bundle in bundles:
        assert bundle.conservation.element_conserved is True
        provenance = bundle.to_doc()[PROVENANCE_KEY]
        assert len(provenance["reaction_smiles_sha256"]) == 64
        assert len(provenance["endpoint_materials_sha256"]) == 64
        assert len(provenance["graph_payload_sha256"]) == 64
