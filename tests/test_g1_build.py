"""Offline end-to-end tests for the G1 build, coverage, and CLI stages.

Every fixture is a synthetic inventory Parquet built under ``tmp_path`` from
hand-written component records whose SMILES map labels obey the stored
invariant (the atom carrying map ``k`` sits at XYZ row ``k - 1``).  The tests
lock the decisive contracts: first-failure-wins rejection mapping, hand-verified
reaction-global map translation of the persisted index tables, soft/hard
bond-geometry verdicts, shard arithmetic, idempotent rejection-ledger appends,
deterministic coverage sampling, and byte-level rebuild determinism.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from pes2ts_core.cli import main
from pes2ts_core.g0.rejections import LEDGER_FILENAME
from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.build import (
    COVERAGE_FILENAME,
    EXIT_G1_BUILD_FAILED,
    MANIFEST_FILENAME,
    SUMMARY_COLUMNS,
    SUMMARY_FILENAME,
    build_g1,
    shard_name,
)
from pes2ts_core.g1.coverage import coverage_report, coverage_seed, manual_sample
from pes2ts_core.g1.document import CATEGORY_NAMES, CHANGE_SCHEMA_VERSION
from pes2ts_core.g1.verify import verify_g1
from pes2ts_core.utils.hashing import sha256_bytes
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

#: The asymmetric one-component/one-product fixture whose unique isomorphism
#: makes the global map translation hand-verifiable: local map 4 (Cl) is the
#: reaction-global map 7.
HAPPY_SMILES = "[F:1][C:2]([H:3])([Cl:7])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:7])[O:5].[H:6]"
SAME_SMILES = "[F:1][C:2]([H:3])([Cl:7])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:7])[O:5][H:6]"
WATER_SMILES = "[H:5][O:4][H:6]>>[H:5][O:4].[H:6]"

R0_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]"
R0_Z = (9, 6, 1, 17, 8, 1)
R0_X = (
    (0.0, 0.0, 0.0),
    (1.35, 0.0, 0.0),
    (1.9, 0.6, 0.3),
    (1.7, -1.2, 0.6),
    (2.6, 0.7, -0.2),
    (2.2, 1.5, 0.2),
)
P0_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5]"
P0_Z = (9, 6, 1, 17, 8)
P0_X = R0_X[:5]
P1_SMILES = "[H:1]"
P1_Z = (1,)
P1_X = ((0.0, 0.0, 0.0),)

#: A permuted water inventory (map 3 on the first atom), so the reactant
#: matching enumerates two isomorphisms and records ambiguity.
WATER_R0_SMILES = "[H:3][O:1][H:2]"
WATER_R0_Z = (8, 1, 1)
WATER_R0_X = ((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0))
WATER_P0_SMILES = "[H:1][O:2]"
WATER_P0_Z = (1, 8)
WATER_P0_X = ((0.96, 0.0, 0.0), (0.0, 0.0, 0.0))
WATER_P1_SMILES = "[H:1]"
WATER_P1_Z = (1,)
WATER_P1_X = ((0.0, 0.0, 0.0),)

ROW_FIELDS = (
    "reaction_id", "dataset_version", "reaction_smiles",
    "n_reactant_components", "n_product_components", "elements",
    "total_atoms_reactants", "total_atoms_products",
    "charge_total_reactants", "charge_total_products",
    "multiplicity_max", "components",
)


@dataclass(frozen=True, slots=True)
class _Component:
    """Recipe for one inventory component record."""

    tag: str
    smiles: str
    atomic_numbers: tuple[int, ...]
    coordinates: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class _Reaction:
    """Recipe for one synthetic inventory row."""

    reaction_id: str
    reaction_smiles: str
    components: tuple[_Component, ...]
    charge_total_reactants: int = 0
    charge_total_products: int = 0
    multiplicity_max: int = 1


R0 = _Component("R0", R0_SMILES, R0_Z, R0_X)
P0 = _Component("P0", P0_SMILES, P0_Z, P0_X)
P1 = _Component("P1", P1_SMILES, P1_Z, P1_X)
WATER_R0 = _Component("R0", WATER_R0_SMILES, WATER_R0_Z, WATER_R0_X)
WATER_P0 = _Component("P0", WATER_P0_SMILES, WATER_P0_Z, WATER_P0_X)
WATER_P1 = _Component("P1", WATER_P1_SMILES, WATER_P1_Z, WATER_P1_X)
#: One long C-Cl bond (1/5 fails): a soft geometry annotation, still valid.
SOFT_R0 = _Component(
    "R0", R0_SMILES, R0_Z,
    ((0.0, 0.0, 0.0), (1.35, 0.0, 0.0), (1.9, 0.6, 0.3),
     (5.0, -1.2, 0.6), (2.6, 0.7, -0.2), (2.2, 1.5, 0.2)),
)
#: Three of five C-bonds fail (>50%): a systematic corruption, hard-rejected.
HARD_R0 = _Component(
    "R0", R0_SMILES, R0_Z,
    ((0.0, 0.0, 0.0), (1.35, 0.0, 0.0), (5.0, 0.0, 0.0),
     (6.0, 0.0, 0.0), (5.0, 0.9, 0.0), (5.0, 1.86, 0.0)),
)
TAMPERED_ELEMENT_R0 = _Component("R0", R0_SMILES, (9, 6, 1, 1, 8, 1), R0_X)
TAMPERED_SMILES_R0 = _Component(
    "R0", "[F:1][C:2]([H:3])([Cl:4])", (9, 6, 1, 17),
    ((0.0, 0.0, 0.0), (1.35, 0.0, 0.0), (1.9, 0.6, 0.3), (1.7, -1.2, 0.6)),
)


def _happy(reaction_id: str, *, smiles: str = HAPPY_SMILES, r0: _Component = R0) -> _Reaction:
    """Return the asymmetric bond-breaking fixture (R0 optionally replaced)."""
    return _Reaction(reaction_id, smiles, (r0, P0, P1))


def _water(reaction_id: str) -> _Reaction:
    """Return the symmetric water-breaking fixture (ambiguous index)."""
    return _Reaction(reaction_id, WATER_SMILES, (WATER_R0, WATER_P0, WATER_P1))


def _component_record(component: _Component) -> dict[str, Any]:
    """Return the nested inventory record of one component."""
    return {
        "tag": component.tag,
        "smiles": component.smiles,
        "atomic_numbers": list(component.atomic_numbers),
        "coordinates": [list(row) for row in component.coordinates],
        "charge": 0,
        "multiplicity": 1,
        "EHG": [0.0, 0.0, 0.0],
    }


def _row(reaction: _Reaction) -> dict[str, Any]:
    """Return one full synthetic inventory row."""
    records = [_component_record(component) for component in reaction.components]
    reactants = [c for c in reaction.components if c.tag.startswith("R")]
    products = [c for c in reaction.components if c.tag.startswith("P")]
    elements = sorted(
        {ELEMENT_SYMBOLS[z - 1] for c in reaction.components for z in c.atomic_numbers}
    )
    return {
        "reaction_id": reaction.reaction_id,
        "dataset_version": "zenodo-1-rev1",
        "reaction_smiles": reaction.reaction_smiles,
        "n_reactant_components": len(reactants),
        "n_product_components": len(products),
        "elements": elements,
        "total_atoms_reactants": sum(len(c.atomic_numbers) for c in reactants),
        "total_atoms_products": sum(len(c.atomic_numbers) for c in products),
        "charge_total_reactants": reaction.charge_total_reactants,
        "charge_total_products": reaction.charge_total_products,
        "multiplicity_max": reaction.multiplicity_max,
        "components": records,
    }


def _config(root: Path) -> dict[str, Any]:
    """Return a minimal self-contained config rooted at *root*."""
    return {
        "source": {"zenodo_record": 1, "zenodo_revision": "1", "files": {}},
        "paths": {"interim": str(root / "interim"), "manifests": str(root / "manifests")},
        "split": {"seed": 42},
        "g1": {
            "shard_size": 1000,
            "match_cap": 10000,
            "max_candidates": 64,
            "bond_tolerance": 0.45,
            "neighborhood_shell": 1,
            "sample_size": 20,
            "sample_seed": 42,
        },
    }


def _write_inventory(config: dict[str, Any], reactions: Sequence[_Reaction]) -> Path:
    """Write the synthetic inventory Parquet and return its path."""
    rows = [_row(reaction) for reaction in reactions]
    table = pa.table({field: [row[field] for row in rows] for field in ROW_FIELDS})
    path = Path(config["paths"]["interim"]) / "inventory.parquet"
    write_parquet(path, table)
    return path


def _document(config: dict[str, Any], reaction_id: str, shard_size: int = 1000) -> dict[str, Any]:
    """Return the parsed reaction-change document of *reaction_id*."""
    path = Path(config["paths"]["interim"]) / "g1" / "reaction_change" / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the parsed unified-ledger records of *config*."""
    path = Path(config["paths"]["manifests"]) / LEDGER_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary_stub(reaction_id: str, **overrides: Any) -> dict[str, Any]:
    """Return one hand-written summary row for the pure coverage tests."""
    row: dict[str, Any] = {
        "reaction_id": reaction_id,
        "status": "valid",
        "failure_code": None,
        "n_formed": 1,
        "n_broken": 0,
        "n_order_changed": 0,
        "n_h_migration": 0,
        "core_size": 2,
        "shell_size": 2,
        "index_status": "unique",
        "pairing_status": "unique",
        "geometry_ok": True,
        "n_atoms": 4,
        **{name: False for name in CATEGORY_NAMES},
    }
    row.update(overrides)
    return row


def test_valid_document_shape_and_hand_verified_global_maps(tmp_path: Path) -> None:
    # Given: the asymmetric bond-breaking fixture with unique isomorphisms
    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001"), _water("RXN_0000000101")])

    # When
    result = build_g1(config)

    # Then: both reactions are valid and one document exists per row
    assert (result.n_total, result.n_valid, result.n_rejected) == (2, 2, 0)
    assert result.summary_path == tmp_path / "interim" / SUMMARY_FILENAME
    assert result.manifest_path == tmp_path / "manifests" / MANIFEST_FILENAME
    assert result.reactions_dir == tmp_path / "interim" / "g1" / "reaction_change"
    document = _document(config, "RXN_0000000001")

    # And: the top-level schema is complete and stable
    assert document["schema_version"] == CHANGE_SCHEMA_VERSION
    assert document["dataset_version"] == "zenodo-1-rev1"
    assert document["reaction_id"] == "RXN_0000000001"
    assert document["source"]["reaction_smiles_sha256"] == sha256_bytes(HAPPY_SMILES.encode("utf-8"))
    assert document["validation"] == {"status": "valid", "failure_code": None, "failure_detail": None}
    assert document["mapping"] == {
        "n_atoms": 12, "maps_unique": True, "map_sets_equal": True,
        "charge_consistent": True, "spin_consistent": True,
        "n_reactant_components": 1, "n_product_components": 2,
    }

    # And: the bond-change detail matches the broken O-H bond and its migration
    changes = document["bond_changes"]
    assert changes["formed"] == []
    assert changes["broken"] == [{"atoms": [5, 6], "order_r": 1.0, "order_p": None}]
    assert changes["order_changed"] == []
    assert changes["hydrogen_migration"] == [{"h": 6, "from": 5, "to": None}]
    assert changes["preview_counters"] == {
        "n_bonds_formed": 0, "n_bonds_broken": 1, "n_bond_order_changed": 0,
    }
    assert changes["preview_match"] is True
    assert document["reaction_center"] == {"core": [5, 6], "with_shell": [2, 5, 6]}
    assert document["categories"] == {
        "pure_formed": False, "pure_broken": True, "both": False,
        "order_change_only": False, "has_order_change": False,
        "h_migration": True, "multi_component": True,
    }
    # And: the private union-edge carrier never reaches the persisted JSON
    assert "_union_edges" not in json.dumps(document)

    # And: the R0 rows carry REACTION-GLOBAL maps (local map 4 = Cl -> global 7)
    r0 = document["reactants"][0]
    assert (r0["tag"], r0["index_base"], r0["n_atoms"]) == ("R0", 0, 6)
    assert (r0["status"], r0["truncated"], r0["n_candidates"]) == ("unique", False, 1)
    assert r0["element_check"] is True
    assert (r0["geometry_check_ok"], r0["geometry_worst"]) == (True, "")
    assert r0["rows"] == [
        {"local_index": 0, "map": 1, "element": "F", "global_index": 0},
        {"local_index": 1, "map": 2, "element": "C", "global_index": 1},
        {"local_index": 2, "map": 3, "element": "H", "global_index": 2},
        {"local_index": 3, "map": 7, "element": "Cl", "global_index": 3},
        {"local_index": 4, "map": 5, "element": "O", "global_index": 4},
        {"local_index": 5, "map": 6, "element": "H", "global_index": 5},
    ]
    assert r0["candidates"] == [[
        {"map": 1, "local_index": 0}, {"map": 2, "local_index": 1},
        {"map": 3, "local_index": 2}, {"map": 7, "local_index": 3},
        {"map": 5, "local_index": 4}, {"map": 6, "local_index": 5},
    ]]

    # And: the product side owns an independent global coordinate system
    p0, p1 = document["products"]
    assert (p0["tag"], p0["index_base"], p0["n_atoms"]) == ("P0", 0, 5)
    assert [row["map"] for row in p0["rows"]] == [1, 2, 3, 7, 5]
    assert [row["global_index"] for row in p0["rows"]] == [0, 1, 2, 3, 4]
    assert (p1["tag"], p1["index_base"]) == ("P1", 5)
    assert p1["rows"] == [{"local_index": 0, "map": 6, "element": "H", "global_index": 5}]
    assert document["ambiguity"] == {"index": "unique", "pairing": "unique"}

    # And: the symmetric water fixture records its two-isomorphism ambiguity
    water = _document(config, "RXN_0000000101")
    assert water["ambiguity"] == {"index": "ambiguous", "pairing": "unique"}
    assert water["reactants"][0]["status"] == "symmetric_ambiguous"
    assert water["reactants"][0]["n_candidates"] == 2
    assert water["categories"]["pure_broken"] is True
    assert water["categories"]["h_migration"] is True


def test_rejections_map_first_failure_to_one_typed_code(tmp_path: Path) -> None:
    # Given: one row per failure branch (parse, unchanged, component, element)
    config = _config(tmp_path)
    _write_inventory(config, [
        _Reaction("RXN_0000000001", "O>O", ()),
        _happy("RXN_0000000002", smiles=SAME_SMILES),
        _happy("RXN_0000000003", r0=TAMPERED_SMILES_R0),
        _happy("RXN_0000000004", r0=TAMPERED_ELEMENT_R0),
    ])

    # When
    result = build_g1(config)

    # Then: every row is rejected exactly once with its typed code
    assert (result.n_total, result.n_valid, result.n_rejected) == (4, 0, 4)
    parse_error = _document(config, "RXN_0000000001")
    assert parse_error["validation"] == {
        "status": "rejected", "failure_code": "G1_MAP_ERROR", "failure_detail": "no_arrow",
    }
    unchanged = _document(config, "RXN_0000000002")
    assert unchanged["validation"]["failure_code"] == "G1_NO_BOND_CHANGE"
    assert unchanged["validation"]["failure_detail"] == "formed=0 broken=0 order_changed=0"
    assert unchanged["mapping"]["n_atoms"] == 12
    mismatch = _document(config, "RXN_0000000003")
    assert mismatch["validation"]["failure_code"] == "G1_COMPONENT_MISMATCH"
    assert "0" in mismatch["validation"]["failure_detail"]
    element = _document(config, "RXN_0000000004")
    assert element["validation"]["failure_code"] == "G1_INDEX_MISMATCH"

    # And: invalid indexes never enter QC -- rejected docs carry empty tables
    for document in (parse_error, unchanged, mismatch, element):
        assert document["validation"]["status"] == "rejected"
        assert document["reactants"] == []
        assert document["products"] == []
        assert document["validation"]["failure_code"] is not None
    # And: a parse failure leaves the skeleton defaults untouched, while a
    # later rejection keeps the already-computed detail it does not invalidate
    assert parse_error["categories"] == {name: False for name in CATEGORY_NAMES}
    assert all(
        parse_error["bond_changes"][key] == []
        for key in ("formed", "broken", "order_changed", "hydrogen_migration")
    )
    assert mismatch["bond_changes"]["broken"] == [{"atoms": [5, 6], "order_r": 1.0, "order_p": None}]


def test_geometry_soft_annotation_and_hard_rejection(tmp_path: Path) -> None:
    # Given: one component with a single absurd bond and one mostly absurd
    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001", r0=SOFT_R0), _happy("RXN_0000000002", r0=HARD_R0)])

    # When
    result = build_g1(config)

    # Then: the soft violation stays valid and is annotated per component
    assert (result.n_total, result.n_valid, result.n_rejected) == (2, 1, 1)
    soft = _document(config, "RXN_0000000001")
    assert soft["validation"]["status"] == "valid"
    r0 = soft["reactants"][0]
    assert r0["geometry_check_ok"] is False
    assert "(2, 4)" in r0["geometry_worst"]
    assert soft["products"][0]["geometry_check_ok"] is True

    # And: the >50% violation is hard-rejected with its failing fraction
    hard = _document(config, "RXN_0000000002")
    assert hard["validation"]["failure_code"] == "G1_BOND_GEOMETRY"
    assert "3/5" in hard["validation"]["failure_detail"]
    assert hard["reactants"] == [] and hard["products"] == []

    # And: the summary reflects the soft geometry flag
    summary = read_parquet(result.summary_path).to_pylist()
    by_id = {row["reaction_id"]: row for row in summary}
    assert by_id["RXN_0000000001"]["geometry_ok"] is False
    assert by_id["RXN_0000000002"]["geometry_ok"] is True


def test_summary_parquet_row_contract(tmp_path: Path) -> None:
    # Given: a valid, a rejected, and an ambiguous fixture
    config = _config(tmp_path)
    _write_inventory(config, [
        _happy("RXN_0000000003"),
        _happy("RXN_0000000001", smiles=SAME_SMILES),
        _water("RXN_0000000002"),
    ])

    # When
    result = build_g1(config)

    # Then: the summary carries exactly the documented columns, sorted by id
    table = read_parquet(result.summary_path)
    assert set(table.column_names) == set(SUMMARY_COLUMNS)
    rows = table.to_pylist()
    assert [row["reaction_id"] for row in rows] == [
        "RXN_0000000001", "RXN_0000000002", "RXN_0000000003",
    ]
    valid = rows[2]
    assert valid["status"] == "valid"
    assert valid["failure_code"] is None
    assert (valid["n_formed"], valid["n_broken"], valid["n_order_changed"]) == (0, 1, 0)
    assert (valid["n_h_migration"], valid["core_size"], valid["shell_size"]) == (1, 2, 3)
    assert (valid["index_status"], valid["pairing_status"]) == ("unique", "unique")
    assert valid["geometry_ok"] is True
    assert valid["n_atoms"] == 12
    assert valid["pure_broken"] is True and valid["multi_component"] is True
    rejected = rows[0]
    assert rejected["status"] == "rejected"
    assert rejected["failure_code"] == "G1_NO_BOND_CHANGE"
    ambiguous = rows[1]
    assert ambiguous["index_status"] == "ambiguous"


def test_rejection_ledger_is_idempotent_across_reruns(tmp_path: Path) -> None:
    # Given: a fixture with two rejections
    config = _config(tmp_path)
    _write_inventory(config, [
        _Reaction("RXN_0000000001", "O>O", ()),
        _happy("RXN_0000000002", smiles=SAME_SMILES),
        _happy("RXN_0000000003"),
    ])

    # When: the same build runs twice
    first = build_g1(config)
    ledger_after_first = _ledger_records(config)
    second = build_g1(config)
    ledger_after_second = _ledger_records(config)

    # Then: exactly one g1_build entry per rejected reaction, no duplicates
    assert first.n_rejected == second.n_rejected == 2
    g1_entries = [record for record in ledger_after_first if record["stage"] == "g1_build"]
    assert len(g1_entries) == 2
    assert sorted(record["code"] for record in g1_entries) == ["G1_MAP_ERROR", "G1_NO_BOND_CHANGE"]
    assert all(record["source_pointer"].endswith(".json") for record in g1_entries)
    assert ledger_after_second == ledger_after_first


def test_shard_paths_by_numeric_suffix_and_hash_fallback(tmp_path: Path) -> None:
    # Given: ids in two numeric shards plus one non-numeric id
    config = _config(tmp_path)
    _write_inventory(config, [
        _happy("RXN_0000000001"),
        _happy("RXN_0000001001"),
        _happy("CUSTOM_ALPHA"),
    ])

    # When
    result = build_g1(config)

    # Then: the numeric suffix // shard_size names the five-digit shard
    assert (result.reactions_dir / "00000" / "RXN_0000000001.json").is_file()
    assert (result.reactions_dir / "00001" / "RXN_0000001001.json").is_file()
    # And: a non-numeric id falls back to the deterministic hash bucket
    bucket = (int(hashlib.sha256(b"CUSTOM_ALPHA").hexdigest(), 16) % 10**9) // 1000
    assert (result.reactions_dir / f"{bucket:05d}" / "CUSTOM_ALPHA.json").is_file()
    assert shard_name("RXN_0000001001", 1000) == "00001"
    assert shard_name("CUSTOM_ALPHA", 1000) == f"{bucket:05d}"
    assert [row["reaction_id"] for row in read_parquet(result.summary_path).to_pylist()] == [
        "CUSTOM_ALPHA", "RXN_0000000001", "RXN_0000001001",
    ]


def test_unknown_reaction_id_is_rejected(tmp_path: Path) -> None:
    # Given: a fixture that does not contain the requested id
    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001")])

    # When / Then: naming up to five unknown ids
    with pytest.raises(ValueError, match="RXN_0000000999"):
        build_g1(config, reaction_ids=["RXN_0000000999", "RXN_0000000001"])


def test_build_without_g1_config_section_uses_defaults(tmp_path: Path) -> None:
    # Given: an older config without the g1 section
    config = _config(tmp_path)
    config.pop("g1")
    _write_inventory(config, [_happy("RXN_0000000001")])

    # When
    result = build_g1(config)

    # Then: defaults apply, including one-second neighborhood shell expansion
    assert result.n_valid == 1
    assert _document(config, "RXN_0000000001")["reaction_center"]["with_shell"] == [2, 5, 6]


def test_map_and_row_inconsistencies_reject_with_machine_readable_details(tmp_path: Path) -> None:
    # Given: unmapped atoms, unequal map sets, and a charged and a triplet row
    config = _config(tmp_path)
    _write_inventory(config, [
        _Reaction("RXN_0000000001", "O>>O", ()),
        _Reaction("RXN_0000000002", "[CH4:1]>>[CH4:2]", ()),
        _Reaction("RXN_0000000003", HAPPY_SMILES, (R0, P0, P1), charge_total_reactants=1),
        _Reaction("RXN_0000000004", HAPPY_SMILES, (R0, P0, P1), multiplicity_max=3),
    ])

    # When
    result = build_g1(config)

    # Then: each branch rejects once with its stable token
    assert (result.n_valid, result.n_rejected) == (0, 4)
    unmapped = _document(config, "RXN_0000000001")["validation"]
    assert (unmapped["failure_code"], unmapped["failure_detail"]) == ("G1_MAP_ERROR", "unmapped_atom")
    map_sets = _document(config, "RXN_0000000002")["validation"]
    assert (map_sets["failure_code"], map_sets["failure_detail"]) == ("G1_MAP_ERROR", "map_sets_differ")
    charged = _document(config, "RXN_0000000003")["validation"]
    assert (charged["failure_code"], charged["failure_detail"]) == ("G1_MAP_ERROR", "charge_or_spin_inconsistent")
    triplet = _document(config, "RXN_0000000004")["validation"]
    assert (triplet["failure_code"], triplet["failure_detail"]) == ("G1_MAP_ERROR", "charge_or_spin_inconsistent")
    # And: the mapping block still records what was checked before the failure
    charged_mapping = _document(config, "RXN_0000000003")["mapping"]
    assert charged_mapping["maps_unique"] is True and charged_mapping["map_sets_equal"] is True
    assert charged_mapping["charge_consistent"] is False


def test_preview_conflict_rejects_when_preview_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: an imported preview forced to its documented -1 sentinel
    def unavailable(_row: Any) -> dict[str, int]:
        return {"n_bonds_formed": -1, "n_bonds_broken": -1, "n_bond_order_changed": -1}

    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001")])
    monkeypatch.setattr("pes2ts_core.g1.bond_changes.compute_bond_changes_preview", unavailable)
    monkeypatch.setattr("pes2ts_core.g1.document.compute_bond_changes_preview", unavailable)

    # When
    result = build_g1(config)

    # Then: the sentinel is never mistaken for a genuine zero and rejects typed
    assert (result.n_valid, result.n_rejected) == (0, 1)
    document = _document(config, "RXN_0000000001")
    assert document["validation"]["failure_code"] == "G1_PREVIEW_CONFLICT"
    assert document["validation"]["failure_detail"] == "preview_counters_differ"
    assert document["bond_changes"]["preview_counters"] == {
        "n_bonds_formed": -1, "n_bonds_broken": -1, "n_bond_order_changed": -1,
    }
    assert document["reactants"] == [] and document["products"] == []


def test_match_cap_truncation_marks_the_index_truncated(tmp_path: Path) -> None:
    # Given: the two-isomorphism water fixture with the cap set to exactly two
    config = _config(tmp_path)
    config["g1"]["match_cap"] = 2
    _write_inventory(config, [_water("RXN_0000000001")])

    # When
    result = build_g1(config)

    # Then: the component, document, and summary all report the truncation
    assert result.n_valid == 1
    document = _document(config, "RXN_0000000001")
    assert document["reactants"][0]["status"] == "truncated"
    assert document["reactants"][0]["truncated"] is True
    assert document["ambiguity"]["index"] == "truncated"
    summary = read_parquet(result.summary_path).to_pylist()[0]
    assert summary["index_status"] == "truncated"
    coverage = coverage_report([summary])
    assert coverage["categories"]["index_ambiguous"]["n_valid"] == 1


def test_coverage_report_math_on_hand_computed_fixture() -> None:
    # Given: two valid and two rejected rows with known category membership
    rows = [
        _summary_stub("a", pure_formed=True),
        _summary_stub("b", pure_broken=True, h_migration=True,
                      index_status="truncated", n_broken=1),
        _summary_stub("c", status="rejected", failure_code="G1_MAP_ERROR"),
        _summary_stub("d", status="rejected", failure_code="G1_MAP_ERROR"),
    ]

    # When
    report = coverage_report(rows)

    # Then: overall, by-code, and per-category math is exact
    assert report["overall"] == {"n_total": 4, "n_valid": 2, "n_rejected": 2, "coverage": 0.5}
    assert report["by_code"] == {"G1_MAP_ERROR": 2}
    assert report["categories"]["pure_formed"] == {"n_total": 1, "n_valid": 1, "coverage": 1.0}
    assert report["categories"]["pure_broken"] == {"n_total": 1, "n_valid": 1, "coverage": 1.0}
    assert report["categories"]["both"] == {"n_total": 0, "n_valid": 0, "coverage": 0.0}
    assert report["categories"]["index_ambiguous"] == {"n_total": 1, "n_valid": 1, "coverage": 1.0}
    assert report["categories"]["pairing_ambiguous"] == {"n_total": 0, "n_valid": 0, "coverage": 0.0}
    assert report["index_status"] == {"unique": 3, "ambiguous": 0, "truncated": 1}
    assert report["pairing_status"] == {"unique": 4, "ambiguous": 0}


def test_manual_sample_is_deterministic_and_seed_sensitive() -> None:
    # Given: three valid rows of one category and two of another
    rows = [
        _summary_stub("a", pure_formed=True),
        _summary_stub("b", pure_formed=True),
        _summary_stub("c", pure_formed=True),
        _summary_stub("d", pure_broken=True),
    ]

    # When
    first = manual_sample(rows, n=2, seed=7)
    second = manual_sample(rows, n=2, seed=7)
    wider = manual_sample(rows, n=3, seed=7)
    other_seed = manual_sample(rows, n=2, seed=8)

    # Then: the same seed gives the identical ranking, sorted by category+hash
    assert first == second
    expected_ranked = sorted(
        ("a", "b", "c"),
        key=lambda rid: hashlib.sha256(f"7:{rid}".encode("utf-8")).hexdigest(),
    )[:2]
    assert first == [{"reaction_id": "d", "category": "pure_broken"}] + [
        {"reaction_id": rid, "category": "pure_formed"}
        for rid in sorted(expected_ranked, key=lambda rid: hashlib.sha256(f"7:{rid}".encode("utf-8")).hexdigest())
    ]
    assert {entry["reaction_id"] for entry in first} <= {entry["reaction_id"] for entry in wider}
    assert "d" in {entry["reaction_id"] for entry in wider}

    # And: a different seed (or sample size) changes the pick
    assert coverage_seed({"g1": {"sample_seed": 7}, "split": {"seed": 42}}) == (7, "g1.sample_seed")
    assert coverage_seed({"split": {"seed": 42}}) == (42, "split.seed")
    assert coverage_seed({}) == (42, "split.seed")
    assert {entry["reaction_id"] for entry in other_seed} != {
        entry["reaction_id"] for entry in first if entry["category"] == "pure_formed"
    }


def test_rebuild_is_byte_identical_modulo_generated_at(tmp_path: Path) -> None:
    # Given: a mixed fixture built once
    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001"), _water("RXN_0000000002"), _Reaction("RXN_0000000003", "O>O", ())])
    first = build_g1(config)
    original_documents = {
        path.relative_to(first.reactions_dir): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(first.reactions_dir.rglob("*.json"))
    }
    summary_before = first.summary_path.read_bytes()
    manifest_before = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    coverage_before = json.loads((tmp_path / "manifests" / COVERAGE_FILENAME).read_text(encoding="utf-8"))
    ledger_before = _ledger_records(config)

    # When: the same input is built again
    _ = build_g1(config)

    # Then: every document matches modulo the volatile timestamp
    for relative, document in original_documents.items():
        fresh = json.loads((first.reactions_dir / relative).read_text(encoding="utf-8"))
        document.pop("generated_at")
        fresh.pop("generated_at")
        assert fresh == document
    # And: the non-timestamp artifacts are byte-identical
    assert first.summary_path.read_bytes() == summary_before
    manifest_after = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest_before.pop("generated_at")
    manifest_after.pop("generated_at")
    coverage_after = json.loads((tmp_path / "manifests" / COVERAGE_FILENAME).read_text(encoding="utf-8"))
    coverage_before.pop("generated_at")
    coverage_after.pop("generated_at")
    assert manifest_after == manifest_before
    assert coverage_after == coverage_before
    assert _ledger_records(config) == ledger_before


def test_cli_build_sample_and_verify_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Given: the CLI wired to the fixture config
    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001"), _water("RXN_0000000002")])
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When: build runs
    build_code = main(["g1", "build"])

    # Then: artifacts exist and the exit code is clean
    assert build_code == 0
    assert (tmp_path / "interim" / SUMMARY_FILENAME).is_file()
    manifest = json.loads((tmp_path / "manifests" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "g1_manifest_v1"
    assert (manifest["n_total"], manifest["n_valid"], manifest["n_rejected"]) == (2, 2, 0)
    assert manifest["n_files"] == 2
    assert manifest["shard_size"] == 1000
    assert manifest["config"]["neighborhood_shell"] == 1
    assert manifest["summary_sha256"] == sha256_bytes((tmp_path / "interim" / SUMMARY_FILENAME).read_bytes())
    coverage = json.loads((tmp_path / "manifests" / COVERAGE_FILENAME).read_text(encoding="utf-8"))
    assert coverage["schema_version"] == "g1_manifest_v1"
    assert coverage["seed"] == 42 and coverage["seed_source"] == "g1.sample_seed"

    # When: verify runs
    assert main(["g1", "verify"]) == 0

    # When: sample runs
    capsys.readouterr()
    sample_code = main(["g1", "sample", "--n", "3"])
    output = capsys.readouterr().out

    # Then: each printed line carries the checked fields
    assert sample_code == 0
    lines = [line for line in output.splitlines() if line.strip()]
    assert lines
    assert all("formed=" in line and "index=" in line and "pairing=" in line for line in lines)

    # And: category filtering keeps only that category
    capsys.readouterr()
    assert main(["g1", "sample", "--category", "pure_broken"]) == 0
    filtered = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert filtered and all(" pure_broken " in line for line in filtered)


def test_cli_verify_fails_on_tampered_and_missing_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a built tree
    config = _config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001")])
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)
    assert main(["g1", "build"]) == 0
    document_path = tmp_path / "interim" / "g1" / "reaction_change" / "00000" / "RXN_0000000001.json"

    # When: the schema version is tampered
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["schema_version"] = "not_g1_v1"
    write_json(document_path, document)

    # Then: verify reports the violation and exits 22
    assert main(["g1", "verify"]) == EXIT_G1_BUILD_FAILED

    # When: the document is restored but deleted entirely
    document["schema_version"] = CHANGE_SCHEMA_VERSION
    write_json(document_path, document)
    assert main(["g1", "verify"]) == 0
    document_path.unlink()
    assert main(["g1", "verify"]) == EXIT_G1_BUILD_FAILED

    # And: a stray document is detected too
    write_json(document_path, document)
    stray = document_path.parent / "RXN_0000000999.json"
    write_json(stray, document)
    assert main(["g1", "verify"]) == EXIT_G1_BUILD_FAILED
    stray.unlink()
    assert verify_g1(config).problems == ()


def test_cli_missing_inputs_exit_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a fixture tree without inventory, cohort, or summary
    config = _config(tmp_path)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # Then: every missing input maps to the shared exit code 3
    assert main(["g1", "build"]) == 3
    assert main(["g1", "sample"]) == 3
    assert main(["g1", "verify"]) == 3

    # Given: an inventory but no cohort artifact yet
    _write_inventory(config, [_happy("RXN_0000000001")])
    assert main(["g1", "build", "--cohort", "trial"]) == 3


def test_cli_cohort_subset_and_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: three inventory rows and a trial cohort of one member
    config = _config(tmp_path)
    _write_inventory(config, [
        _happy("RXN_0000000001"),
        _happy("RXN_0000000002"),
        _happy("RXN_0000000003"),
    ])
    write_json(
        Path(config["paths"]["interim"]) / "cohort_trial.json",
        {"schema_version": "g0_manifest_v1", "cohort": "trial", "members": ["RXN_0000000002"]},
    )
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When: the trial cohort is built
    assert main(["g1", "build", "--cohort", "trial"]) == 0

    # Then: only the cohort member got a document
    manifest = json.loads((tmp_path / "manifests" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["n_total"] == 1
    assert (tmp_path / "interim" / "g1" / "reaction_change" / "00000" / "RXN_0000000002.json").is_file()
    assert not (tmp_path / "interim" / "g1" / "reaction_change" / "00000" / "RXN_0000000001.json").exists()

    # When: a fresh tree builds only the first sorted id
    fresh = tmp_path / "fresh"
    fresh_config = _config(fresh)
    _write_inventory(fresh_config, [
        _happy("RXN_0000000003"), _happy("RXN_0000000001"), _happy("RXN_0000000002"),
    ])
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: fresh_config)
    assert main(["g1", "build", "--limit", "1"]) == 0
    fresh_manifest = json.loads((fresh / "manifests" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert fresh_manifest["n_total"] == 1
    assert (fresh / "interim" / "g1" / "reaction_change" / "00000" / "RXN_0000000001.json").is_file()

#: Ethane: 2C single bond + 4 explicit H.
ETHANE_SMILES = "[C:1]([H:3])([H:4])[C:2]([H:5])([H:6])"
ETHANE_Z = (6, 6, 1, 1, 1, 1)
ETHANE_X = (
    (0.00, 0.00, 0.00),  # C1 map 1
    (1.54, 0.00, 0.00),  # C2 map 2
    (-0.50, 0.90, 0.00), # H map 3
    (-0.50, -0.90, 0.00),# H map 4
    (2.04, 0.90, 0.00),  # H map 5
    (2.04, -0.90, 0.00), # H map 6
)

#: Ethene: same atoms, C=C double bond.
ETHENE_SMILES = "[C:1]([H:3])([H:4])=[C:2]([H:5])([H:6])"
ETHENE_Z = (6, 6, 1, 1, 1, 1)
ETHENE_X = (
    (0.00, 0.00, 0.00),  # C1 map 1
    (1.34, 0.00, 0.00),  # C2 map 2
    (-0.50, 0.90, 0.00), # H map 3
    (-0.50, -0.90, 0.00),# H map 4
    (1.84, 0.90, 0.00),  # H map 5
    (1.84, -0.90, 0.00), # H map 6
)

ETHANE_R0 = _Component("R0", ETHANE_SMILES, ETHANE_Z, ETHANE_X)
ETHENE_P0 = _Component("P0", ETHENE_SMILES, ETHENE_Z, ETHENE_X)


def test_order_change_only_categories_e2e(tmp_path: Path) -> None:
    # Given: an ethane-to-ethene single→double bond order change
    config = _config(tmp_path)
    _write_inventory(config, [
        _Reaction(
            "RXN_0000000001",
            f"{ETHANE_SMILES}>>{ETHENE_SMILES}",
            (ETHANE_R0, ETHENE_P0),
        ),
    ])

    # When
    result = build_g1(config)

    # Then: the reaction is valid with the expected bond-change counts
    assert (result.n_total, result.n_valid, result.n_rejected) == (1, 1, 0)
    document = _document(config, "RXN_0000000001")
    changes = document["bond_changes"]
    assert len(changes["formed"]) == 1
    assert len(changes["broken"]) == 1
    assert len(changes["order_changed"]) == 1
    assert changes["formed"][0]["atoms"] == changes["broken"][0]["atoms"]
    assert changes["formed"][0]["atoms"] == changes["order_changed"][0]["atoms"]

    # And: the non-disjoint category flags are all true
    categories = document["categories"]
    assert categories["order_change_only"] is True
    assert categories["both"] is True
    assert categories["has_order_change"] is True

    # And: the summary parquet row carries the same booleans
    summary_rows = read_parquet(result.summary_path).to_pylist()
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert summary["order_change_only"] is True
    assert summary["both"] is True
    assert summary["has_order_change"] is True
