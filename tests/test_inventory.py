"""Offline tests for the TS-free reactant/product inventory.

All fixtures are synthetic and built inside ``tmp_path``: a tiny HDF5 file
following the combined Reaction-QM layout (reactions 1/2 valid, 3 atom-count
mismatch, 4 charged, 6 HDF5-only, 7 element mismatch, 8 map mismatch, 9
triplet) and a CSV carrying the matching rows plus a CSV-only id and one
reader-level ``NO_ARROW`` row.  The fixture config declares both files with
their real MD5s, so ``verify_sources`` passes without any monkeypatching.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from pes2ts_core.cli import main
from pes2ts_core.g0.fetch import ChecksumMismatch
from pes2ts_core.g0.inventory import build_inventory
from pes2ts_core.g0.rejections import LEDGER_FILENAME, SUMMARY_FILENAME
from pes2ts_core.utils.hashing import md5_file
from pes2ts_core.utils.parquet_io import read_parquet

WATER_Z = (8, 1, 1)
WATER_X = ((0.0, 0.0, 0.117), (0.0, 0.757, -0.469), (0.0, -0.757, -0.469))
METHANE_Z = (6, 1, 1, 1, 1)
METHANE_X = (
    (0.0, 0.0, 0.0),
    (0.63, 0.63, 0.63),
    (-0.63, -0.63, 0.63),
    (0.63, -0.63, -0.63),
    (-0.63, 0.63, -0.63),
)
DEFAULT_EHG = (-76.4, -76.3, -76.2)
TS_LABEL = "TS-label"

BUNDLE = "bundle_a"
H5_NAME = "fixture_TZVP.h5"
CSV_NAME = "fixture_reaction_info.csv"
IRC_NAME = "fixture_TZVP_IRC.h5"

CSV_HEADER = "reaction_id,reaction_smiles,dE_dagger,dE,dH_dagger,dH,dG_dagger,dG"
CSV_ENERGIES = "-1.0,-2.0,-3.0,-4.0,-5.0,-6.0"
CSV_ROWS: tuple[tuple[str, str], ...] = (
    ("RXN_0000000001", "O>>O"),
    ("RXN_0000000002", "C.O>>C.O"),
    ("RXN_0000000003", "O>>O"),
    ("RXN_0000000004", "O>>O"),
    ("RXN_0000000005", "O>>O"),
    ("RXN_0000000007", "O>>O"),
    ("RXN_0000000008", "[CH3:1][OH:2]>>[CH2:1]=[O:3]"),
    ("RXN_0000000009", "O>>O"),
    ("RXN_0000000010", "O>O"),
)
EXPECTED_HISTOGRAM: dict[str, int] = {
    "ATOM_COUNT_MISMATCH": 1,
    "BAD_SMILES": 1,
    "CHARGE_NOT_NEUTRAL": 1,
    "ELEMENT_MISMATCH": 1,
    "MISSING_IN_CSV": 1,
    "MISSING_IN_H5": 1,
    "NO_ARROW": 1,
    "SPIN_NOT_SINGLET": 1,
}
COMPONENT_FIELDS = frozenset(
    {"tag", "smiles", "atomic_numbers", "coordinates", "charge", "multiplicity", "EHG"}
)


@dataclass(frozen=True, slots=True)
class _Species:
    """Recipe for one species node of the synthetic HDF5 fixture."""

    tag: str
    smiles: str = "O"
    atomic_numbers: tuple[int, ...] = WATER_Z
    coordinates: tuple[tuple[float, ...], ...] = WATER_X
    charge: int = 0
    multiplicity: int = 1


def _add_species(reaction: h5py.Group, spec: _Species) -> None:
    species = reaction.create_group(spec.tag)
    species.create_dataset("smiles", data=np.bytes_(spec.smiles))
    species.create_dataset("EHG", data=np.asarray(DEFAULT_EHG, dtype=np.float64))
    species.create_dataset("charge", data=spec.charge)
    species.create_dataset("multiplicity", data=spec.multiplicity)
    species.create_dataset(
        "atomic_numbers", data=np.asarray(spec.atomic_numbers, dtype=np.int64)
    )
    species.create_dataset(
        "coordinates", data=np.asarray(spec.coordinates, dtype=np.float64)
    )


def _add_reaction(
    bundle: h5py.Group, reaction_id: str, species: Sequence[_Species]
) -> None:
    reaction = bundle.create_group(reaction_id)
    for spec in species:
        _add_species(reaction, spec)
    _add_species(reaction, _Species(tag="TS", smiles=TS_LABEL))


def _build_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        bundle = handle.create_group(BUNDLE)
        _add_reaction(bundle, "RXN_0000000001", (_Species("R0"), _Species("P0")))
        _add_reaction(
            bundle,
            "RXN_0000000002",
            (
                _Species("R0", "C", METHANE_Z, METHANE_X),
                _Species("R1"),
                _Species("P0", "C", METHANE_Z, METHANE_X),
                _Species("P1"),
            ),
        )
        _add_reaction(
            bundle,
            "RXN_0000000003",
            (
                _Species("R0"),
                _Species(
                    "P0",
                    atomic_numbers=(8, 1, 1, 1),
                    coordinates=(*WATER_X, (0.0, 0.0, 1.0)),
                ),
            ),
        )
        _add_reaction(bundle, "RXN_0000000004", (_Species("R0", charge=1), _Species("P0")))
        _add_reaction(bundle, "RXN_0000000006", (_Species("R0"), _Species("P0")))
        _add_reaction(
            bundle,
            "RXN_0000000007",
            (
                _Species("R0"),
                _Species(
                    "P0",
                    "N",
                    (7, 1, 1),
                    ((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, -0.8, 0.0)),
                ),
            ),
        )
        _add_reaction(bundle, "RXN_0000000008", (_Species("R0"), _Species("P0")))
        _add_reaction(
            bundle, "RXN_0000000009", (_Species("R0", multiplicity=3), _Species("P0"))
        )


def _write_csv(path: Path) -> None:
    lines = [CSV_HEADER]
    lines.extend(
        f"{reaction_id},{smiles},{CSV_ENERGIES}" for reaction_id, smiles in CSV_ROWS
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture_config(tmp_path: Path) -> dict[str, Any]:
    raw = tmp_path / "raw"
    raw.mkdir()
    h5_path = raw / H5_NAME
    csv_path = raw / CSV_NAME
    _build_h5(h5_path)
    _write_csv(csv_path)
    files = {
        H5_NAME: {
            "filename": H5_NAME,
            "url": "https://example.invalid/h5",
            "md5": md5_file(h5_path),
        },
        CSV_NAME: {
            "filename": CSV_NAME,
            "url": "https://example.invalid/csv",
            "md5": md5_file(csv_path),
        },
    }
    return {
        "source": {"zenodo_record": 18551029, "zenodo_revision": "1", "files": files},
        "paths": {
            "raw": str(raw),
            "interim": str(tmp_path / "interim"),
            "manifests": str(tmp_path / "manifests"),
        },
    }


def test_build_inventory_keeps_valid_rows_and_excludes_ts(tmp_path: Path) -> None:
    # Given: the synthetic fixture (2 valid reactions plus one reaction per
    # rejection class)
    config = _fixture_config(tmp_path)

    # When
    result = build_inventory(config)

    # Then: only the two valid reactions survive
    assert result.n_rows == 2
    assert result.n_rejections == 8
    assert result.parquet_path == tmp_path / "interim" / "inventory.parquet"
    assert result.manifest_path == tmp_path / "manifests" / "inventory_manifest.json"

    table = read_parquet(result.parquet_path)
    assert table.num_rows == 2
    # And: the G0 gate holds - no TS column exists anywhere in the schema
    assert not any(
        name.lower() == "ts" or name.lower().startswith("ts_")
        for name in table.column_names
    )

    rows = {row["reaction_id"]: row for row in table.to_pylist()}
    assert set(rows) == {"RXN_0000000001", "RXN_0000000002"}

    unimolecular = rows["RXN_0000000001"]
    assert unimolecular["dataset_version"] == "zenodo-18551029-rev1"
    assert unimolecular["reaction_smiles"] == "O>>O"
    assert unimolecular["n_reactant_components"] == 1
    assert unimolecular["n_product_components"] == 1
    assert unimolecular["elements"] == ["H", "O"]
    assert unimolecular["total_atoms_reactants"] == 3
    assert unimolecular["total_atoms_products"] == 3
    assert unimolecular["charge_total_reactants"] == 0
    assert unimolecular["charge_total_products"] == 0
    assert unimolecular["multiplicity_max"] == 1
    assert [component["tag"] for component in unimolecular["components"]] == [
        "P0",
        "R0",
    ]

    component = unimolecular["components"][1]
    assert set(component) == COMPONENT_FIELDS
    assert component == {
        "tag": "R0",
        "smiles": "O",
        "atomic_numbers": [8, 1, 1],
        "coordinates": [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
        "charge": 0,
        "multiplicity": 1,
        "EHG": [-76.4, -76.3, -76.2],
    }

    bimolecular = rows["RXN_0000000002"]
    assert bimolecular["n_reactant_components"] == 2
    assert bimolecular["n_product_components"] == 2
    assert bimolecular["elements"] == ["C", "H", "O"]
    assert bimolecular["total_atoms_reactants"] == 8
    assert bimolecular["total_atoms_products"] == 8

    # And: no TS tag or TS label round-trips into the parquet rows
    assert TS_LABEL not in json.dumps(table.to_pylist())
    assert all(
        component["tag"] != "TS"
        for row in table.to_pylist()
        for component in row["components"]
    )


def test_rejection_ledger_and_manifest_record_every_exclusion(tmp_path: Path) -> None:
    # Given / When
    config = _fixture_config(tmp_path)
    result = build_inventory(config)

    # Then: the manifest summarizes rows, sources, and the rejection histogram
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "g0_manifest_v1"
    assert manifest["dataset_version"] == "zenodo-18551029-rev1"
    assert manifest["n_rows"] == 2
    assert manifest["n_reactions_in_csv"] == 8
    assert manifest["n_reactions_in_h5"] == 8
    assert manifest["rejection_histogram"] == EXPECTED_HISTOGRAM
    assert "generated_at" in manifest

    # And: every exclusion has exactly one ledger line with full provenance
    manifests_dir = tmp_path / "manifests"
    records = [
        json.loads(line)
        for line in (manifests_dir / LEDGER_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert Counter(record["code"] for record in records) == Counter(
        EXPECTED_HISTOGRAM
    )
    # CSV-reader rejections and inventory rejections share the one ledger
    assert Counter(record["stage"] for record in records) == {
        "build_inventory": 7,
        "read_reaction_info_csv": 1,
    }
    assert all(record["detail"] and record["source_pointer"] for record in records)

    by_code = {record["code"]: record for record in records}
    assert by_code["CHARGE_NOT_NEUTRAL"]["reaction_id"] == "RXN_0000000004"
    assert by_code["SPIN_NOT_SINGLET"]["reaction_id"] == "RXN_0000000009"
    assert by_code["ATOM_COUNT_MISMATCH"]["reaction_id"] == "RXN_0000000003"
    assert by_code["ELEMENT_MISMATCH"]["reaction_id"] == "RXN_0000000007"
    assert by_code["BAD_SMILES"]["reaction_id"] == "RXN_0000000008"
    assert by_code["MISSING_IN_H5"]["reaction_id"] == "RXN_0000000005"
    assert by_code["MISSING_IN_CSV"]["reaction_id"] == "RXN_0000000006"
    assert "atom map number sets differ" in by_code["BAD_SMILES"]["detail"]

    # And: the summary mirrors the histogram
    summary = json.loads((manifests_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["schema_version"] == "g0_manifest_v1"
    assert summary["total"] == 8
    assert summary["by_code"] == EXPECTED_HISTOGRAM


def test_missing_main_h5_raises_file_not_found(tmp_path: Path) -> None:
    # Given: a fixture whose main HDF5 was removed from the raw tree
    config = _fixture_config(tmp_path)
    (tmp_path / "raw" / H5_NAME).unlink()

    # When / Then: the strict error tells the operator to run g0 fetch first
    with pytest.raises(FileNotFoundError, match="g0 fetch"):
        build_inventory(config)

    assert not (tmp_path / "interim" / "inventory.parquet").exists()
    assert not (tmp_path / "manifests" / "inventory_manifest.json").exists()


def test_corrupt_source_propagates_checksum_mismatch(tmp_path: Path) -> None:
    # Given: a fixture whose CSV was mutated after the config md5 was taken
    config = _fixture_config(tmp_path)
    csv_path = tmp_path / "raw" / CSV_NAME
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    # When / Then
    with pytest.raises(ChecksumMismatch):
        build_inventory(config)


def test_irc_source_entry_is_never_selected_as_main_h5(tmp_path: Path) -> None:
    # Given: an additional configured IRC archive in the raw tree
    config = _fixture_config(tmp_path)
    irc_path = tmp_path / "raw" / IRC_NAME
    irc_path.write_bytes(b"ground-truth path payload that is not HDF5")
    config["source"]["files"][IRC_NAME] = {
        "filename": IRC_NAME,
        "url": "https://example.invalid/irc",
        "md5": md5_file(irc_path),
    }

    # When / Then: the main archive is still the only HDF5 used
    result = build_inventory(config)

    assert result.n_rows == 2


def test_build_inventory_is_repeatable(tmp_path: Path) -> None:
    # Given
    config = _fixture_config(tmp_path)
    manifests_dir = tmp_path / "manifests"

    # When: the same config is built twice
    first = build_inventory(config)
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    first_ledger = (manifests_dir / LEDGER_FILENAME).read_text(encoding="utf-8")

    second = build_inventory(config)
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    second_ledger = (manifests_dir / LEDGER_FILENAME).read_text(encoding="utf-8")

    # Then: artifacts are overwritten, not appended, and stay identical except
    # for the volatile generation timestamp
    assert (first.n_rows, first.n_rejections) == (second.n_rows, second.n_rejections)
    first_manifest.pop("generated_at")
    second_manifest.pop("generated_at")
    assert first_manifest == second_manifest
    assert first_ledger == second_ledger


def test_cli_inventory_writes_artifacts_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the CLI whose config loader returns the fixture config
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "inventory"])

    # Then
    assert exit_code == 0
    assert (tmp_path / "interim" / "inventory.parquet").is_file()
    assert (tmp_path / "manifests" / "inventory_manifest.json").is_file()
