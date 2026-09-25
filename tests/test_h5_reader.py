"""Unit tests for the combined Reaction-QM HDF5 reader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

from pes2ts_core.g0.reader import (
    H5SchemaError,
    SpeciesRecord,
    iter_h5_reactions,
    read_h5_reaction,
    read_h5_species,
)

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
REACTION_PATH = f"{BUNDLE}/RXN_0000000001"
SPECIES_PATH = f"{REACTION_PATH}/R0"


def _add_species(
    reaction: h5py.Group,
    tag: str,
    *,
    smiles: str = "O",
    atomic_numbers: Sequence[int] = WATER_Z,
    coordinates: Sequence[Sequence[float]] = WATER_X,
    EHG: Sequence[float] = DEFAULT_EHG,
    charge: int = 0,
    multiplicity: int = 1,
    string_encoding: str = "fixed",
) -> h5py.Group:
    species = reaction.create_group(tag)
    if string_encoding == "vlen":
        species.create_dataset("smiles", data=smiles, dtype=h5py.string_dtype())
    else:
        species.create_dataset("smiles", data=np.bytes_(smiles))
    species.create_dataset("EHG", data=np.asarray(EHG, dtype=np.float64))
    species.create_dataset("charge", data=charge)
    species.create_dataset("multiplicity", data=multiplicity)
    species.create_dataset(
        "atomic_numbers", data=np.asarray(atomic_numbers, dtype=np.int64)
    )
    species.create_dataset(
        "coordinates", data=np.asarray(coordinates, dtype=np.float64)
    )
    return species


def _build_one_reaction_file(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        reaction = handle.create_group(REACTION_PATH)
        _add_species(reaction, "R0", smiles="O")
        _add_species(reaction, "P0", smiles="O")
        _add_species(reaction, "TS", smiles=TS_LABEL)
    return path


def _build_two_bundle_fixture(path: Path, *, string_encoding: str = "fixed") -> Path:
    with h5py.File(path, "w") as handle:
        uni = handle.create_group("B3LYPD3_TZVP_100_200/RXN_0000000001")
        _add_species(uni, "R0", smiles="O", string_encoding=string_encoding)
        _add_species(uni, "P0", smiles="O", string_encoding=string_encoding)
        _add_species(uni, "TS", smiles=TS_LABEL, string_encoding=string_encoding)

        bi = handle.create_group("bundle_a/RXN_0000000002")
        for tag, smiles, z, xyz in (
            ("R0", "C", METHANE_Z, METHANE_X),
            ("R1", "O", WATER_Z, WATER_X),
            ("P0", "C", METHANE_Z, METHANE_X),
            ("P1", "O", WATER_Z, WATER_X),
            ("TS", TS_LABEL, METHANE_Z, METHANE_X),
        ):
            _add_species(
                bi,
                tag,
                smiles=smiles,
                atomic_numbers=z,
                coordinates=xyz,
                string_encoding=string_encoding,
            )
    return path


@pytest.mark.parametrize("string_encoding", ["fixed", "vlen"])
def test_two_bundles_yield_sorted_reactions_with_all_species(
    tmp_path: Path, string_encoding: str
) -> None:
    # Given: two bundle groups (the real-style name and an arbitrary one), one
    # unimolecular and one bimolecular reaction
    path = _build_two_bundle_fixture(
        tmp_path / "two_bundles.h5", string_encoding=string_encoding
    )

    # When
    results = list(iter_h5_reactions(path))

    # Then: bundle-name order and sorted per-bundle keys yield sorted IDs
    assert [reaction_id for reaction_id, _, _ in results] == [
        "RXN_0000000001",
        "RXN_0000000002",
    ]

    # And: the unimolecular reaction keeps R/P separate from TS, sorted by tag
    # (lexicographically, so P0 precedes R0)
    _, uni_rp, uni_ts = results[0]
    assert [record.tag for record in uni_rp] == ["P0", "R0"]
    assert [record.tag for record in uni_ts] == ["TS"]
    assert all(
        isinstance(record, SpeciesRecord) for record in (*uni_rp, *uni_ts)
    )
    r0 = next(record for record in uni_rp if record.tag == "R0")
    assert r0.smiles == "O"
    assert np.array_equal(r0.atomic_numbers, WATER_Z)
    assert np.issubdtype(r0.atomic_numbers.dtype, np.integer)
    assert r0.coordinates.shape == (3, 3)
    assert r0.coordinates.dtype == np.float64
    assert np.allclose(r0.coordinates, WATER_X)
    assert r0.charge == 0
    assert r0.multiplicity == 1
    assert r0.EHG.shape == (3,)
    assert r0.EHG.dtype == np.float64
    assert np.allclose(r0.EHG, DEFAULT_EHG)
    # And: the TS smiles is a label kept verbatim, never RDKit-interpreted
    assert uni_ts[0].smiles == TS_LABEL

    # And: the bimolecular reaction discovers R0, R1, P0, P1 without assuming
    # a component count
    _, bi_rp, bi_ts = results[1]
    assert [record.tag for record in bi_rp] == ["P0", "P1", "R0", "R1"]
    assert [record.tag for record in bi_ts] == ["TS"]
    methane = next(record for record in bi_rp if record.tag == "R0")
    water = next(record for record in bi_rp if record.tag == "R1")
    assert np.array_equal(methane.atomic_numbers, METHANE_Z)
    assert methane.coordinates.shape == (5, 3)
    assert water.smiles == "O"


def test_coordinates_wrong_shape_names_reaction_and_dataset(tmp_path: Path) -> None:
    # Given: a reaction whose R0 coordinates have shape (3, 2)
    path = _build_one_reaction_file(tmp_path / "bad_coordinates.h5")
    with h5py.File(path, "r+") as handle:
        del handle[f"{SPECIES_PATH}/coordinates"]
        handle[SPECIES_PATH].create_dataset("coordinates", data=np.zeros((3, 2)))

    # When / Then
    with pytest.raises(H5SchemaError) as excinfo:
        _ = list(iter_h5_reactions(path))
    message = str(excinfo.value)
    assert "RXN_0000000001" in message
    assert "'coordinates'" in message
    assert "(3, 2)" in message


def test_ehg_wrong_shape_names_reaction_and_dataset(tmp_path: Path) -> None:
    # Given: a reaction whose R0 EHG has shape (2,)
    path = _build_one_reaction_file(tmp_path / "bad_ehg.h5")
    with h5py.File(path, "r+") as handle:
        del handle[f"{SPECIES_PATH}/EHG"]
        handle[SPECIES_PATH].create_dataset("EHG", data=np.zeros(2))

    # When / Then
    with pytest.raises(H5SchemaError) as excinfo:
        _ = list(iter_h5_reactions(path))
    message = str(excinfo.value)
    assert "RXN_0000000001" in message
    assert "'EHG'" in message
    assert "(2,)" in message


def test_missing_atomic_numbers_names_reaction_and_dataset(tmp_path: Path) -> None:
    # Given: a species without the required atomic_numbers dataset
    path = _build_one_reaction_file(tmp_path / "missing_atomic_numbers.h5")
    with h5py.File(path, "r+") as handle:
        del handle[f"{SPECIES_PATH}/atomic_numbers"]

    # When / Then
    with pytest.raises(H5SchemaError) as excinfo:
        _ = list(iter_h5_reactions(path))
    message = str(excinfo.value)
    assert "RXN_0000000001" in message
    assert "'R0'" in message
    assert "missing dataset 'atomic_numbers'" in message


def test_unknown_species_tag_names_reaction_and_tag(tmp_path: Path) -> None:
    # Given: a reaction group carrying an unexpected species tag
    path = _build_one_reaction_file(tmp_path / "unknown_tag.h5")
    with h5py.File(path, "r+") as handle:
        handle[REACTION_PATH].create_group("X0")

    # When / Then
    with pytest.raises(H5SchemaError) as excinfo:
        _ = list(iter_h5_reactions(path))
    message = str(excinfo.value)
    assert "RXN_0000000001" in message
    assert "'X0'" in message


def test_iter_h5_reactions_is_lazy_and_opens_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a recording proxy around h5py.File
    path = _build_one_reaction_file(tmp_path / "modes.h5")
    modes: list[object] = []
    real_file = h5py.File

    def recording_file(*args: object, **kwargs: object) -> h5py.File:
        modes.append(args[1] if len(args) > 1 else kwargs.get("mode"))
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", recording_file)

    # When: creating the generator...
    generator = iter_h5_reactions(path)

    # Then: nothing is opened yet
    assert modes == []

    # When: consuming the first item...
    assert next(generator)[0] == "RXN_0000000001"

    # Then: the file was opened exactly once, read-only
    assert modes == ["r"]

    # And: the generator exhausts after the single reaction
    assert list(generator) == []


def test_non_reaction_keys_are_skipped(tmp_path: Path) -> None:
    # Given: bundle-level metadata and a top-level dataset next to one reaction
    path = _build_one_reaction_file(tmp_path / "lenient.h5")
    with h5py.File(path, "r+") as handle:
        handle.create_dataset("notes", data="top-level metadata")
        handle[BUNDLE].create_group("metadata")
        handle[BUNDLE].create_dataset("version", data=1)

    # When
    results = list(iter_h5_reactions(path))

    # Then: only the RXN_ key is treated as a reaction
    assert [reaction_id for reaction_id, _, _ in results] == ["RXN_0000000001"]


def test_invalid_rxn_key_raises_schema_error(tmp_path: Path) -> None:
    # Given: an RXN_-prefixed key that cannot be normalized
    path = tmp_path / "bad_key.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group(f"{BUNDLE}/RXN_not_a_number")

    # When / Then
    with pytest.raises(H5SchemaError) as excinfo:
        _ = list(iter_h5_reactions(path))
    message = str(excinfo.value)
    assert "RXN_not_a_number" in message
    assert BUNDLE in message


def test_duplicate_reaction_across_bundles_names_both(tmp_path: Path) -> None:
    # Given: the same reaction ID in two different bundles
    path = tmp_path / "duplicate.h5"
    with h5py.File(path, "w") as handle:
        for bundle_name in ("bundle_a", "bundle_b"):
            reaction = handle.create_group(f"{bundle_name}/RXN_0000000001")
            _add_species(reaction, "R0", smiles="O")
            _add_species(reaction, "P0", smiles="O")
            _add_species(reaction, "TS", smiles=TS_LABEL)

    # When / Then
    with pytest.raises(H5SchemaError) as excinfo:
        _ = list(iter_h5_reactions(path))
    message = str(excinfo.value)
    assert "RXN_0000000001" in message
    assert "bundle_a" in message
    assert "bundle_b" in message


def test_read_h5_species_and_reaction_standalone(tmp_path: Path) -> None:
    # Given: an open fixture file
    path = _build_one_reaction_file(tmp_path / "standalone.h5")
    with h5py.File(path, "r") as handle:
        # When
        record = read_h5_species(handle[SPECIES_PATH])
        rp_species, ts_species = read_h5_reaction(handle[REACTION_PATH])

    # Then
    assert isinstance(record, SpeciesRecord)
    assert record.tag == "R0"
    assert record.smiles == "O"
    assert record.charge == 0
    assert record.multiplicity == 1
    assert [species.tag for species in rp_species] == ["P0", "R0"]
    assert [species.tag for species in ts_species] == ["TS"]


def test_flattened_compound_dataset_species_layout(tmp_path: Path) -> None:
    # Given: species stored as flattened compound datasets instead of sub-groups
    path = tmp_path / "compound.h5"
    n_atoms = len(WATER_Z)
    dtype = np.dtype(
        [
            ("smiles", h5py.string_dtype()),
            ("EHG", "f8", (3,)),
            ("charge", "i8"),
            ("multiplicity", "i8"),
            ("atomic_numbers", "i8", (n_atoms,)),
            ("coordinates", "f8", (n_atoms, 3)),
        ]
    )
    with h5py.File(path, "w") as handle:
        reaction = handle.create_group(REACTION_PATH)
        for tag in ("R0", "P0", "TS"):
            data = np.array(
                (
                    tag.encode(),
                    DEFAULT_EHG,
                    0,
                    1,
                    WATER_Z,
                    WATER_X,
                ),
                dtype=dtype,
            )
            reaction.create_dataset(tag, data=data)

    # When
    results = list(iter_h5_reactions(path))

    # Then
    assert [reaction_id for reaction_id, _, _ in results] == ["RXN_0000000001"]
    _, rp_species, ts_species = results[0]
    assert [record.tag for record in rp_species] == ["P0", "R0"]
    assert [record.tag for record in ts_species] == ["TS"]
    r0 = next(record for record in rp_species if record.tag == "R0")
    assert r0.smiles == "R0"
    assert np.array_equal(r0.atomic_numbers, WATER_Z)
    assert r0.coordinates.shape == (n_atoms, 3)
    assert r0.charge == 0
    assert r0.multiplicity == 1
    assert np.allclose(r0.EHG, DEFAULT_EHG)
