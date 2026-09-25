"""Unit tests for the reaction-info CSV reader."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from pes2ts_core.g0.reader import (
    ENERGY_COLUMNS,
    READ_STAGE,
    CsvSchemaError,
    read_reaction_info_csv,
)
from pes2ts_core.g0.rejections import Rejection, RejectionCode

HEADER: list[str] = ["reaction_id", "reaction_smiles", *ENERGY_COLUMNS]
GOOD_SMILES = "CCO.CN>>CC(=O)N"
GOOD_ENERGIES = ["-1.5", "10.25", "-2.0", "11.5", "-3.0", "12.5"]


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    """Write *rows* with *header* to *path* and return the path."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_good_duplicate_and_no_arrow_rows(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "reaction_info.csv",
        HEADER,
        [
            ["1", GOOD_SMILES, *GOOD_ENERGIES],
            ["0000000001", GOOD_SMILES, *GOOD_ENERGIES],
            ["2", "CCO.CN", "-1.0", "9.0", "-1.5", "10.0", "-2.0", "11.0"],
        ],
    )

    records, rejections = read_reaction_info_csv(csv_path)

    assert len(records) == 1
    record = records[0]
    assert set(record) == {"reaction_id", "reaction_smiles", *ENERGY_COLUMNS}
    assert record["reaction_id"] == "RXN_0000000001"
    assert record["reaction_smiles"] == GOOD_SMILES
    assert record["dE"] == -1.5
    assert record["dE_dagger"] == 10.25
    assert record["dH"] == -2.0
    assert record["dH_dagger"] == 11.5
    assert record["dG"] == -3.0
    assert record["dG_dagger"] == 12.5

    assert [rejection.code for rejection in rejections] == [
        RejectionCode.DUPLICATE_ID,
        RejectionCode.NO_ARROW,
    ]
    for rejection in rejections:
        assert isinstance(rejection, Rejection)
        assert rejection.stage == READ_STAGE
    assert rejections[0].reaction_id == "RXN_0000000001"
    assert rejections[0].source_pointer == f"{csv_path}:3"
    assert f"{csv_path}:2" in rejections[0].detail
    assert rejections[1].reaction_id == "RXN_0000000002"
    assert rejections[1].source_pointer == f"{csv_path}:4"


def test_missing_reaction_id_column_lists_actual_columns(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "no_id.csv", ["id", "smiles"], [["1", "CCO>>CC=O"]]
    )
    with pytest.raises(CsvSchemaError) as excinfo:
        read_reaction_info_csv(csv_path)
    assert str(excinfo.value) == "Missing reaction_id column; found: ['id', 'smiles']"


def test_missing_smiles_column_lists_actual_columns(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "no_smiles.csv", ["reaction_id", "rxn"], [["1", GOOD_SMILES]]
    )
    with pytest.raises(CsvSchemaError) as excinfo:
        read_reaction_info_csv(csv_path)
    message = str(excinfo.value)
    assert "reaction SMILES" in message
    assert "['reaction_id', 'rxn']" in message


def test_zero_byte_file_raises_schema_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(CsvSchemaError) as excinfo:
        read_reaction_info_csv(csv_path)
    assert str(excinfo.value) == "Missing reaction_id column; found: []"


def test_header_only_file_yields_no_records_and_no_rejections(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path / "header_only.csv", HEADER, [])
    records, rejections = read_reaction_info_csv(csv_path)
    assert records == []
    assert rejections == []


def test_unparseable_component_is_rejected_naming_the_component(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "broken_component.csv",
        HEADER,
        [["7", "CCO>>C1CC2CCC3", *GOOD_ENERGIES]],
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert records == []
    assert [rejection.code for rejection in rejections] == [
        RejectionCode.UNPARSEABLE_MAPPED
    ]
    assert rejections[0].reaction_id == "RXN_0000000007"
    assert "C1CC2CCC3" in rejections[0].detail
    assert rejections[0].source_pointer == f"{csv_path}:2"


@pytest.mark.parametrize(
    "smiles",
    ["CCO>>", ">>CC=O", "CCO.>>CC=O", "CCO>>.CC=O"],
)
def test_empty_side_or_component_is_bad_smiles(tmp_path: Path, smiles: str) -> None:
    csv_path = write_csv(
        tmp_path / "empty_side.csv", HEADER, [["3", smiles, *GOOD_ENERGIES]]
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert records == []
    assert [rejection.code for rejection in rejections] == [RejectionCode.BAD_SMILES]


def test_multiple_arrows_is_no_arrow(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "two_arrows.csv",
        HEADER,
        [["4", "CCO>>CC=O>>", *GOOD_ENERGIES]],
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert records == []
    assert [rejection.code for rejection in rejections] == [RejectionCode.NO_ARROW]
    assert "found 2" in rejections[0].detail


def test_bad_id_and_rejected_rows_do_not_claim_the_id(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "bad_id.csv",
        HEADER,
        [
            ["not-a-number", GOOD_SMILES, *GOOD_ENERGIES],
            ["1", "CCO.CN", *GOOD_ENERGIES],
            ["0000000001", GOOD_SMILES, *GOOD_ENERGIES],
        ],
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert [rejection.code for rejection in rejections] == [
        RejectionCode.BAD_ID,
        RejectionCode.NO_ARROW,
    ]
    assert rejections[0].reaction_id == "not-a-number"
    assert rejections[1].reaction_id == "RXN_0000000001"
    assert len(records) == 1
    assert records[0]["reaction_id"] == "RXN_0000000001"


def test_missing_energy_columns_produce_none(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "no_energy.csv",
        ["reaction_id", "reaction_smiles"],
        [["5", GOOD_SMILES]],
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert rejections == []
    assert len(records) == 1
    for column in ENERGY_COLUMNS:
        assert records[0][column] is None


def test_blank_and_non_numeric_energy_cells_become_none(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "weird_energy.csv",
        HEADER,
        [["6", GOOD_SMILES, "", "n/a", "1.0", "2", "3", "4"]],
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert rejections == []
    record = records[0]
    assert record["dE"] is None
    assert record["dE_dagger"] is None
    assert record["dH"] == 1.0
    assert record["dH_dagger"] == 2.0
    assert record["dG"] == 3.0
    assert record["dG_dagger"] == 4.0


@pytest.mark.parametrize("column", ["reaction_smiles", "SMILES", "smiles"])
def test_accepted_smiles_column_names(tmp_path: Path, column: str) -> None:
    csv_path = write_csv(
        tmp_path / "smiles_column.csv",
        ["reaction_id", column],
        [["1", GOOD_SMILES]],
    )
    records, rejections = read_reaction_info_csv(csv_path)

    assert rejections == []
    assert records[0]["reaction_smiles"] == GOOD_SMILES


def test_smiles_column_priority_is_first_present(tmp_path: Path) -> None:
    # Lower-priority "smiles" holds garbage; the reader must use "SMILES".
    csv_path = write_csv(
        tmp_path / "priority.csv",
        ["reaction_id", "smiles", "SMILES"],
        [["1", "not a reaction smiles", GOOD_SMILES]],
    )
    records, rejections = read_reaction_info_csv(csv_path)
    assert rejections == []
    assert records[0]["reaction_smiles"] == GOOD_SMILES

    # "reaction_smiles" outranks "SMILES" when both are present.
    csv_path_2 = write_csv(
        tmp_path / "priority_2.csv",
        ["reaction_id", "SMILES", "reaction_smiles"],
        [["1", GOOD_SMILES, "CCO>>CC=O"]],
    )
    records_2, _ = read_reaction_info_csv(csv_path_2)
    assert records_2[0]["reaction_smiles"] == "CCO>>CC=O"


def test_string_path_is_accepted(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "string_path.csv", HEADER, [["1", GOOD_SMILES, *GOOD_ENERGIES]]
    )
    records, rejections = read_reaction_info_csv(str(csv_path))

    assert rejections == []
    assert records[0]["reaction_id"] == "RXN_0000000001"
