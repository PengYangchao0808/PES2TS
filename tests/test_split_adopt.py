"""Offline tests for adopting the authors' official reaction-level split.

Fixtures are synthetic and built under ``tmp_path``: three split CSVs whose
names end in the configured suffixes, an interim inventory Parquet written
directly, and a minimal config that maps ``source.files`` at those fixtures.
The split stage never downloads or verifies sources, so no HDF5 or network
access is involved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.cli import main
from pes2ts_core.g0 import VOLATILE_KEYS
from pes2ts_core.g0.rejections import LEDGER_FILENAME, SUMMARY_FILENAME
from pes2ts_core.g0.split import (
    EXIT_SPLIT_SCHEMA_ERROR,
    SPLIT_ASSIGNMENT_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    SplitSchemaError,
    adopt_official_split,
)
from pes2ts_core.utils.hashing import md5_file
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

TRAIN_NAME = "fixture-RXN_train.csv"
VALID_NAME = "fixture-RXN_valid.csv"
TEST_NAME = "fixture-RXN_test.csv"
INVENTORY_NAME = "inventory.parquet"
SPLIT_NAMES = (TRAIN_NAME, VALID_NAME, TEST_NAME)


def _rid(number: int) -> str:
    """Return the canonical reaction ID for *number*."""
    return f"RXN_{number:010d}"


def _csv(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a CSV document with *header* and *rows*."""
    return "\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n"


def _split_csv(
    ids: Sequence[str], header: Sequence[str] = ("reaction_id",)
) -> str:
    """Render a split CSV carrying only reaction IDs."""
    return _csv(header, [(reaction_id,) for reaction_id in ids])


def _fixture_config(
    tmp_path: Path,
    *,
    inventory: Sequence[str],
    train: str,
    valid: str,
    test: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a minimal split config whose sources are synthetic CSV fixtures."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    files: dict[str, Any] = {}
    for name, content in (
        (TRAIN_NAME, train),
        (VALID_NAME, valid),
        (TEST_NAME, test),
    ):
        path = raw / name
        path.write_text(content, encoding="utf-8")
        files[name] = {
            "filename": name,
            "url": f"https://example.invalid/{name}",
            "md5": md5_file(path),
        }
    interim = tmp_path / "interim"
    write_parquet(interim / INVENTORY_NAME, {"reaction_id": list(inventory)})
    return {
        "source": {
            "zenodo_record": 18551029,
            "zenodo_revision": "1",
            "files": files,
        },
        "paths": {
            "raw": str(raw),
            "interim": str(interim),
            "manifests": str(tmp_path / "manifests"),
        },
        "split": {"seed": seed},
    }


def _ledger_records(root: Path) -> list[dict[str, Any]]:
    """Return the parsed unified-ledger entries under *root*."""
    path = root / "manifests" / LEDGER_FILENAME
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _without_volatile(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return *manifest* without the keys allowed to change between runs."""
    return {key: value for key, value in manifest.items() if key not in VOLATILE_KEYS}


def test_adopt_official_split_happy_path(tmp_path: Path) -> None:
    # Given: four inventory reactions fully covered by a 2/1/1 split
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(number) for number in (1, 2, 3, 4)],
        train=_csv(
            ("reaction_id", "reaction_smiles"),
            [(_rid(1), "O>>O"), (_rid(2), "C.O>>C.O")],
        ),
        valid=_split_csv([_rid(3)]),
        test=_split_csv([_rid(4)]),
    )

    # When
    result = adopt_official_split(config)

    # Then: the result and files reflect the fixture, nothing else
    assert result.covered == 4
    assert result.counts == {"train": 2, "valid": 1, "test": 1, "total": 4}
    assert result.n_not_in_split == 0
    assert result.assignment_path == tmp_path / "interim" / SPLIT_ASSIGNMENT_FILENAME
    assert result.manifest_path == tmp_path / "manifests" / SPLIT_MANIFEST_FILENAME

    table = read_parquet(result.assignment_path)
    assert table.column_names == ["reaction_id", "split"]
    assert table.num_rows == 4
    assert table.to_pylist() == [
        {"reaction_id": _rid(1), "split": "train"},
        {"reaction_id": _rid(2), "split": "train"},
        {"reaction_id": _rid(3), "split": "valid"},
        {"reaction_id": _rid(4), "split": "test"},
    ]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "g0_manifest_v1"
    assert manifest["dataset_version"] == "zenodo-18551029-rev1"
    assert manifest["strategy"] == "authors_official"
    assert manifest["seed"] == 42
    assert manifest["counts"] == {"train": 2, "valid": 1, "test": 1, "total": 4}
    assert manifest["covered"] == 4
    assert manifest["n_inventory"] == 4
    assert manifest["n_not_in_split"] == 0
    assert manifest["n_split_only"] == 0
    assert manifest["split_only_examples"] == []
    assert manifest["leak_status"] == "pending"
    assert "generated_at" in manifest

    # And: every source CSV carries a 64-hex SHA-256
    assert set(manifest["source_sha256"]) == set(SPLIT_NAMES)
    for digest in manifest["source_sha256"].values():
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_id_in_two_splits_raises_naming_id_and_both_files(tmp_path: Path) -> None:
    # Given: the same reaction assigned to train and valid
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([_rid(1)]),
        test=_split_csv([]),
    )

    # When / Then
    with pytest.raises(SplitSchemaError) as excinfo:
        adopt_official_split(config)

    message = str(excinfo.value)
    assert _rid(1) in message
    assert TRAIN_NAME in message
    assert VALID_NAME in message
    # And: no artifact was produced
    assert not (tmp_path / "interim" / SPLIT_ASSIGNMENT_FILENAME).exists()
    assert not (tmp_path / "manifests" / SPLIT_MANIFEST_FILENAME).exists()


def test_missing_id_column_raises_listing_actual_columns(tmp_path: Path) -> None:
    # Given: a split CSV whose header has no reaction-id column
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_csv(("smiles", "dE_dagger"), [("O>>O", "-1.0")]),
        valid=_split_csv([_rid(1)]),
        test=_split_csv([]),
    )

    # When / Then
    with pytest.raises(SplitSchemaError) as excinfo:
        adopt_official_split(config)

    message = str(excinfo.value)
    assert "has no reaction-id column" in message
    assert "smiles" in message
    assert "dE_dagger" in message


def test_inventory_id_absent_from_split_becomes_rejection(tmp_path: Path) -> None:
    # Given: an inventory of three reactions covered only by train+valid
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1), _rid(2), _rid(3)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([_rid(2)]),
        test=_split_csv([]),
    )

    # When
    result = adopt_official_split(config)

    # Then: the uncovered reaction is counted, excluded, and rejected
    assert result.covered == 2
    assert result.n_not_in_split == 1
    assert result.counts == {"train": 1, "valid": 1, "test": 0, "total": 2}
    assert read_parquet(result.assignment_path).num_rows == 2

    records = _ledger_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["reaction_id"] == _rid(3)
    assert record["code"] == "NOT_IN_SPLIT"
    assert record["stage"] == "adopt_official_split"
    assert _rid(3) in record["detail"]
    assert record["source_pointer"] == str(tmp_path / "interim" / INVENTORY_NAME)


def test_bad_id_in_split_csv_raises_with_file_line_and_value(tmp_path: Path) -> None:
    # Given: a non-numeric ID inside the train CSV
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_csv(("reaction_id",), [(_rid(1),), ("abc",), (_rid(2),)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )

    # When / Then
    with pytest.raises(SplitSchemaError) as excinfo:
        adopt_official_split(config)

    message = str(excinfo.value)
    assert "abc" in message
    assert TRAIN_NAME in message
    assert "line 3" in message


def test_duplicate_id_within_one_file_raises(tmp_path: Path) -> None:
    # Given: train assigns the same reaction twice
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_split_csv([_rid(1), _rid(1)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )

    # When / Then
    with pytest.raises(SplitSchemaError) as excinfo:
        adopt_official_split(config)

    message = str(excinfo.value)
    assert _rid(1) in message
    assert "more than once" in message
    assert TRAIN_NAME in message


def test_id_column_candidates_fall_back_in_priority_order(tmp_path: Path) -> None:
    # Given: rxn_id (train) and id (valid) headers instead of reaction_id
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1), _rid(2)],
        train=_csv(("rxn_id",), [(_rid(1),)]),
        valid=_csv(("id",), [(_rid(2),)]),
        test=_csv(("reaction_id",), []),
    )

    # When
    result = adopt_official_split(config)

    # Then
    assert result.covered == 2
    assert result.counts == {"train": 1, "valid": 1, "test": 0, "total": 2}


def test_eighty_ten_ten_fixture_sanity(tmp_path: Path) -> None:
    # Given: an 8/1/1 fixture over ten inventory reactions
    train_ids = [_rid(number) for number in range(1, 9)]
    valid_ids = [_rid(9)]
    test_ids = [_rid(10)]
    config = _fixture_config(
        tmp_path,
        inventory=[*train_ids, *valid_ids, *test_ids],
        train=_split_csv(train_ids),
        valid=_split_csv(valid_ids),
        test=_split_csv(test_ids),
        seed=7,
    )

    # When
    result = adopt_official_split(config)

    # Then: counts are exactly the fixture's, with full coverage
    assert result.counts == {"train": 8, "valid": 1, "test": 1, "total": 10}
    assert result.covered == 10
    assert result.n_not_in_split == 0

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["counts"]["total"] == manifest["n_inventory"] == 10


def test_split_only_ids_are_counted_and_examples_capped(tmp_path: Path) -> None:
    # Given: 25 split IDs that are not in the one-row inventory
    inventory = [_rid(1)]
    extras = [_rid(number) for number in range(2, 27)]
    config = _fixture_config(
        tmp_path,
        inventory=inventory,
        train=_split_csv([_rid(1), *extras]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )

    # When
    result = adopt_official_split(config)

    # Then: only the inventory-backed row is assigned
    assert result.covered == 1
    assert read_parquet(result.assignment_path).num_rows == 1

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_split_only"] == 25
    assert manifest["split_only_examples"] == sorted(extras)[:20]
    assert len(manifest["split_only_examples"]) == 20


def test_existing_ledger_entries_are_preserved_and_runs_are_idempotent(
    tmp_path: Path,
) -> None:
    # Given: a ledger line written by an earlier stage
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1), _rid(2)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    preexisting = {
        "reaction_id": _rid(99),
        "stage": "build_inventory",
        "code": "BAD_SMILES",
        "detail": "pre-existing entry",
        "source_pointer": "previous.jsonl:1",
    }
    (manifests_dir / LEDGER_FILENAME).write_text(
        json.dumps(preexisting) + "\n", encoding="utf-8"
    )

    # When
    adopt_official_split(config)

    # Then: the earlier entry survives and the new rejection follows
    records = _ledger_records(tmp_path)
    assert len(records) == 2
    assert records[0] == preexisting
    assert records[1]["code"] == "NOT_IN_SPLIT"
    assert records[1]["reaction_id"] == _rid(2)

    summary = json.loads((manifests_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["by_code"] == {"BAD_SMILES": 1, "NOT_IN_SPLIT": 1}

    # And: a second run does not duplicate any ledger entry
    adopt_official_split(config)
    assert _ledger_records(tmp_path) == records


def test_split_files_are_never_mutated(tmp_path: Path) -> None:
    # Given: the three split CSVs
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )
    raw_paths = [tmp_path / "raw" / name for name in SPLIT_NAMES]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in raw_paths]

    # When
    adopt_official_split(config)

    # Then
    after = [(path.read_bytes(), path.stat().st_mtime_ns) for path in raw_paths]
    assert after == before


def test_manifest_is_deterministic_modulo_volatile_keys(tmp_path: Path) -> None:
    # Given
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1), _rid(2), _rid(3)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([_rid(2)]),
        test=_split_csv([]),
    )

    # When: the same config runs twice
    first = adopt_official_split(config)
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    first_assignment = first.assignment_path.read_bytes()
    first_ledger = (tmp_path / "manifests" / LEDGER_FILENAME).read_text(encoding="utf-8")

    second = adopt_official_split(config)
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    second_assignment = second.assignment_path.read_bytes()
    second_ledger = (tmp_path / "manifests" / LEDGER_FILENAME).read_text(encoding="utf-8")

    # Then: everything but the volatile keys is byte/structurally identical
    assert _without_volatile(first_manifest) == _without_volatile(second_manifest)
    assert first_assignment == second_assignment
    assert first_ledger == second_ledger


def test_missing_inventory_raises_with_g0_inventory_hint(tmp_path: Path) -> None:
    # Given: a fixture whose inventory parquet was removed
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )
    (tmp_path / "interim" / INVENTORY_NAME).unlink()

    # When / Then
    with pytest.raises(FileNotFoundError, match="g0 inventory"):
        adopt_official_split(config)


def test_missing_split_file_raises_with_g0_fetch_hint(tmp_path: Path) -> None:
    # Given: a fixture whose valid split CSV was removed from the raw tree
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )
    (tmp_path / "raw" / VALID_NAME).unlink()

    # When / Then
    with pytest.raises(FileNotFoundError, match="g0 fetch"):
        adopt_official_split(config)


def test_ambiguous_split_suffix_raises_listing_names(tmp_path: Path) -> None:
    # Given: a second configured CSV ending in the train suffix
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )
    extra = tmp_path / "raw" / "second_train.csv"
    extra.write_text(_split_csv([_rid(1)]), encoding="utf-8")
    config["source"]["files"]["second_train.csv"] = {"filename": "second_train.csv"}

    # When / Then
    with pytest.raises(ValueError) as excinfo:
        adopt_official_split(config)

    message = str(excinfo.value)
    assert TRAIN_NAME in message
    assert "second_train.csv" in message


def test_cli_split_writes_artifacts_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the CLI whose config loader returns the fixture config
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1), _rid(2)],
        train=_split_csv([_rid(1)]),
        valid=_split_csv([_rid(2)]),
        test=_split_csv([]),
    )
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "split"])

    # Then
    assert exit_code == 0
    assert (tmp_path / "interim" / SPLIT_ASSIGNMENT_FILENAME).is_file()
    assert (tmp_path / "manifests" / SPLIT_MANIFEST_FILENAME).is_file()


def test_cli_split_schema_error_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a split CSV without a reaction-id column
    config = _fixture_config(
        tmp_path,
        inventory=[_rid(1)],
        train=_csv(("smiles",), [("O>>O",)]),
        valid=_split_csv([]),
        test=_split_csv([]),
    )
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "split"])

    # Then: the documented split-schema exit code is returned
    assert exit_code == EXIT_SPLIT_SCHEMA_ERROR
    assert not (tmp_path / "interim" / SPLIT_ASSIGNMENT_FILENAME).exists()
