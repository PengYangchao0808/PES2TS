"""Tests for full-dataset exact-duplicate and reverse-reaction detection.

A tiny pyarrow inventory locks the dedup contract: one direction-agnostic
identity group can hold a same-direction exact duplicate and a written reverse,
the canonical member is the lowest reaction ID, the inventory file is never
mutated, both JSON artifacts are byte-identical across runs, and the rejection
ledger is appended to instead of clobbered. A ``realdata`` test measures the
full 199,890-row pass against a documented linear-time budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.dedup import (
    DUPLICATE_GROUPS_FILENAME,
    DUPLICATE_LEDGER_FILENAME,
    detect_duplicates,
)
from pes2ts_core.g0.rejections import LEDGER_FILENAME, SUMMARY_FILENAME
from pes2ts_core.utils.hashing import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_INVENTORY_PATH = PROJECT_ROOT / "data" / "interim" / "inventory.parquet"

CANONICAL_ID = "RXN_0000000001"
MAPPED_DUPLICATE_ID = "RXN_0000000002"
REVERSE_ID = "RXN_0000000003"
DISTINCT_ID = "RXN_0000000004"

CANONICAL_SMILES = "CCO>>CC=O"
MAPPED_DUPLICATE_SMILES = "[CH3:9][CH2:1][OH:7]>>[CH3:2][CH:3]=[O:4]"
REVERSE_SMILES = "CC=O>>CCO"
DISTINCT_SMILES = "c1ccccc1>>c1ccc(O)cc1"

#: Canonical member listed last so selection is by ID, not file position.
DUPLICATE_FIXTURE: tuple[tuple[str, str], ...] = (
    (MAPPED_DUPLICATE_ID, MAPPED_DUPLICATE_SMILES),
    (REVERSE_ID, REVERSE_SMILES),
    (CANONICAL_ID, CANONICAL_SMILES),
    (DISTINCT_ID, DISTINCT_SMILES),
)

PREEXISTING_REJECTION: dict[str, str] = {
    "reaction_id": "RXN_0000000009",
    "stage": "build_inventory",
    "code": "MISSING_IN_H5",
    "detail": "pre-existing rejection",
    "source_pointer": "inventory.csv:RXN_0000000009",
}


def _write_inventory(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    """Write a two-column inventory plus one ignored column."""
    table = pa.table(
        {
            "reaction_id": [row[0] for row in rows],
            "reaction_smiles": [row[1] for row in rows],
            "note": ["ignored" for _ in rows],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "source": {"zenodo_record": 1, "zenodo_revision": "1"},
        "paths": {
            "interim": str(tmp_path / "interim"),
            "manifests": str(tmp_path / "manifests"),
        },
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_preexisting_rejection(manifests_dir: Path) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    document = json.dumps(PREEXISTING_REJECTION)
    (manifests_dir / LEDGER_FILENAME).write_text(document + "\n", encoding="utf-8")


def test_group_contains_exact_duplicate_and_written_reverse(tmp_path: Path) -> None:
    # Given: canonical + map-permuted duplicate + written reverse + distinct row
    inventory = tmp_path / "interim" / "inventory.parquet"
    _write_inventory(inventory, DUPLICATE_FIXTURE)
    digest_before = sha256_file(inventory)
    config = _config(tmp_path)

    # When: the full-dataset duplicate pass runs
    result = detect_duplicates(inventory, config)

    # Then: counts see one 3-member group, one duplicate, and one reverse
    assert result.n_rows == 4
    assert result.n_unique == 2
    assert result.n_duplicate_groups == 1
    assert result.n_exact_duplicates == 1
    assert result.n_reverse_pairs == 1
    assert result.n_rows == (
        result.n_unique + result.n_exact_duplicates + result.n_reverse_pairs
    )
    assert result.groups_path == (
        Path(config["paths"]["interim"]) / DUPLICATE_GROUPS_FILENAME
    )
    assert result.ledger_path == (
        Path(config["paths"]["manifests"]) / DUPLICATE_LEDGER_FILENAME
    )

    groups_document = _read_json(result.groups_path)
    assert groups_document["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert groups_document["dataset_version"] == "zenodo-1-rev1"
    assert len(groups_document["groups"]) == 1
    group = groups_document["groups"][0]
    assert group["member_reaction_ids"] == [
        CANONICAL_ID,
        MAPPED_DUPLICATE_ID,
        REVERSE_ID,
    ]
    assert group["canonical_reaction_id"] == CANONICAL_ID

    ledger_document = _read_json(result.ledger_path)
    assert ledger_document["n_inventory_rows"] == 4
    assert ledger_document["n_unique_identities"] == 2
    assert ledger_document["n_duplicate_groups"] == 1
    assert ledger_document["n_exact_duplicates"] == 1
    assert ledger_document["n_reverse_pairs"] == 1
    assert ledger_document["groups"] == groups_document["groups"]
    assert ledger_document["pairs"] == [
        {
            "kind": "duplicate",
            "canonical_reaction_id": CANONICAL_ID,
            "other_reaction_id": MAPPED_DUPLICATE_ID,
        },
        {
            "kind": "reverse",
            "canonical_reaction_id": CANONICAL_ID,
            "other_reaction_id": REVERSE_ID,
        },
    ]

    rejection_lines = (result.ledger_path.parent / LEDGER_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    records = [json.loads(line) for line in rejection_lines]
    assert [record["code"] for record in records] == ["DUPLICATE_OF", "REVERSE_OF"]
    assert [record["reaction_id"] for record in records] == [
        MAPPED_DUPLICATE_ID,
        REVERSE_ID,
    ]
    assert all(record["stage"] == "detect_duplicates" for record in records)
    assert CANONICAL_ID in records[0]["detail"]
    assert CANONICAL_ID in records[1]["detail"]
    summary = _read_json(result.ledger_path.parent / SUMMARY_FILENAME)
    assert summary["total"] == 2
    assert summary["by_code"] == {"DUPLICATE_OF": 1, "REVERSE_OF": 1}

    # And: detection is read-only -- the inventory bytes are unchanged
    assert sha256_file(inventory) == digest_before


def test_distinct_inventory_writes_empty_artifacts_and_preserves_ledger(
    tmp_path: Path,
) -> None:
    # Given: two reactions with distinct identities and a pre-existing rejection
    inventory = tmp_path / "interim" / "inventory.parquet"
    _write_inventory(
        inventory, ((CANONICAL_ID, CANONICAL_SMILES), (DISTINCT_ID, DISTINCT_SMILES))
    )
    config = _config(tmp_path)
    manifests_dir = Path(config["paths"]["manifests"])
    _write_preexisting_rejection(manifests_dir)

    # When
    result = detect_duplicates(inventory, config)

    # Then: no groups and no pairs anywhere, and the earlier rejection survives
    assert result.n_unique == 2
    assert result.n_duplicate_groups == 0
    assert result.n_exact_duplicates == 0
    assert result.n_reverse_pairs == 0
    assert _read_json(result.groups_path)["groups"] == []
    ledger_document = _read_json(result.ledger_path)
    assert ledger_document["groups"] == []
    assert ledger_document["pairs"] == []
    lines = (manifests_dir / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["code"] == "MISSING_IN_H5"
    summary = _read_json(manifests_dir / SUMMARY_FILENAME)
    assert summary["total"] == 1
    assert summary["by_code"] == {"MISSING_IN_H5": 1}


def test_two_runs_produce_byte_identical_artifacts(tmp_path: Path) -> None:
    # Given: the duplicate fixture and one config
    inventory = tmp_path / "interim" / "inventory.parquet"
    _write_inventory(inventory, DUPLICATE_FIXTURE)
    config = _config(tmp_path)

    # When: the pass runs twice over the unchanged inventory
    first = detect_duplicates(inventory, config)
    first_groups = first.groups_path.read_bytes()
    first_ledger = first.ledger_path.read_bytes()
    second = detect_duplicates(inventory, config)

    # Then: both artifacts are byte-identical because neither embeds a timestamp
    assert second.groups_path.read_bytes() == first_groups
    assert second.ledger_path.read_bytes() == first_ledger


def test_rejections_append_to_the_existing_ledger(tmp_path: Path) -> None:
    # Given: a manually written ledger line and the duplicate fixture
    inventory = tmp_path / "interim" / "inventory.parquet"
    _write_inventory(inventory, DUPLICATE_FIXTURE)
    config = _config(tmp_path)
    manifests_dir = Path(config["paths"]["manifests"])
    _write_preexisting_rejection(manifests_dir)

    # When: dedup records its two rejections
    detect_duplicates(inventory, config)

    # Then: the pre-existing line is preserved and both new lines are appended
    lines = (manifests_dir / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["code"] for line in lines] == [
        "MISSING_IN_H5",
        "DUPLICATE_OF",
        "REVERSE_OF",
    ]


@pytest.mark.realdata
def test_real_inventory_full_pass_within_budget(tmp_path: Path) -> None:
    # Given: the real inventory, when it has been built
    if not REAL_INVENTORY_PATH.is_file():
        pytest.skip(f"real inventory not present: {REAL_INVENTORY_PATH}")
    n_rows = pq.read_table(REAL_INVENTORY_PATH).num_rows
    config = _config(tmp_path)

    # When: the full-dataset pass runs (outputs stay in tmp_path)
    start = time.perf_counter()
    result = detect_duplicates(REAL_INVENTORY_PATH, config)
    elapsed = time.perf_counter() - start

    # Then: every row is accounted for inside the documented 10-minute budget
    assert result.n_rows == n_rows
    assert result.n_rows == (
        result.n_unique + result.n_exact_duplicates + result.n_reverse_pairs
    )
    budget_seconds = 600.0
    assert elapsed < budget_seconds, (
        f"dedup of {n_rows} rows took {elapsed:.1f}s; budget "
        f"{budget_seconds:.0f}s (one O(N) identity pass at ~15k rows/s plus "
        "tiny intra-group comparisons)"
    )
