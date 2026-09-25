"""Offline tests for the mandatory DRFP cross-split near-duplicate audit.

Fixtures are synthetic and built under ``tmp_path``: a two-column inventory
Parquet, a split assignment Parquet, and a minimal config. The DRFP analog pair
is validated with an independent :class:`drfp.DrfpEncoder` computation before
the audit runs, so the test asserts real fingerprint similarity instead of
trusting a fixture comment. No HDF5, network, or real-data access is involved;
the single ``realdata`` test skips unless the real inventory and split exist.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from drfp import DrfpEncoder
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from pes2ts_core.cli import main
from pes2ts_core.g0.dedup import DUPLICATE_LEDGER_FILENAME
from pes2ts_core.g0.fingerprints import (
    FINGERPRINTS_PARQUET_FILENAME,
    load_fingerprints,
)
from pes2ts_core.g0.neardup import (
    AUDIT_STAGE,
    BUDGET_EXCEEDED_REASON,
    EXIT_AUDIT_BUDGET_EXCEEDED,
    LEAK_AUDIT_FILENAME,
    AuditBudgetExceeded,
    compute_fingerprints,
    cross_split_leak_audit,
)
from pes2ts_core.g0.rejections import LEDGER_FILENAME
from pes2ts_core.g0.split import SPLIT_ASSIGNMENT_FILENAME
from pes2ts_core.utils.parquet_io import read_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_INVENTORY_PATH = PROJECT_ROOT / "data" / "interim" / "inventory.parquet"
REAL_SPLIT_PATH = PROJECT_ROOT / "data" / "interim" / "split_assignment.parquet"
INVENTORY_NAME = "inventory.parquet"
FP_SIZE = 2048
THRESHOLD = 0.90


def _rid(number: int) -> str:
    """Return the canonical reaction ID for *number*."""
    return f"RXN_{number:010d}"


#: Same esterification chemistry at two chain lengths: distinct identities,
#: identical DRFP difference substructures (verified empirically below).
TRAIN_ANALOG = "CCCCCCCC(=O)O.CO>>CCCCCCCC(=O)OC.O"
TEST_ANALOG = "CCCCCCCCC(=O)O.CO>>CCCCCCCCC(=O)OC.O"
TRAIN_AROMATIC = "c1ccc(cc1)C(=O)O.CCO>>c1ccc(cc1)C(=O)OCC.O"
TRAIN_DIELS = "C=CC=C.C=C>>C1CC=CCC1"
TEST_UNRELATED = "c1ccccc1.Br>>Brc1ccccc1"
VALID_AMIDE = "CC(=O)O.CCN>>CC(=O)NCC.O"
TRAIN_IDS = (_rid(1), _rid(2), _rid(3))
PROBE_IDS = (_rid(4), _rid(5), _rid(6))
HAPPY_ROWS: tuple[tuple[str, str], ...] = (
    (_rid(1), TRAIN_ANALOG),
    (_rid(2), TRAIN_AROMATIC),
    (_rid(3), TRAIN_DIELS),
    (_rid(4), TEST_ANALOG),
    (_rid(5), TEST_UNRELATED),
    (_rid(6), VALID_AMIDE),
)
HAPPY_ASSIGNMENT: tuple[tuple[str, str], ...] = (
    (_rid(1), "train"),
    (_rid(2), "train"),
    (_rid(3), "train"),
    (_rid(4), "test"),
    (_rid(5), "test"),
    (_rid(6), "valid"),
)


def _write_inventory(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    """Write a two-column inventory Parquet with an ignored extra column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "reaction_id": [row[0] for row in rows],
            "reaction_smiles": [row[1] for row in rows],
            "note": ["ignored" for _ in rows],
        }
    )
    pq.write_table(table, path)


def _write_assignment(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    """Write a two-column split assignment Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "reaction_id": [row[0] for row in rows],
                "split": [row[1] for row in rows],
            }
        ),
        path,
    )


def _config(
    tmp_path: Path,
    *,
    threshold: float = THRESHOLD,
    fp_size: int = FP_SIZE,
    block_size: int = 1000,
    max_seconds: float = 7200.0,
    max_pairs: int = 10000,
) -> dict[str, Any]:
    """Build a minimal near-dup config rooted at *tmp_path*."""
    return {
        "source": {"zenodo_record": 1, "zenodo_revision": "1"},
        "paths": {
            "interim": str(tmp_path / "interim"),
            "manifests": str(tmp_path / "manifests"),
        },
        "near_dup": {
            "threshold": threshold,
            "fp_size": fp_size,
            "block_size": block_size,
            "max_seconds": max_seconds,
            "max_pairs": max_pairs,
        },
    }


def _happy_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Write the standard six-reaction fixture and return its config."""
    config = _config(tmp_path, **overrides)
    interim = Path(config["paths"]["interim"])
    _write_inventory(interim / INVENTORY_NAME, HAPPY_ROWS)
    _write_assignment(interim / SPLIT_ASSIGNMENT_FILENAME, HAPPY_ASSIGNMENT)
    return config


def _drfp_bits(reaction_smiles: str, fp_size: int = FP_SIZE) -> str:
    """Return the 0/1 string of the fingerprint of one reaction SMILES."""
    (array,) = DrfpEncoder.encode(
        [reaction_smiles], n_folded_length=fp_size, radius=3, rings=True
    )
    return "".join(str(int(value)) for value in array.tolist())


def _drfp_vectors(
    rows: tuple[tuple[str, str], ...], fp_size: int = FP_SIZE
) -> dict[str, ExplicitBitVect]:
    """Return ``reaction_id -> ExplicitBitVect`` computed independently."""
    vectors: dict[str, ExplicitBitVect] = {}
    for reaction_id, reaction_smiles in rows:
        vectors[reaction_id] = DataStructs.CreateFromBitString(
            _drfp_bits(reaction_smiles, fp_size)
        )
    return vectors


def _similarity(
    vectors: dict[str, ExplicitBitVect], probe_id: str, reference_id: str
) -> float:
    """Return the Tanimoto similarity of two independently built vectors."""
    return DataStructs.BulkTanimotoSimilarity(
        vectors[probe_id], [vectors[reference_id]]
    )[0]


def _manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Return the parsed leak-audit manifest of *config*."""
    path = Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the parsed unified-ledger entries of *config*."""
    path = Path(config["paths"]["manifests"]) / LEDGER_FILENAME
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_happy_path_flags_analog_and_ignores_unrelated(tmp_path: Path) -> None:
    # Given: a fixture whose analog pair is validated with an independent DRFP
    # computation -- an octanoate/nonanoate ester pair must exceed the threshold
    config = _happy_config(tmp_path)
    vectors = _drfp_vectors(HAPPY_ROWS)
    analog_similarity = _similarity(vectors, _rid(4), _rid(1))
    assert analog_similarity >= THRESHOLD, "fixture analog pair is not a real near-duplicate"
    for probe_id in PROBE_IDS:
        for train_id in TRAIN_IDS:
            if (probe_id, train_id) == (_rid(4), _rid(1)):
                continue
            assert _similarity(vectors, probe_id, train_id) < THRESHOLD
    expected_max = max(
        _similarity(vectors, probe_id, train_id)
        for probe_id in PROBE_IDS
        for train_id in TRAIN_IDS
    )
    expected_hits = [
        (probe_id, train_id)
        for probe_id in PROBE_IDS
        for train_id in TRAIN_IDS
        if _similarity(vectors, probe_id, train_id) >= THRESHOLD
    ]

    # When: fingerprints are computed and the full audit runs
    fingerprint_result = compute_fingerprints(config)
    result = cross_split_leak_audit(config, fingerprint_result.fingerprints)

    # Then: exactly the analog pair is flagged and the unrelated probe is not
    assert fingerprint_result.n == len(HAPPY_ROWS)
    assert fingerprint_result.fp_size == FP_SIZE
    assert result.complete is True
    assert result.manifest_path == Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME
    assert result.n_pairs_over_threshold == len(expected_hits) == 1
    assert result.max_similarity == pytest.approx(expected_max, abs=1e-12)
    assert expected_max == pytest.approx(analog_similarity, abs=1e-12)

    manifest = _manifest(config)
    assert manifest["schema_version"] == "g0_manifest_v1"
    assert manifest["dataset_version"] == "zenodo-1-rev1"
    assert manifest["method"] == "DRFP"
    assert manifest["params"] == {
        "fp_size": FP_SIZE,
        "radius": 3,
        "rings": True,
        "threshold": THRESHOLD,
    }
    assert manifest["complete"] is True
    assert manifest["n_probes"] == 3
    assert manifest["n_references"] == 3
    assert manifest["n_comparisons"] == 9
    assert manifest["n_pairs_over_threshold"] == 1
    assert manifest["max_similarity"] == pytest.approx(expected_max, abs=1e-12)
    assert manifest["n_cross_split_known_duplicates"] == 0
    assert len(manifest["top_offenses"]) == 1
    offense = manifest["top_offenses"][0]
    assert offense["probe_reaction_id"] == _rid(4)
    assert offense["reference_reaction_id"] == _rid(1)
    assert offense["similarity"] == pytest.approx(analog_similarity, abs=1e-12)
    assert _rid(5) not in json.dumps(manifest["top_offenses"])
    assert "duration_seconds" in manifest
    assert "generated_at" in manifest
    # And: this module never writes a leakage verdict
    assert "leak_status" not in manifest


def test_fingerprints_parquet_round_trip(tmp_path: Path) -> None:
    # Given: the standard fixture
    config = _happy_config(tmp_path)

    # When: fingerprints are computed and persisted
    result = compute_fingerprints(config)

    # Then: the table holds the canonical 0/1 strings in inventory order
    table = read_parquet(result.path)
    assert table.column_names == ["fp_bits", "reaction_id"]
    assert table.num_rows == len(HAPPY_ROWS)
    stored_ids = table.column("reaction_id").to_pylist()
    stored_bits = table.column("fp_bits").to_pylist()
    assert stored_ids == [row[0] for row in HAPPY_ROWS]
    for reaction_id, bits in zip(stored_ids, stored_bits, strict=True):
        assert bits == _drfp_bits(dict(HAPPY_ROWS)[reaction_id])

    # And: loading rebuilds the exact same vectors
    loaded = load_fingerprints(config)
    in_memory = dict(result.fingerprints)
    assert [reaction_id for reaction_id, _ in loaded] == stored_ids
    for reaction_id, vector in loaded:
        assert vector.GetNumBits() == FP_SIZE
        assert (
            DataStructs.BulkTanimotoSimilarity(vector, [in_memory[reaction_id]])[0]
            == 1.0
        )

    # And: re-running produces byte-identical Parquet
    first_bytes = result.path.read_bytes()
    compute_fingerprints(config)
    assert result.path.read_bytes() == first_bytes
    assert result.path.name == FINGERPRINTS_PARQUET_FILENAME


def test_budget_abort_writes_incomplete_manifest_and_ledger(tmp_path: Path) -> None:
    # Given: a zero-second budget over the standard fixture
    config = _happy_config(tmp_path, max_seconds=0.0)
    fingerprints = compute_fingerprints(config)

    # When: the audit starts
    with pytest.raises(AuditBudgetExceeded, match="budget exceeded"):
        cross_split_leak_audit(config, fingerprints.fingerprints)

    # Then: the manifest is explicitly incomplete and never claims clean
    manifest = _manifest(config)
    assert manifest["complete"] is False
    assert manifest["reason"] == BUDGET_EXCEEDED_REASON
    assert manifest["n_probes"] == 3
    assert manifest["n_references"] == 3
    assert manifest["n_comparisons"] == 9
    assert manifest["n_probes_completed"] == 0
    assert manifest["n_comparisons_completed"] == 0
    assert manifest["top_offenses"] == []
    assert "leak_status" not in manifest
    assert "clean" not in json.dumps(manifest).lower()

    # And: the typed rejection is appended to the unified ledger
    records = _ledger_records(config)
    assert [record["code"] for record in records] == ["AUDIT_BUDGET_EXCEEDED"]
    assert records[0]["stage"] == AUDIT_STAGE
    assert records[0]["reaction_id"] in {_rid(4), _rid(5), _rid(6)}
    assert str(Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME) == (
        records[0]["source_pointer"]
    )


def test_known_cross_split_duplicate_is_excluded_but_reported(tmp_path: Path) -> None:
    # Given: a train/test identity-equal reverse pair recorded in the ledger
    rows = (
        (_rid(1), "CCO>>CC=O"),
        (_rid(2), "c1ccccc1.Br>>Brc1ccccc1"),
        (_rid(3), "CC=O>>CCO"),
    )
    assignment = ((_rid(1), "train"), (_rid(2), "train"), (_rid(3), "test"))
    config = _config(tmp_path, max_seconds=7200.0)
    interim = Path(config["paths"]["interim"])
    _write_inventory(interim / INVENTORY_NAME, rows)
    _write_assignment(interim / SPLIT_ASSIGNMENT_FILENAME, assignment)
    manifests_dir = Path(config["paths"]["manifests"])
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / DUPLICATE_LEDGER_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "g0_manifest_v1",
                "dataset_version": "zenodo-1-rev1",
                "n_inventory_rows": 3,
                "n_unique_identities": 2,
                "n_duplicate_groups": 1,
                "n_exact_duplicates": 0,
                "n_reverse_pairs": 1,
                "groups": [
                    {
                        "identity": "0" * 64,
                        "member_reaction_ids": [_rid(1), _rid(3)],
                        "canonical_reaction_id": _rid(1),
                    }
                ],
                "pairs": [
                    {
                        "kind": "reverse",
                        "canonical_reaction_id": _rid(1),
                        "other_reaction_id": _rid(3),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    vectors = _drfp_vectors(rows)
    assert _similarity(vectors, _rid(3), _rid(1)) >= THRESHOLD
    assert _similarity(vectors, _rid(3), _rid(2)) < THRESHOLD

    # When: the audit runs with the ledger on disk
    fingerprints = compute_fingerprints(config)
    result = cross_split_leak_audit(config, fingerprints.fingerprints)

    # Then: the known duplicate is reported, not double-counted as an offense
    manifest = _manifest(config)
    assert result.complete is True
    assert result.n_pairs_over_threshold == 0
    assert manifest["n_pairs_over_threshold"] == 0
    assert manifest["n_cross_split_known_duplicates"] == 1
    assert manifest["top_offenses"] == []
    # And: the overall maximum still exposes the 1.0 similarity honestly
    assert manifest["max_similarity"] == pytest.approx(1.0, abs=1e-12)
    assert _ledger_records(config) == []


def test_blocked_scan_still_compares_every_pair(tmp_path: Path) -> None:
    # Given: a block size smaller than the reference count
    config = _happy_config(tmp_path, block_size=2)

    # When
    fingerprints = compute_fingerprints(config)
    result = cross_split_leak_audit(config, fingerprints.fingerprints)

    # Then: every probe x reference comparison still happens exactly once
    manifest = _manifest(config)
    assert manifest["n_comparisons"] == 3 * 3
    assert result.n_pairs_over_threshold == 1


def test_max_pairs_cap_keeps_best_offenses_but_count_stays_exact(
    tmp_path: Path,
) -> None:
    # Given: three identical train references and three identical probes
    rows = tuple((_rid(number), "CCO>>CC=O") for number in range(1, 7))
    assignment = tuple(
        [(_rid(number), "train") for number in range(1, 4)]
        + [(_rid(number), "test") for number in range(4, 7)]
    )
    config = _config(tmp_path, max_pairs=2)
    interim = Path(config["paths"]["interim"])
    _write_inventory(interim / INVENTORY_NAME, rows)
    _write_assignment(interim / SPLIT_ASSIGNMENT_FILENAME, assignment)

    # When
    fingerprints = compute_fingerprints(config)
    result = cross_split_leak_audit(config, fingerprints.fingerprints)

    # Then: the count is exact while only the two best records are stored
    manifest = _manifest(config)
    assert result.n_pairs_over_threshold == 9
    assert manifest["n_pairs_over_threshold"] == 9
    assert len(manifest["top_offenses"]) == 2
    assert [
        (record["probe_reaction_id"], record["reference_reaction_id"])
        for record in manifest["top_offenses"]
    ] == [(_rid(4), _rid(1)), (_rid(4), _rid(2))]


def test_missing_inputs_raise_with_actionable_hints(tmp_path: Path) -> None:
    # Given: a fixture whose split assignment is missing
    config = _happy_config(tmp_path)
    (Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME).unlink()

    # When / Then: the split hint is named before any encoding work
    with pytest.raises(FileNotFoundError, match="g0 split"):
        compute_fingerprints(config)

    # And: an inventory without the split hint is named too
    config = _happy_config(tmp_path / "second")
    (Path(config["paths"]["interim"]) / INVENTORY_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="g0 inventory"):
        compute_fingerprints(config)

    # And: an audit without a fingerprint table names the audit hint
    config = _happy_config(tmp_path / "third")
    with pytest.raises(FileNotFoundError, match="g0 audit"):
        cross_split_leak_audit(config)


def test_audit_never_silently_skips_reactions_without_fingerprints(
    tmp_path: Path,
) -> None:
    # Given: an explicit but empty fingerprint table
    config = _happy_config(tmp_path)

    # When / Then: the audit refuses to treat missing fingerprints as clean
    with pytest.raises(ValueError, match="Fingerprints missing"):
        cross_split_leak_audit(config, ())


def test_two_hundred_reaction_audit_completes_under_thirty_seconds(
    tmp_path: Path,
) -> None:
    # Given: 200 reactions drawn from a fixed pool (160 train, 20 valid, 20 test)
    pool = (
        TRAIN_ANALOG,
        TEST_ANALOG,
        TRAIN_AROMATIC,
        TRAIN_DIELS,
        TEST_UNRELATED,
        VALID_AMIDE,
        "CC(=O)O.CO>>CC(=O)OC.O",
        "CC(=O)O.CCO>>CC(=O)OCC.O",
        "CBr.[OH-]>>CO.[Br-]",
        "C=C.O>>CCO",
    )
    rows = tuple((_rid(number), pool[number % len(pool)]) for number in range(1, 201))
    assignment = tuple(
        [(_rid(number), "train") for number in range(1, 161)]
        + [(_rid(number), "valid") for number in range(161, 181)]
        + [(_rid(number), "test") for number in range(181, 201)]
    )
    config = _config(tmp_path)
    interim = Path(config["paths"]["interim"])
    _write_inventory(interim / INVENTORY_NAME, rows)
    _write_assignment(interim / SPLIT_ASSIGNMENT_FILENAME, assignment)

    # When: fingerprints plus the full 40 x 160 audit run
    start = time.perf_counter()
    fingerprints = compute_fingerprints(config)
    result = cross_split_leak_audit(config, fingerprints.fingerprints)
    elapsed = time.perf_counter() - start

    # Then: the run is complete, fully audited, and well inside the budget
    assert result.complete is True
    assert result.n_pairs_over_threshold > 0
    manifest = _manifest(config)
    assert manifest["n_probes"] == 40
    assert manifest["n_references"] == 160
    assert manifest["n_comparisons"] == 6400
    assert elapsed < 30.0, f"200-reaction audit took {elapsed:.1f}s; budget 30s"


def test_cli_audit_exits_zero_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the CLI whose config loader returns the fixture config
    config = _happy_config(tmp_path)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "audit"])

    # Then
    assert exit_code == 0
    assert (Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME).is_file()


def test_cli_audit_budget_abort_exits_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a zero-second audit budget
    config = _happy_config(tmp_path, max_seconds=0.0)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "audit"])

    # Then: the documented budget exit code is returned and nothing is complete
    assert exit_code == EXIT_AUDIT_BUDGET_EXCEEDED
    manifest = _manifest(config)
    assert manifest["complete"] is False
    assert _ledger_records(config)[0]["code"] == "AUDIT_BUDGET_EXCEEDED"


@pytest.mark.realdata
def test_real_data_audit_completes() -> None:
    # Given: the real inventory and split, when they have been built
    if not (REAL_INVENTORY_PATH.is_file() and REAL_SPLIT_PATH.is_file()):
        pytest.skip(f"real inventory/split not present: {REAL_INVENTORY_PATH.parent}")
    from pes2ts_core.config_loader import load_config

    config = load_config()
    for key in ("raw", "interim", "manifests"):
        config["paths"][key] = str(PROJECT_ROOT / config["paths"][key])

    # When: the full real-data fingerprinting and audit run
    fingerprints = compute_fingerprints(config)
    result = cross_split_leak_audit(config, fingerprints.fingerprints)

    # Then: every probe x train comparison is accounted for
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.complete is True
    assert manifest["complete"] is True
    assert manifest["n_comparisons"] == manifest["n_probes"] * manifest["n_references"]
    assert manifest["n_probes"] > 0
    assert manifest["n_references"] > 0
