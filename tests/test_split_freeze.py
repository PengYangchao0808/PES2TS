"""Offline tests for the split freeze stage and its leak decision policy.

Fixtures are synthetic and built under ``tmp_path``: three split CSVs, an
interim inventory Parquet, and a minimal config. Each fixture runs the real
prerequisite chain (``adopt_official_split`` -> ``compute_fingerprints`` ->
``cross_split_leak_audit``) before ``freeze_split`` is exercised, so the tests
assert the policy against genuine audit artifacts instead of hand-written
verdicts. No HDF5, network, or real-data access is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from pes2ts_core.cli import main
from pes2ts_core.g0 import VOLATILE_KEYS
from pes2ts_core.g0.dedup import DUPLICATE_LEDGER_FILENAME
from pes2ts_core.g0.fingerprints import load_fingerprints
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.neardup import (
    LEAK_AUDIT_FILENAME,
    LeakAuditResult,
    compute_fingerprints,
    cross_split_leak_audit,
)
from pes2ts_core.g0.rejections import LEDGER_FILENAME
from pes2ts_core.g0.split import (
    EXIT_AUDIT_INCOMPLETE,
    EXIT_LEAK_FOUND,
    REMEDIATION_EXCLUDE_LEAKY,
    REMEDIATION_REBUILD,
    SPLIT_ASSIGNMENT_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    AuditIncompleteError,
    RemediationFailedError,
    adopt_official_split,
    freeze_split,
)
from pes2ts_core.utils.hashing import md5_file
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

TRAIN_ANALOG = "CCCCCCCC(=O)O.CO>>CCCCCCCC(=O)OC.O"
TEST_ANALOG = "CCCCCCCCC(=O)O.CO>>CCCCCCCCC(=O)OC.O"
TRAIN_AROMATIC = "c1ccc(cc1)C(=O)O.CCO>>c1ccc(cc1)C(=O)OCC.O"
TRAIN_DIELS = "C=CC=C.C=C>>C1CC=CCC1"
TEST_UNRELATED = "c1ccccc1.Br>>Brc1ccccc1"
VALID_AMIDE = "CC(=O)O.CCN>>CC(=O)NCC.O"
THRESHOLD = 0.90
CLUSTER_THRESHOLD = 0.80


def _rid(number: int) -> str:
    """Return the canonical reaction ID for *number*."""
    return f"RXN_{number:010d}"


#: A held-out analog of the train ester pair: the audit must flag it.
LEAK_ROWS: tuple[tuple[str, str], ...] = (
    (_rid(1), TRAIN_ANALOG),
    (_rid(2), TRAIN_AROMATIC),
    (_rid(3), TRAIN_DIELS),
    (_rid(4), TEST_ANALOG),
    (_rid(5), TEST_UNRELATED),
    (_rid(6), VALID_AMIDE),
)
#: The leak fixture also works without rid4; these probes are all unrelated.
CLEAN_ROWS: tuple[tuple[str, str], ...] = (
    (_rid(1), TRAIN_ANALOG),
    (_rid(2), TRAIN_AROMATIC),
    (_rid(3), TRAIN_DIELS),
    (_rid(5), TEST_UNRELATED),
    (_rid(6), VALID_AMIDE),
)
#: Identity-equal written reverses recorded in the duplicate ledger. The
#: second train reaction reuses the verified unrelated fixture fact.
KNOWN_ROWS: tuple[tuple[str, str], ...] = (
    (_rid(1), "CCO>>CC=O"),
    (_rid(2), TEST_UNRELATED),
    (_rid(3), "CC=O>>CCO"),
)


def _split_rows(train: tuple[str, ...], valid: tuple[str, ...], test: tuple[str, ...]) -> dict[str, list[str]]:
    """Return the fixture split definition keyed by label."""
    return {"train": list(train), "valid": list(valid), "test": list(test)}


LEAK_SPLITS = _split_rows(
    (_rid(1), _rid(2), _rid(3)), (_rid(6),), (_rid(4), _rid(5))
)
CLEAN_SPLITS = _split_rows((_rid(1), _rid(2), _rid(3)), (_rid(6),), (_rid(5),))
KNOWN_SPLITS = _split_rows((_rid(1), _rid(2)), (), (_rid(3),))
KNOWN_LEDGER: dict[str, Any] = {
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


def _base_config(root: Path) -> dict[str, Any]:
    """Build a minimal freeze config rooted at *root*."""
    return {
        "source": {
            "zenodo_record": 1,
            "zenodo_revision": "1",
            "files": {},
        },
        "paths": {
            "raw": str(root / "raw"),
            "interim": str(root / "interim"),
            "manifests": str(root / "manifests"),
        },
        "split": {"seed": 42},
        "near_dup": {
            "threshold": THRESHOLD,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "fp_size": 2048,
            "block_size": 1000,
            "max_seconds": 7200.0,
            "max_pairs": 10000,
        },
    }


def _write_split_sources(config: dict[str, Any], splits: dict[str, list[str]]) -> None:
    """Write the three fixture split CSVs and register their configured md5s."""
    raw = Path(config["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    for label, suffix in (
        ("train", "_train.csv"),
        ("valid", "_valid.csv"),
        ("test", "_test.csv"),
    ):
        name = f"fixture-RXN{suffix}"
        path = raw / name
        path.write_text(
            "\n".join(["reaction_id", *splits[label]]) + "\n", encoding="utf-8"
        )
        config["source"]["files"][name] = {
            "filename": name,
            "url": f"https://example.invalid/{name}",
            "md5": md5_file(path),
        }


def _write_inventory(config: dict[str, Any], rows: tuple[tuple[str, str], ...]) -> None:
    """Write the two-column inventory Parquet."""
    write_parquet(
        Path(config["paths"]["interim"]) / INVENTORY_PARQUET_FILENAME,
        {
            "reaction_id": [row[0] for row in rows],
            "reaction_smiles": [row[1] for row in rows],
        },
    )


def _prepare(
    root: Path,
    rows: tuple[tuple[str, str], ...],
    splits: dict[str, list[str]],
    *,
    duplicate_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full prerequisite chain and return the fixture config."""
    config = _base_config(root)
    _write_split_sources(config, splits)
    _write_inventory(config, rows)
    adopt_official_split(config)
    if duplicate_ledger is not None:
        write_json(
            Path(config["paths"]["manifests"]) / DUPLICATE_LEDGER_FILENAME,
            duplicate_ledger,
        )
    fingerprints = compute_fingerprints(config)
    cross_split_leak_audit(config, fingerprints.fingerprints)
    return config


def _manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Return the parsed split manifest of *config*."""
    path = Path(config["paths"]["manifests"]) / SPLIT_MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_bytes(config: dict[str, Any]) -> bytes:
    """Return the leak-audit manifest bytes of *config*."""
    return (Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME).read_bytes()


def _assignment(config: dict[str, Any]) -> dict[str, str]:
    """Return ``reaction_id -> split`` from the assignment Parquet."""
    table = read_parquet(
        Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    )
    return {
        str(row["reaction_id"]): str(row["split"])
        for row in table.to_pylist()
    }


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


def _without_volatile(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return *manifest* without the keys allowed to change between runs."""
    return {key: value for key, value in manifest.items() if key not in VOLATILE_KEYS}


def _independent_clusters(config: dict[str, Any]) -> tuple[list[str], list[list[int]]]:
    """Recompute Butina clusters without the production helper.

    Distances are rebuilt here from the persisted fingerprints with the same
    condensed layout ``Butina.ClusterData`` expects, so the "no cluster
    straddles splits" assertion is independent of
    :mod:`pes2ts_core.g0.split_rebuild`.
    """
    by_id = dict(load_fingerprints(config))
    sorted_ids = sorted(by_id)
    vectors = [by_id[reaction_id] for reaction_id in sorted_ids]
    distances: list[float] = []
    for index in range(1, len(vectors)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            vectors[index], vectors[:index]
        )
        distances.extend(1.0 - float(value) for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(vectors),
        1.0 - CLUSTER_THRESHOLD,
        isDistData=True,
        reordering=False,
    )
    return sorted_ids, [[int(index) for index in cluster] for cluster in clusters]


def test_freeze_halts_on_leak_without_touching_assignment(tmp_path: Path) -> None:
    # Given: a test analog of a train ester pair plus the full prerequisite chain
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    assignment_path = Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    before = assignment_path.read_bytes()

    # When: the default freeze runs
    result = freeze_split(config)

    # Then: it halts, leaves the assignment byte-identical, and names the pair
    assert result.leak_status == "LEAK_FOUND"
    assert result.halted is True
    assert result.skipped is False
    assert result.n_pairs_over_threshold >= 1
    assert result.excluded_ids == ()
    assert assignment_path.read_bytes() == before

    manifest = _manifest(config)
    assert manifest["leak_status"] == "LEAK_FOUND"
    assert manifest["frozen"] is True
    assert manifest["remediation"] == {"applied": "none", "reason": "halt-no-flag"}
    assert manifest["counts"] == {"train": 3, "valid": 1, "test": 2, "total": 6}
    assert manifest["covered"] == 6
    assert manifest["audit"]["complete"] is True
    assert manifest["audit"]["n_pairs_over_threshold"] == result.n_pairs_over_threshold
    offense = manifest["offending_tuples"][0]
    assert offense["probe_reaction_id"] == _rid(4)
    assert offense["reference_reaction_id"] == _rid(1)
    assert offense["kind"] == "near_dup"
    assert offense["similarity"] >= THRESHOLD
    assert manifest["threshold"] == THRESHOLD

    # And: nothing was excluded yet
    assert [
        record["code"]
        for record in _ledger_records(config)
        if record["code"] == "LEAK_EXCLUDED"
    ] == []


def test_cli_freeze_halts_with_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a leaky fixture whose config loader returns the fixture config
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "freeze"])

    # Then: the documented leak exit code is returned and the split is untouched
    assert exit_code == EXIT_LEAK_FOUND
    assert _manifest(config)["leak_status"] == "LEAK_FOUND"


def test_exclude_leaky_removes_probe_and_freezes_clean(tmp_path: Path) -> None:
    # Given: the leaky fixture
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    train_before = {
        reaction_id
        for reaction_id, label in _assignment(config).items()
        if label == "train"
    }

    # When: the explicit exclusion remediation runs
    result = freeze_split(config, REMEDIATION_EXCLUDE_LEAKY)

    # Then: only the held-out analog is removed, train stays intact, audit clean
    assert result.leak_status == "clean"
    assert result.halted is False
    assert result.skipped is False
    assert result.excluded_ids == (_rid(4),)
    assert result.n_pairs_over_threshold == 0
    assert result.n_cross_split_known_duplicates == 0

    assignment = _assignment(config)
    assert _rid(4) not in assignment
    assert {
        reaction_id for reaction_id, label in assignment.items() if label == "train"
    } == train_before
    assert assignment[_rid(5)] == "test"
    assert assignment[_rid(6)] == "valid"

    ledger_records = _ledger_records(config)
    excluded = [
        record for record in ledger_records if record["code"] == "LEAK_EXCLUDED"
    ]
    assert len(excluded) == 1
    assert excluded[0]["reaction_id"] == _rid(4)
    assert excluded[0]["stage"] == "freeze_split"
    assert _rid(1) in excluded[0]["detail"]

    manifest = _manifest(config)
    assert manifest["leak_status"] == "clean"
    assert manifest["strategy"] == "authors_official"
    assert manifest["counts"] == {"train": 3, "valid": 1, "test": 1, "total": 5}
    assert manifest["covered"] == 5
    assert manifest["n_inventory"] == 6
    assert manifest["n_not_in_split"] == 0
    assert manifest["remediation"] == {
        "applied": "exclude-leaky",
        "excluded_ids": [_rid(4)],
    }
    assert manifest["audit"]["n_pairs_over_threshold"] == 0
    assert manifest["audit"]["n_cross_split_known_duplicates"] == 0
    assert manifest["offending_tuples"][0]["probe_reaction_id"] == _rid(4)


def test_cli_freeze_exclude_leaky_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a leaky fixture and the CLI wired to it
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "freeze", "--remediation", "exclude-leaky"])

    # Then
    assert exit_code == 0
    assert _manifest(config)["leak_status"] == "clean"
    assert _rid(4) not in _assignment(config)


def test_rebuild_preserves_clusters_and_is_deterministic(tmp_path: Path) -> None:
    # Given: two identical leaky fixtures in independent roots
    config_a = _prepare(tmp_path / "a", LEAK_ROWS, LEAK_SPLITS)
    config_b = _prepare(tmp_path / "b", LEAK_ROWS, LEAK_SPLITS)

    # When: the rebuild remediation runs in both
    result_a = freeze_split(config_a, REMEDIATION_REBUILD)
    _ = freeze_split(config_b, REMEDIATION_REBUILD)

    # Then: the frozen split is clean and carries the rebuilt strategy
    assert result_a.leak_status == "clean"
    assert result_a.halted is False
    manifest = _manifest(config_a)
    assert manifest["strategy"] == "rebuilt_butina_v1"
    assert manifest["leak_status"] == "clean"
    assert manifest["remediation"]["applied"] == "rebuild"
    rebuilt = manifest["remediation"]["rebuilt"]
    assert rebuilt["cluster_cutoff"] == pytest.approx(1.0 - CLUSTER_THRESHOLD)
    assert rebuilt["seed"] == 42
    assert rebuilt["n_clusters"] >= 1
    assert manifest["counts"]["total"] == 6
    assert manifest["covered"] == manifest["n_inventory"] == 6

    # And: every recomputed Butina cluster lies entirely inside one split
    assignment = _assignment(config_a)
    sorted_ids, clusters = _independent_clusters(config_a)
    assert set(assignment) == set(sorted_ids)
    for cluster in clusters:
        assert len({assignment[sorted_ids[index]] for index in cluster}) == 1

    # And: two independent full rebuild runs match modulo volatile keys
    assert _without_volatile(manifest) == _without_volatile(_manifest(config_b))
    assignment_a = Path(config_a["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    assignment_b = Path(config_b["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    assert assignment_a.read_bytes() == assignment_b.read_bytes()


def test_clean_audit_freezes_clean_and_second_call_skips(tmp_path: Path) -> None:
    # Given: a fixture whose probes are all unrelated to train
    config = _prepare(tmp_path, CLEAN_ROWS, CLEAN_SPLITS)
    manifest_path = Path(config["paths"]["manifests"]) / SPLIT_MANIFEST_FILENAME

    # When: the default freeze runs
    result = freeze_split(config)

    # Then: the verdict is clean with no offending tuples
    assert result.leak_status == "clean"
    assert result.halted is False
    assert result.skipped is False
    manifest = _manifest(config)
    assert manifest["leak_status"] == "clean"
    assert manifest["remediation"] == {"applied": "none"}
    assert manifest["offending_tuples"] == []
    assert manifest["audit"]["n_pairs_over_threshold"] == 0

    # And: a second call skips without rewriting the manifest
    before = manifest_path.read_bytes()
    mtime = manifest_path.stat().st_mtime_ns
    second = freeze_split(config)
    assert second.skipped is True
    assert second.leak_status == "clean"
    assert manifest_path.read_bytes() == before
    assert manifest_path.stat().st_mtime_ns == mtime


def test_clean_audit_ignores_explicit_remediation_flag(tmp_path: Path) -> None:
    # Given: a clean fixture asked for exclusion remediation
    config = _prepare(tmp_path, CLEAN_ROWS, CLEAN_SPLITS)

    # When
    result = freeze_split(config, REMEDIATION_EXCLUDE_LEAKY)

    # Then: no remediation is applied and the note says why
    assert result.leak_status == "clean"
    assert result.excluded_ids == ()
    manifest = _manifest(config)
    assert manifest["remediation"] == {
        "applied": "none",
        "note": "no-remediation-needed",
    }
    assert _ledger_records(config) == []


def test_incomplete_audit_can_never_be_frozen(tmp_path: Path) -> None:
    # Given: an audit manifest hand-edited to complete=false
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    audit_path = Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME
    document = json.loads(audit_path.read_text(encoding="utf-8"))
    document["complete"] = False
    audit_path.write_text(json.dumps(document), encoding="utf-8")

    # When / Then: every remediation path is refused before any freeze logic
    with pytest.raises(AuditIncompleteError, match="never be frozen"):
        freeze_split(config)
    with pytest.raises(AuditIncompleteError, match="never be frozen"):
        freeze_split(config, REMEDIATION_REBUILD)

    # And: the adoption manifest is still pending, never clean
    manifest = _manifest(config)
    assert manifest["leak_status"] == "pending"
    assert "frozen" not in manifest


def test_cli_freeze_incomplete_audit_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a tampered incomplete audit and the CLI wired to the fixture
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    audit_path = Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME
    document = json.loads(audit_path.read_text(encoding="utf-8"))
    document["complete"] = False
    audit_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "freeze"])

    # Then
    assert exit_code == EXIT_AUDIT_INCOMPLETE
    assert _manifest(config)["leak_status"] == "pending"


def test_known_cross_split_duplicate_halts_then_excludes(tmp_path: Path) -> None:
    # Given: a train/test identity-equal reverse pair recorded in the ledger
    config = _prepare(
        tmp_path, KNOWN_ROWS, KNOWN_SPLITS, duplicate_ledger=KNOWN_LEDGER
    )

    # When: the default freeze runs
    result = freeze_split(config)

    # Then: the known duplicate alone triggers the halt, with its own kind
    assert result.leak_status == "LEAK_FOUND"
    assert result.halted is True
    assert result.n_pairs_over_threshold == 0
    assert result.n_cross_split_known_duplicates == 1
    manifest = _manifest(config)
    assert manifest["remediation"] == {"applied": "none", "reason": "halt-no-flag"}
    offense = manifest["offending_tuples"][0]
    assert offense["kind"] == "known_duplicate"
    assert offense["probe_reaction_id"] == _rid(3)
    assert offense["reference_reaction_id"] == _rid(1)
    assert offense["similarity"] == pytest.approx(1.0, abs=1e-12)

    # When: the exclusion remediation runs from the halted state
    second = freeze_split(config, REMEDIATION_EXCLUDE_LEAKY)

    # Then: the held-out member is excluded and the split freezes clean
    assert second.leak_status == "clean"
    assert second.excluded_ids == (_rid(3),)
    assignment = _assignment(config)
    assert _rid(3) not in assignment
    assert assignment[_rid(1)] == "train"
    excluded = [
        record
        for record in _ledger_records(config)
        if record["code"] == "LEAK_EXCLUDED"
    ]
    assert len(excluded) == 1
    assert excluded[0]["reaction_id"] == _rid(3)
    assert _rid(1) in excluded[0]["detail"]
    assert _manifest(config)["audit"]["n_cross_split_known_duplicates"] == 0


def test_failed_remediation_restores_artifacts_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a leaky fixture whose candidate re-audit is forced to still leak
    config = _prepare(tmp_path, LEAK_ROWS, LEAK_SPLITS)
    assignment_path = Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    assignment_before = assignment_path.read_bytes()
    audit_before = _audit_bytes(config)
    real_audit = cross_split_leak_audit

    def fake_audit(
        cfg: dict[str, Any],
        fingerprints: Any = None,
        *,
        collect_all_pairs: bool = False,
    ) -> LeakAuditResult:
        result = real_audit(cfg, fingerprints, collect_all_pairs=collect_all_pairs)
        return replace(result, n_pairs_over_threshold=1)

    monkeypatch.setattr(
        "pes2ts_core.g0.split_remediate.cross_split_leak_audit", fake_audit
    )

    # When: the exclusion remediation cannot reach a clean candidate
    with pytest.raises(RemediationFailedError, match="pre-remediation assignment"):
        freeze_split(config, REMEDIATION_EXCLUDE_LEAKY)

    # Then: no mixed state — the assignment is byte-identical and the audit
    # document matches the restored state modulo volatile keys
    assert assignment_path.read_bytes() == assignment_before
    assert _without_volatile(
        json.loads(_audit_bytes(config))
    ) == _without_volatile(json.loads(audit_before))
    manifest = _manifest(config)
    assert manifest["leak_status"] == "LEAK_FOUND"
    assert manifest["remediation"]["applied"] == "exclude-leaky"
    assert manifest["remediation"]["status"] == "failed"
    assert [
        record
        for record in _ledger_records(config)
        if record["code"] == "LEAK_EXCLUDED"
    ] == []


def test_freeze_determinism_two_full_exclude_runs(tmp_path: Path) -> None:
    # Given: two identical leaky fixtures
    config_a = _prepare(tmp_path / "a", LEAK_ROWS, LEAK_SPLITS)
    config_b = _prepare(tmp_path / "b", LEAK_ROWS, LEAK_SPLITS)

    # When: both run the exclusion remediation
    _ = freeze_split(config_a, REMEDIATION_EXCLUDE_LEAKY)
    _ = freeze_split(config_b, REMEDIATION_EXCLUDE_LEAKY)

    # Then: manifests, assignments, and ledgers match after stripping volatiles
    assert _without_volatile(_manifest(config_a)) == _without_volatile(
        _manifest(config_b)
    )
    assignment_a = Path(config_a["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    assignment_b = Path(config_b["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    assert assignment_a.read_bytes() == assignment_b.read_bytes()
    # The ledger's source_pointer embeds the fixture root by design, so compare
    # the root-independent record fields instead.
    ledger_fields = ("reaction_id", "stage", "code", "detail")
    assert [
        tuple(record[field] for field in ledger_fields)
        for record in _ledger_records(config_a)
    ] == [
        tuple(record[field] for field in ledger_fields)
        for record in _ledger_records(config_b)
    ]
