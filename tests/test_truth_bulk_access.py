"""Audited bulk truth accessors: permission gate, audit log, streaming."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pytest

from pes2ts_core.g0.truth import truth_reader
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import write_parquet


def _build_truth_tree(root: Path) -> Path:
    manifests = root / "manifests"
    ground_truth = root / "ground_truth"
    sources = ground_truth / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    ts_path = ground_truth / "ts.parquet"
    write_parquet(ts_path, pa.table({
        "reaction_id": ["RXN_0000000001", "RXN_0000000002"],
        "atomic_numbers": [[8, 1, 1], [8, 1, 1]],
        "coordinates": [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.9, 0.0]],
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.9, 0.0]],
        ],
        "EHG": [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        "charge": [0, 0], "multiplicity": [1, 1],
        "reaction_smiles": ["a>>b", "a>>b"],
    }))
    irc_h5 = sources / "fixture_IRC.h5"
    with h5py.File(irc_h5, "w") as handle:
        for name, n_frames in (("RXN_0000000001", 5), ("RXN_0000000002", 7)):
            group = handle.create_group(f"bundle/{name}")
            group.create_dataset(
                "coordinates",
                data=np.arange(n_frames * 3 * 3, dtype=np.float64).reshape(n_frames, 3, 3),
            )
    write_json(manifests / "truth_manifest.json", {
        "schema_version": "g0_manifest_v1",
        "ts_parquet": {"path": str(ts_path), "sha256": sha256_file(ts_path), "n_rows": 2},
        "irc_index": {"path": str(ground_truth / "irc_index.parquet"), "sha256": "x", "n_rows": 2},
        "sources": [{
            "filename": irc_h5.name, "original_path": str(irc_h5),
            "relocated_path": str(irc_h5), "sha256": sha256_file(irc_h5),
            "size_bytes": irc_h5.stat().st_size,
        }],
        "n_reactions_without_ts": 0,
    })
    return manifests


def _log_entries(manifests: Path) -> list[dict[str, object]]:
    path = manifests / "truth_access_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_load_ts_table_bulk_reads_and_audits(tmp_path: Path) -> None:
    manifests = _build_truth_tree(tmp_path)
    with pytest.raises(PermissionError):
        truth_reader.load_ts_table(False, manifests_dir=manifests)
    table = truth_reader.load_ts_table(True, manifests_dir=manifests)
    assert table.num_rows == 2
    entries = _log_entries(manifests)
    assert entries[-1]["function"] == "load_ts_table"
    assert entries[-1]["reaction_id"] == "<ts_table>"


def test_iter_irc_trajectories_streams_every_reaction_with_audit(tmp_path: Path) -> None:
    manifests = _build_truth_tree(tmp_path)
    with pytest.raises(PermissionError):
        truth_reader.iter_irc_trajectories(False, manifests_dir=manifests)
    stream: Iterator[object] = truth_reader.iter_irc_trajectories(
        True, manifests_dir=manifests, caller="tests",
    )
    collected = [frames.reaction_id for frames in stream]
    assert collected == ["RXN_0000000001", "RXN_0000000002"]
    entries = _log_entries(manifests)
    streamed = [entry for entry in entries if entry["function"] == "iter_irc_trajectories"]
    assert [entry["reaction_id"] for entry in streamed] == collected
    assert all(entry.get("caller") == "tests" for entry in streamed)


def test_per_reaction_lookup_records_caller(tmp_path: Path) -> None:
    manifests = _build_truth_tree(tmp_path)
    frames = truth_reader.load_irc_frames(
        "RXN_0000000002", True, manifests_dir=manifests, caller="unit",
    )
    assert frames["n_frames"] == 7
    entries = _log_entries(manifests)
    assert entries[-1]["caller"] == "unit"
