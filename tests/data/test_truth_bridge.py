"""Tests for the R1 truth bridge (``pes2ts_data.truth_bridge``).

Fixtures are synthetic truth artifacts under ``tmp_path`` (a minimal
``truth_manifest.json`` + ``ts.parquet`` + ``irc_index.parquet`` + frozen split
manifest), so the real quarantined truth and the real audit log are never read
or written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pes2ts_data
from pes2ts_data.truth_bridge import (
    BRIDGE_CALLER,
    TruthBridgeError,
    load_irc_frames,
    load_irc_index,
    load_ts_geometry,
    reaction_in_frozen_split,
)

REACTION_ID = "RXN_0000000007"
HOLDOUT_ID = "RXN_0000000009"


def _build_truth_root(tmp_path: Path) -> tuple[Path, Path]:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    ts_path = tmp_path / "ts.parquet"
    pq.write_table(pa.table({
        "reaction_id": [REACTION_ID],
        "atomic_numbers": [[6, 6, 6]],
        "coordinates": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        "EHG": [[0.0, 0.0, 0.0]],
        "charge": [0],
        "multiplicity": [1],
        "reaction_smiles": ["C>>C"],
    }), ts_path)
    index_path = tmp_path / "irc_index.parquet"
    pq.write_table(pa.table({
        "reaction_id": [REACTION_ID],
        "n_frames": [3],
    }), index_path)
    (tmp_path / "B3LYPD3_TZVP_IRC.h5").write_bytes(b"not-a-real-hdf5")
    (manifests / "truth_manifest.json").write_text(json.dumps({
        "schema_version": "g0_manifest_v1",
        "ts_parquet": {"path": str(ts_path), "n_rows": 1},
        "irc_index": {"path": str(index_path), "n_rows": 1},
        "sources": [{"filename": "B3LYPD3_TZVP_IRC.h5",
                     "relocated_path": str(tmp_path / "B3LYPD3_TZVP_IRC.h5")}],
    }), encoding="utf-8")
    split_path = tmp_path / "split_ids.json"
    split_path.write_text(json.dumps({
        "schema_version": "pesn_irc_split_v1",
        "subsets": {"train_fit": [REACTION_ID]},
        "family_holdout": {"reaction_ids": [HOLDOUT_ID]},
    }), encoding="utf-8")
    return manifests, split_path


def _audit_lines(manifests: Path) -> list[dict]:
    path = manifests / "truth_access_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Audited access
# ---------------------------------------------------------------------------

def test_ts_read_is_audited(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    row = load_ts_geometry(REACTION_ID, manifests_dir=manifests,
                           split_manifest_path=split_path)
    assert row["reaction_id"] == REACTION_ID
    assert list(row["atomic_numbers"]) == [6, 6, 6]

    lines = _audit_lines(manifests)
    assert len(lines) == 1
    record = lines[0]
    assert record["function"] == "load_ts_geometry"
    assert record["reaction_id"] == REACTION_ID
    assert record["caller"] == BRIDGE_CALLER
    assert record["accessor_module"].endswith("truth_reader")


def test_caller_override_and_second_read_are_audited(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    load_ts_geometry(REACTION_ID, manifests_dir=manifests,
                     split_manifest_path=split_path, caller="zone3.builder:42")
    load_ts_geometry(REACTION_ID, manifests_dir=manifests,
                     split_manifest_path=split_path)
    lines = _audit_lines(manifests)
    assert len(lines) == 2
    assert lines[0]["caller"] == "zone3.builder:42"
    assert lines[1]["caller"] == BRIDGE_CALLER


def test_irc_index_read_is_audited(tmp_path: Path):
    manifests, _ = _build_truth_root(tmp_path)
    rows = load_irc_index(manifests_dir=manifests)
    assert rows and rows[0]["reaction_id"] == REACTION_ID
    lines = _audit_lines(manifests)
    assert len(lines) == 1
    assert lines[0]["function"] == "load_irc_index"
    assert lines[0]["reaction_id"] == "<index>"


# ---------------------------------------------------------------------------
# Refusals (name-based bypass and gate hardening)
# ---------------------------------------------------------------------------

def test_name_based_bypass_is_rejected(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    for name in ("../data/ground_truth/ts.parquet", "data/truth.ts.parquet",
                 "B3LYPD3_TZVP.h5", "truth", ""):
        with pytest.raises(TruthBridgeError) as excinfo:
            load_ts_geometry(name, manifests_dir=manifests,
                             split_manifest_path=split_path)
        assert excinfo.value.code == "INVALID_REACTION_ID", name
    # No refused call may append an audit line (gate runs before any read).
    assert _audit_lines(manifests) == []


def test_gate_override_is_rejected(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    with pytest.raises(TruthBridgeError) as excinfo:
        load_ts_geometry(REACTION_ID, manifests_dir=manifests,
                         split_manifest_path=split_path, allow_truth=False)
    assert excinfo.value.code == "TRUTH_GATE_CLOSED"
    with pytest.raises(TruthBridgeError) as excinfo:
        load_irc_frames(REACTION_ID, manifests_dir=manifests,
                        split_manifest_path=split_path, allow_truth=True)
    assert excinfo.value.code == "TRUTH_GATE_CLOSED"
    with pytest.raises(TruthBridgeError) as excinfo:
        load_irc_index(manifests_dir=manifests, allow_truth=False)
    assert excinfo.value.code == "TRUTH_GATE_CLOSED"
    assert _audit_lines(manifests) == []


def test_reaction_outside_frozen_split_is_rejected(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    with pytest.raises(TruthBridgeError) as excinfo:
        load_ts_geometry("RXN_9999999999", manifests_dir=manifests,
                         split_manifest_path=split_path)
    assert excinfo.value.code == "REACTION_NOT_IN_FROZEN_SPLIT"
    assert _audit_lines(manifests) == []


def test_missing_split_manifest_fails_closed(tmp_path: Path):
    manifests, _ = _build_truth_root(tmp_path)
    with pytest.raises(TruthBridgeError) as excinfo:
        load_ts_geometry(REACTION_ID, manifests_dir=manifests,
                         split_manifest_path=tmp_path / "no_split.json")
    assert excinfo.value.code == "FROZEN_SPLIT_UNAVAILABLE"


def test_missing_truth_artifacts_are_typed(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    # Reaction is frozen but absent from the synthetic TS table.
    with pytest.raises(TruthBridgeError) as excinfo:
        load_ts_geometry(HOLDOUT_ID, manifests_dir=manifests,
                         split_manifest_path=split_path)
    assert excinfo.value.code == "REACTION_NOT_IN_TRUTH"

    (manifests / "truth_manifest.json").unlink()
    with pytest.raises(TruthBridgeError) as excinfo:
        load_ts_geometry(REACTION_ID, manifests_dir=manifests,
                         split_manifest_path=split_path)
    assert excinfo.value.code == "TRUTH_ARTIFACT_MISSING"


def test_irc_frames_missing_archive_is_typed(tmp_path: Path):
    manifests, split_path = _build_truth_root(tmp_path)
    with pytest.raises(TruthBridgeError) as excinfo:
        load_irc_frames(REACTION_ID, manifests_dir=manifests,
                        split_manifest_path=split_path)
    assert excinfo.value.code in {"TRUTH_ARTIFACT_MISSING", "REACTION_NOT_IN_TRUTH"}


# ---------------------------------------------------------------------------
# Membership helper + R1 packaging rules
# ---------------------------------------------------------------------------

def test_reaction_in_frozen_split_helper(tmp_path: Path):
    _, split_path = _build_truth_root(tmp_path)
    assert reaction_in_frozen_split(REACTION_ID, split_manifest_path=split_path)
    assert reaction_in_frozen_split(HOLDOUT_ID, split_manifest_path=split_path)
    assert not reaction_in_frozen_split("RXN_9999999999",
                                        split_manifest_path=split_path)
    assert not reaction_in_frozen_split("../ts.parquet",
                                        split_manifest_path=split_path)


def test_bridge_is_not_reexported_from_package_init():
    assert "truth_bridge" not in pes2ts_data.__all__
    assert not hasattr(pes2ts_data, "load_ts_geometry")
    assert not hasattr(pes2ts_data, "load_irc_frames")


def test_bridge_module_respects_line_budget():
    module_path = (Path(__file__).resolve().parents[2]
                   / "pes2ts_data" / "truth_bridge.py")
    assert len(module_path.read_text(encoding="utf-8").splitlines()) <= 120
