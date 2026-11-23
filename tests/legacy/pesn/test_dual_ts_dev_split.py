"""V4.2 internal dev split tests — synthetic fixtures.

Covers: identity-cluster unit (clusters never span train/dev), deterministic
lexicographic ordering + positions 5/15/25/35/45, the degenerate
contains-insertion replacement rule (count 0 or 5 → one swap from position
46), determinism (two runs → identical split_sha256), and load verification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.dual_ts_dev_split import (
    DEV_POSITIONS,
    DevSplitError,
    build_dev_split,
    load_dev_split,
)


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _synthetic_lineage(tmp_path: Path, n_groups: int = 50) -> Path:
    gids = [f"G{i:03d}" for i in range(n_groups)]
    lineage = {
        "schema_version": "pesn_dual_ts_split_lineage_v1",
        "tool_version": "x",
        "created_utc": "2026-10-17T00:00:00Z",
        "sources": [{"role": "x", "path": "x", "sha256": "0" * 64}],
        "v1_binding": {
            "reaction_to_group": {g: g for g in gids},
            "reaction_binding_evidence": {g: "selection_evidenced" for g in gids},
        },
        "partitions": {"fit": {"group_ids": gids, "n_groups": n_groups,
                               "branch_potential": n_groups * 2},
                       "monitor": {"group_ids": [], "n_groups": 0,
                                   "branch_potential": 0},
                       "test": {"group_ids": [], "n_groups": 0,
                                "branch_potential": 0},
                       "family_holdout": {"group_ids": [], "n_groups": 0,
                                          "branch_potential": 0}},
        "dual_ts_wave_selections": {
            "wave": {"fit": {"group_ids": gids, "n_groups": n_groups,
                             "n_reaction_ids": n_groups,
                             "selection": "direct_fit50"},
                     "monitor": {"group_ids": [], "n_groups": 0},
                     "test": {"group_ids": [], "n_groups": 0}},
        },
        "family_audit": {"sealed_families": [], "n_sealed_families": 0,
                         "sealed_reaction_ids": [], "n_sealed_reactions": 0,
                         "group_exposure_per_partition": {}, "conflicts": [],
                         "note": ""},
        "unassigned": [],
        "audit": {"selection_partition_overlap_conflicts": [],
                  "group_partition_conflicts": [],
                  "isolated_but_partitioned_reaction_ids": [],
                  "binding_limitations": "", "counts": {}},
        "artifact_sha256": "0" * 64,
    }
    return _write(tmp_path / "lineage.json", lineage)


def _synthetic_manifest(tmp_path: Path, insertion_groups=None) -> Path:
    """Native manifest with 50 reactions; ``insertion_groups`` = set of group
    ids that 'contain insertion' (all others do not).  Default gives dev
    insertion count 4 (no degenerate replacement)."""
    if insertion_groups is None:
        insertion_groups = {"G004", "G014", "G024", "G034"}
    per_reaction = []
    for i in range(50):
        g = f"G{i:03d}"
        per_reaction.append({
            "reaction_id": g,
            "reaction_group_id": g,
            "n_atoms": 8 + (i % 5),
            "actions_native": 100 + i,
            "has_insertion": g in insertion_groups,
            "disputed": False,
            "bond_authority_flag": "geometry_verified",
        })
    manifest = {
        "schema_version": "pes2ts_dual_ts_native_rebuild_v1",
        "per_reaction": per_reaction,
        "population": {"reactions": 50},
    }
    return _write(tmp_path / "manifest.json", manifest)


def _duplicate_groups(tmp_path: Path, groups) -> Path:
    return _write(tmp_path / "duplicate_groups.json", {"groups": groups})


def test_dev_positions_and_lexicographic_order(tmp_path):
    lineage = _synthetic_lineage(tmp_path)
    manifest = _synthetic_manifest(tmp_path)
    payload = build_dev_split(str(lineage), str(manifest),
                              str(tmp_path / "dev_split.json"))
    assert payload["n_clusters"] == 50
    assert payload["dev_group_ids"] == ["G004", "G014", "G024", "G034", "G044"]
    assert payload["dev_positions"] == list(DEV_POSITIONS)
    # 45 train groups remain.
    assert len(payload["train_group_ids"]) == 45


def test_determinism_two_runs_same_hash(tmp_path):
    lineage = _synthetic_lineage(tmp_path)
    manifest = _synthetic_manifest(tmp_path)
    p1 = build_dev_split(str(lineage), str(manifest),
                         str(tmp_path / "a.json"))
    p2 = build_dev_split(str(lineage), str(manifest),
                         str(tmp_path / "b.json"))
    assert p1["split_sha256"] == p2["split_sha256"]


def test_replacement_when_dev_insertion_count_5(tmp_path):
    lineage = _synthetic_lineage(tmp_path)
    # all 5 dev positions contain insertion -> count 5 -> replacement.
    manifest = _synthetic_manifest(tmp_path,
                                   insertion_groups={"G004", "G014", "G024",
                                                     "G034", "G044"})
    payload = build_dev_split(str(lineage), str(manifest),
                              str(tmp_path / "dev_split.json"))
    assert payload["dev_insertion_count"] == 5
    assert payload["replacement"] is not None
    # G045 (position 46) is the first no-insertion cluster -> swaps in.
    assert payload["replacement"]["replacement_cluster"] == "G045"
    assert payload["replacement"]["replaced_cluster"] == "G044"
    assert "G045" in payload["dev_group_ids"]
    assert "G044" not in payload["dev_group_ids"]


def test_no_replacement_when_count_4(tmp_path):
    lineage = _synthetic_lineage(tmp_path)
    manifest = _synthetic_manifest(tmp_path,
                                   insertion_groups={"G004", "G014", "G024",
                                                     "G034"})
    payload = build_dev_split(str(lineage), str(manifest),
                              str(tmp_path / "dev_split.json"))
    assert payload["dev_insertion_count"] == 4
    assert payload["replacement"] is None


def test_identity_cluster_never_spans_train_dev(tmp_path):
    lineage = _synthetic_lineage(tmp_path)
    # make G004 and G040 identity-bound duplicate partners (same canonical).
    dg = _duplicate_groups(tmp_path, [
        {"canonical_reaction_id": "G004",
         "member_reaction_ids": ["G004", "G040"]},
    ])
    manifest = _synthetic_manifest(tmp_path)
    payload = build_dev_split(str(lineage), str(manifest),
                              str(tmp_path / "dev_split.json"),
                              duplicate_groups_path=str(dg))
    # the cluster {G004, G040} must be entirely in dev or entirely in train.
    clusters = payload["clusters"]
    merged = [c for c in clusters if "G004" in c["member_group_ids"]]
    assert len(merged) == 1
    assert sorted(merged[0]["member_group_ids"]) == ["G004", "G040"]
    # the whole cluster's min_group_id is in dev, and G040 never appears as
    # its own separate cluster in either side.
    assert merged[0]["min_group_id"] in payload["dev_group_ids"]
    assert "G040" not in payload["dev_group_ids"]
    assert "G040" not in payload["train_group_ids"]


def test_load_dev_split_verifies_hash(tmp_path):
    lineage = _synthetic_lineage(tmp_path)
    manifest = _synthetic_manifest(tmp_path)
    path = str(tmp_path / "dev_split.json")
    build_dev_split(str(lineage), str(manifest), path)
    loaded = load_dev_split(path)
    assert loaded["schema_version"] == "pesn_dual_ts_dev_split_v42_v1"
    # tamper -> digest mismatch
    bad = json.loads(Path(path).read_text())
    bad["dev_group_ids"] = ["G000"]
    tampered = tmp_path / "tampered.json"
    _write(tampered, bad)
    with pytest.raises(DevSplitError):
        load_dev_split(str(tampered))


def test_too_few_groups_refuses(tmp_path):
    lineage = _synthetic_lineage(tmp_path, n_groups=10)
    manifest = _synthetic_manifest(tmp_path)
    with pytest.raises(DevSplitError):
        build_dev_split(str(lineage), str(manifest),
                        str(tmp_path / "dev_split.json"))
