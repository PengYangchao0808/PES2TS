"""B1/B2 acceptance: grouping, freeze scope, and sealed-refusal semantics."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pes2ts_core.pesn.offline.splits.freeze import (
    FAMILY_HOLDOUT,
    SplitFreezeError,
    TRAIN_CALIBRATION,
    TRAIN_DEV,
    TRAIN_FIT,
    freeze_split,
    load_split_manifest,
    write_split_manifest,
)
from pes2ts_core.pesn.offline.splits.grouping import (
    build_groups,
    cross_split_conflicts,
    load_canonical_members,
)
from pes2ts_core.utils.jsonio import write_json


def _write_assignment(tmp_path: Path, assignment: dict[str, str]) -> Path:
    interim = tmp_path / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "reaction_id": sorted(assignment),
            "split": [assignment[rid] for rid in sorted(assignment)],
        }
    )
    pq.write_table(table, interim / "split_assignment.parquet")
    return interim


def _write_manifests(tmp_path: Path) -> Path:
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_json(manifests / "split_manifest.json", {"schema_version": "g0_manifest_v1"})
    return manifests


def _write_duplicate_groups(interim: Path, groups: list[dict]) -> None:
    write_json(
        interim / "duplicate_groups.json",
        {
            "schema_version": "g0_manifest_v1",
            "dataset_version": "fixture",
            "groups": groups,
        },
    )


NAMES = [f"RXN_{i:010d}" for i in range(1, 41)]


def _base_assignment() -> dict[str, str]:
    assignment = {}
    for index, name in enumerate(NAMES):
        assignment[name] = "train" if index < 30 else "valid"
    return assignment


def test_grouping_merges_duplicate_members() -> None:
    interim = _write_assignment(Path("/tmp/opencode/pesn_split_fixture_1"), _base_assignment())
    _write_duplicate_groups(
        interim,
        [{"identity": "id1", "member_reaction_ids": [NAMES[0], NAMES[1]], "canonical_reaction_id": NAMES[0]}],
    )
    members = load_canonical_members(interim)
    assert members[NAMES[0]] == NAMES[0] and members[NAMES[1]] == NAMES[0]
    groups = build_groups(NAMES[:5], members)
    assert groups[NAMES[0]] == [NAMES[0], NAMES[1]]


def test_freeze_produces_disjoint_train_side_subsets(tmp_path: Path) -> None:
    assignment = _base_assignment()
    interim = _write_assignment(tmp_path, assignment)
    manifests = _write_manifests(tmp_path)
    _write_duplicate_groups(interim, [])
    family = {name: f"fam_{index % 4}" for index, name in enumerate(NAMES)}

    manifest = freeze_split(interim, manifests, seed=42, family_by_reaction=family)

    subsets = manifest["subsets"]
    fit = set(subsets[TRAIN_FIT])
    dev = set(subsets[TRAIN_DEV])
    calibration = set(subsets[TRAIN_CALIBRATION])
    assert not (fit & dev) and not (fit & calibration) and not (dev & calibration)
    assert (fit | dev | calibration) <= {n for n, s in assignment.items() if s == "train"}
    holdout = set(manifest["family_holdout"]["reaction_ids"])
    assert not (holdout & (fit | dev | calibration))
    assert manifest["frozen"] is True
    assert manifest["known_group_cross_split_conflicts"] == 0


def test_freeze_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    interim = _write_assignment(tmp_path, _base_assignment())
    manifests = _write_manifests(tmp_path)
    _write_duplicate_groups(interim, [])
    family = {name: f"fam_{index % 5}" for index, name in enumerate(NAMES)}
    first = freeze_split(interim, manifests, seed=7, family_by_reaction=family)
    second = freeze_split(interim, manifests, seed=7, family_by_reaction=family)
    assert first["subsets"] == second["subsets"]
    assert first["family_holdout"]["reaction_ids"] == second["family_holdout"]["reaction_ids"]


def test_whole_duplicate_group_stays_together_and_conflicts_isolate(tmp_path: Path) -> None:
    assignment = _base_assignment()
    # NAMES[0] is train, NAMES[1] forced to valid -> one group spans splits.
    assignment[NAMES[1]] = "valid"
    interim = _write_assignment(tmp_path, assignment)
    manifests = _write_manifests(tmp_path)
    _write_duplicate_groups(
        interim,
        [
            {
                "identity": "cross",
                "member_reaction_ids": [NAMES[0], NAMES[1]],
                "canonical_reaction_id": NAMES[0],
            },
            {
                "identity": "inside",
                "member_reaction_ids": [NAMES[2], NAMES[3]],
                "canonical_reaction_id": NAMES[2],
            },
        ],
    )
    family = {}
    manifest = freeze_split(interim, manifests, seed=42, family_by_reaction=family)
    all_subsets = set()
    for subset in manifest["subsets"].values():
        all_subsets.update(subset)
    assert NAMES[0] not in all_subsets and NAMES[1] not in all_subsets
    assert manifest["isolated_conflicts"], "cross-split group must be isolated with a record"
    groups = build_groups(sorted(assignment), load_canonical_members(interim))
    assert len(cross_split_conflicts(groups, assignment)) == 1
    # the same-side duplicate group stays together in one subset
    placed = [
        name
        for name in (NAMES[2], NAMES[3])
        if name in set(manifest["subsets"][TRAIN_FIT])
    ]
    assert len(placed) in (0, 2)


def test_development_exposure_groups_never_enter_subsets(tmp_path: Path) -> None:
    interim = _write_assignment(tmp_path, _base_assignment())
    manifests = _write_manifests(tmp_path)
    _write_duplicate_groups(interim, [])
    family = {}
    manifest = freeze_split(
        interim,
        manifests,
        seed=42,
        family_by_reaction=family,
        development_exposure_ids={NAMES[5]},
    )
    exposure = set(manifest["development_exposure"]["reaction_ids"])
    assert NAMES[5] in exposure
    all_subsets = set()
    for subset in manifest["subsets"].values():
        all_subsets.update(subset)
    assert not (exposure & all_subsets)


def test_write_and_reload_roundtrip(tmp_path: Path) -> None:
    interim = _write_assignment(tmp_path, _base_assignment())
    manifests = _write_manifests(tmp_path)
    _write_duplicate_groups(interim, [])
    manifest = freeze_split(
        interim, manifests, seed=42, family_by_reaction={NAMES[0]: "famx"}
    )
    path = write_split_manifest(manifest, tmp_path / "out" / "splits")
    reloaded = load_split_manifest(path.parent)
    assert reloaded["subsets"] == manifest["subsets"]


def test_invalid_fractions_refused(tmp_path: Path) -> None:
    interim = _write_assignment(tmp_path, _base_assignment())
    manifests = _write_manifests(tmp_path)
    _write_duplicate_groups(interim, [])
    with pytest.raises(SplitFreezeError):
        freeze_split(
            interim,
            manifests,
            seed=42,
            family_by_reaction={},
            dev_fraction=0.9,
            calibration_fraction=0.9,
        )
