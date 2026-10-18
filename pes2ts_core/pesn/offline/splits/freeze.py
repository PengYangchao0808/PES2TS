"""Predictor-owned split freeze under the parent split contract (B1).

Derives ``train_fit`` / ``train_dev`` / ``train_calibration`` from the frozen
parent ``train`` split at canonical-group level, seals a deterministic
reaction-family holdout out of every training-side subset, and records
development-exposure groups as contamination.  The parent assignment, the
parent split manifest, and every legacy artifact stay untouched; all outputs
live in the predictor's own directory with their own identity.

Hard gate (G3.P.0): known group cross-split conflicts must be zero — any
conflicting group is isolated with a typed record, never silently moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

import pyarrow.parquet as pq

from pes2ts_core.pesn.contracts.artifacts import SPLIT_SCHEMA_VERSION, validate_split_manifest
from pes2ts_core.pesn.offline.splits.grouping import (
    GROUPING_VERSION,
    build_groups,
    cross_split_conflicts,
    groups_content_hash,
    load_canonical_members,
    sha256_text,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json, write_json

TRAIN_FIT: Final[str] = "train_fit"
TRAIN_DEV: Final[str] = "train_dev"
TRAIN_CALIBRATION: Final[str] = "train_calibration"
FAMILY_HOLDOUT: Final[str] = "family_holdout"
DEVELOPMENT_EXPOSURE: Final[str] = "development_exposure"
GROUP_CONFLICT: Final[str] = "group_cross_split_conflict"

DEFAULT_DEV_FRACTION: Final[float] = 0.10
DEFAULT_CALIBRATION_FRACTION: Final[float] = 0.10
DEFAULT_FAMILY_HOLDOUT_FRACTION: Final[float] = 0.05


class SplitFreezeError(RuntimeError):
    """The freeze cannot be produced from the given inputs (typed)."""


def _rank(key: str, seed: int) -> int:
    return int(sha256_text(f"{seed}:{key}")[:16], 16)


def load_parent_assignment(interim_dir: str | Path) -> tuple[dict[str, str], str]:
    """Return ``reaction_id -> split`` and the assignment file hash."""
    path = Path(interim_dir) / "split_assignment.parquet"
    if not path.is_file():
        raise SplitFreezeError(f"parent split assignment missing: {path}")
    assignment = {
        str(row["reaction_id"]): str(row["split"])
        for row in pq.read_table(path).to_pylist()
    }
    return assignment, sha256_file(path)


def load_parent_manifest_sha256(manifests_dir: str | Path) -> str:
    path = Path(manifests_dir) / "split_manifest.json"
    if not path.is_file():
        raise SplitFreezeError(f"parent split manifest missing: {path}")
    return sha256_file(path)


def select_family_holdout(
    groups: Mapping[str, list[str]],
    family_by_reaction: Mapping[str, str],
    *,
    seed: int,
    fraction: float,
) -> tuple[set[str], set[str]]:
    """Deterministically seal whole reaction families (group-level).

    Returns ``(holdout_group_ids, holdout_reaction_ids)``.  A group enters the
    holdout when the family owning its plurality of members is selected;
    ranking is by ``sha256(f"{seed}:{family_id}")`` so the selection is a pure
    function of seed + frozen family ids.
    """
    groups_by_family: dict[str, list[str]] = {}
    for group_id in sorted(groups):
        votes: dict[str, int] = {}
        for member in groups[group_id]:
            family = family_by_reaction.get(member)
            if family is not None:
                votes[family] = votes.get(family, 0) + 1
        if not votes:
            continue
        plurality = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        groups_by_family.setdefault(plurality, []).append(group_id)
    ranked_families = sorted(
        groups_by_family,
        key=lambda family_id: (_rank(family_id, seed), family_id),
    )
    total_groups = sum(len(members) for members in groups_by_family.values())
    target = int(round(total_groups * fraction))
    holdout_groups: set[str] = set()
    held = 0
    for family_id in ranked_families:
        if held >= target:
            break
        holdout_groups.update(groups_by_family[family_id])
        held += len(groups_by_family[family_id])
    holdout_reactions = {
        member for gid in holdout_groups for member in groups[gid]
    }
    return holdout_groups, holdout_reactions


def freeze_split(
    interim_dir: str | Path,
    manifests_dir: str | Path,
    *,
    seed: int,
    family_by_reaction: Mapping[str, str],
    development_exposure_ids: set[str] | None = None,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    family_holdout_fraction: float = DEFAULT_FAMILY_HOLDOUT_FRACTION,
) -> dict[str, JSONValue]:
    """Build and validate the predictor split manifest (in memory)."""
    if dev_fraction <= 0 or calibration_fraction <= 0 or dev_fraction + calibration_fraction >= 1:
        raise SplitFreezeError(
            f"invalid fractions: dev={dev_fraction} calibration={calibration_fraction}"
        )
    assignment, assignment_sha = load_parent_assignment(interim_dir)
    parent_manifest_sha = load_parent_manifest_sha256(manifests_dir)
    canonical_members = load_canonical_members(interim_dir)

    parent_train = {rid for rid, split in assignment.items() if split == "train"}
    groups = build_groups(sorted(assignment), canonical_members)
    groups_sha = groups_content_hash(groups)

    conflicts = cross_split_conflicts(groups, assignment)
    conflict_groups = {entry["group_id"] for entry in conflicts}

    holdout_groups, holdout_reactions = select_family_holdout(
        groups,
        family_by_reaction,
        seed=seed,
        fraction=family_holdout_fraction,
    )

    exposure_ids = development_exposure_ids or set()
    exposure_groups = {
        group_id
        for group_id, members in groups.items()
        if any(member in exposure_ids for member in members)
    }

    eligible_groups = [
        group_id
        for group_id, members in sorted(groups.items())
        if group_id not in conflict_groups
        and group_id not in holdout_groups
        and group_id not in exposure_groups
        and any(member in parent_train for member in members)
    ]
    ranked = sorted(eligible_groups, key=lambda gid: (_rank(gid, seed), gid))
    n_dev = max(1, int(round(len(ranked) * dev_fraction)))
    n_calibration = max(1, int(round(len(ranked) * calibration_fraction)))
    dev_groups = set(ranked[:n_dev])
    calibration_groups = set(ranked[n_dev : n_dev + n_calibration])
    fit_groups = set(ranked[n_dev + n_calibration :])

    def members_of(group_ids: set[str]) -> set[str]:
        return {
            member
            for group_id in group_ids
            for member in groups.get(group_id, ())
        }

    fit_members = members_of(fit_groups)
    dev_members = members_of(dev_groups)
    calibration_members = members_of(calibration_groups)

    overlap_checks = [
        ("fit_dev", fit_members & dev_members),
        ("fit_calibration", fit_members & calibration_members),
        ("dev_calibration", dev_members & calibration_members),
        ("fit_holdout", fit_members & holdout_reactions),
        ("dev_holdout", dev_members & holdout_reactions),
        ("calibration_holdout", calibration_members & holdout_reactions),
    ]
    for label, overlap in overlap_checks:
        if overlap:
            raise SplitFreezeError(f"subset overlap in {label}: {sorted(overlap)[:5]}")
    leaked = (fit_members | dev_members | calibration_members) - parent_train
    if leaked:
        raise SplitFreezeError(
            f"subsets contain non-parent-train reactions: {sorted(leaked)[:5]}"
        )

    manifest: dict[str, JSONValue] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "parent_split_sha256": parent_manifest_sha,
        "parent_assignment_sha256": assignment_sha,
        "grouping_version": GROUPING_VERSION,
        "group_members_sha256": groups_sha,
        "seed": seed,
        "fractions": {
            "dev": dev_fraction,
            "calibration": calibration_fraction,
            "family_holdout": family_holdout_fraction,
        },
        "subsets": {
            TRAIN_FIT: sorted(fit_members),
            TRAIN_DEV: sorted(dev_members),
            TRAIN_CALIBRATION: sorted(calibration_members),
        },
        "subset_group_counts": {
            TRAIN_FIT: len(fit_groups),
            TRAIN_DEV: len(dev_groups),
            TRAIN_CALIBRATION: len(calibration_groups),
        },
        "family_holdout": {
            "group_ids": sorted(holdout_groups),
            "reaction_ids": sorted(holdout_reactions),
            "n_reactions": len(holdout_reactions),
        },
        "development_exposure": {
            "group_ids": sorted(exposure_groups),
            "reaction_ids": sorted(members_of(exposure_groups)),
            "contamination": True,
        },
        "isolated_conflicts": conflicts,
        # Conflicting groups are isolated out of every subset before the
        # subsets are formed, so the released subsets carry zero known
        # cross-split group conflicts; the isolation ledger above keeps the
        # full typed record.
        "known_group_cross_split_conflicts": 0,
        "frozen": True,
    }
    validate_split_manifest(manifest)
    return manifest


def write_split_manifest(manifest: Mapping[str, JSONValue], output_dir: str | Path) -> Path:
    """Write the frozen manifest under the predictor's own splits directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pesn_irc_split.json"
    write_json(path, dict(manifest))
    return path


def load_split_manifest(output_dir: str | Path) -> dict[str, JSONValue]:
    path = Path(output_dir) / "pesn_irc_split.json"
    if not path.is_file():
        raise SplitFreezeError(f"predictor split manifest missing: {path}")
    document = read_json(path)
    validate_split_manifest(document)
    return document


__all__ = [
    "DEVELOPMENT_EXPOSURE",
    "FAMILY_HOLDOUT",
    "GROUP_CONFLICT",
    "TRAIN_CALIBRATION",
    "TRAIN_DEV",
    "TRAIN_FIT",
    "SplitFreezeError",
    "freeze_split",
    "load_parent_assignment",
    "load_split_manifest",
    "write_split_manifest",
]
