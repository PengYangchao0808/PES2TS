"""Canonical reaction grouping from non-truth metadata (B1).

A canonical group merges forward/reverse duplicates (G0 ``duplicate_groups``)
and every reaction identity source into one leakage unit: all frames of all
members of a group share one split subset.  Group identity never depends on
path frame numbering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Iterable, Mapping

from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json

GROUPING_VERSION: Final[str] = "pesn_grouping_v1"


def sha256_text(text: str) -> str:
    """Hex sha256 of a UTF-8 string (local helper; utils stays untouched)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GroupingError(RuntimeError):
    """Duplicate-group ledger malformed."""


def load_canonical_members(interim_dir: str | Path) -> dict[str, str]:
    """Return ``reaction_id -> canonical_group_id`` from the G0 ledger.

    The canonical group id is the group's canonical reaction id; every member
    of a duplicate group maps to it, reactions in no group stay absent.
    """
    path = Path(interim_dir) / "duplicate_groups.json"
    if not path.is_file():
        raise GroupingError(f"duplicate-group ledger missing: {path}")
    document: JSONValue = read_json(path)
    groups = document.get("groups") if isinstance(document, dict) else None
    if not isinstance(groups, list):
        raise GroupingError(f"duplicate-group ledger malformed: {path}")
    members: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical_reaction_id")
        member_ids = group.get("member_reaction_ids")
        if not isinstance(canonical, str) or not isinstance(member_ids, list):
            continue
        for member in member_ids:
            if isinstance(member, str):
                members[member] = canonical
    return members


def build_groups(
    reaction_ids: Iterable[str],
    canonical_members: Mapping[str, str],
) -> dict[str, list[str]]:
    """Fold canonical membership into explicit groups over *reaction_ids*."""
    groups: dict[str, list[str]] = {}
    for reaction_id in sorted(reaction_ids):
        group_id = canonical_members.get(reaction_id, reaction_id)
        groups.setdefault(group_id, []).append(reaction_id)
    return groups


def groups_content_hash(groups: Mapping[str, list[str]]) -> str:
    payload = {
        group_id: sorted(members)
        for group_id, members in sorted(groups.items())
    }
    return sha256_text(stable_json_dumps(payload))


def cross_split_conflicts(
    groups: Mapping[str, list[str]],
    assignment: Mapping[str, str],
) -> list[dict[str, JSONValue]]:
    """Report every canonical group whose members span more than one split."""
    conflicts: list[dict[str, JSONValue]] = []
    for group_id in sorted(groups):
        splits = sorted(
            {
                assignment[member]
                for member in groups[group_id]
                if member in assignment
            }
        )
        if len(splits) > 1:
            conflicts.append(
                {
                    "group_id": group_id,
                    "splits": splits,
                    "members": sorted(groups[group_id]),
                }
            )
    return conflicts


__all__ = [
    "GROUPING_VERSION",
    "GroupingError",
    "build_groups",
    "cross_split_conflicts",
    "groups_content_hash",
    "load_canonical_members",
]
