"""Pure, RNG-free selection primitives for deterministic cohorts.

The record-level selection core in :mod:`pes2ts_core.g0.strata` delegates the
per-stratum bookkeeping here and re-exports these helpers, so the public
``pes2ts_core.g0.strata`` entry point keeps exposing the exact rules G1 must
reuse.  Nothing in this module touches the filesystem: every function is a
pure transformation of records, stratum keys, and seed text.

Selection is deterministic by construction.  Within a stratum, members are
ranked by ``sha256(f"{seed_text}:{reaction_id}")`` (ties by ``reaction_id``),
and each stratum receives a largest-remainder quota over exact rational shares
(:class:`fractions.Fraction`), so no RNG is ever involved.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.strata_preview import StratumKey
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes


def strata_index(records: Sequence[Mapping[str, Any]]) -> dict[StratumKey, list[str]]:
    """Group reaction ids by stratum key in first-seen order."""
    grouped: dict[StratumKey, list[str]] = {}
    for record in records:
        key: StratumKey = (
            str(record["element_set"]),
            int(record["n_components"]),
            str(record["heavy_atom_bucket"]),
        )
        grouped.setdefault(key, []).append(str(record["reaction_id"]))
    return grouped


def key_text(key: StratumKey) -> str:
    """Return the stable JSON text form of a stratum key (``|``-joined)."""
    return f"{key[0]}|{key[1]}|{key[2]}"


def selection_key(seed_text: str, reaction_id: str) -> str:
    """Return the deterministic ranking key of *reaction_id* under *seed_text*."""
    return sha256_bytes(f"{seed_text}:{reaction_id}".encode("utf-8"))


def largest_remainder(counts: Mapping[StratumKey, int], size: int) -> dict[StratumKey, int]:
    """Allocate *size* slots proportionally with the largest-remainder method.

    Exact quotas are rational, floors sum to at most *size*, leftover slots go
    to the largest fractional parts (ties by stratum-key sort), and the
    allocation is capped at the total pool.
    """
    total = sum(counts.values())
    if total == 0 or size <= 0:
        return dict.fromkeys(counts, 0)
    size = min(size, total)
    exact = {key: Fraction(count * size, total) for key, count in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    leftover = size - sum(quotas.values())
    ranked = sorted(exact, key=lambda key: (-(exact[key] - int(exact[key])), key))
    for key in ranked[:leftover]:
        quotas[key] += 1
    return quotas


def select_members(grouped: Mapping[StratumKey, list[str]], size: int, seed_text: str) -> list[str]:
    """Select *size* reaction ids, quota per stratum, hash-ranked within."""
    quotas = largest_remainder(
        {key: len(ids) for key, ids in grouped.items()}, size
    )
    selected: list[str] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key], key=lambda rid: (selection_key(seed_text, rid), rid)
        )
        selected.extend(ranked[: quotas[key]])
    return selected


def member_counts(
    grouped: Mapping[StratumKey, list[str]], members: Sequence[str]
) -> Counter[StratumKey]:
    """Count *members* per stratum key."""
    owner = {rid: key for key, ids in grouped.items() for rid in ids}
    return Counter(owner[rid] for rid in members)


def cohort_document(
    cohort: str, version: str, members: Sequence[str], generated_at: str,
    grouped: Mapping[StratumKey, list[str]], counts: Mapping[StratumKey, int],
) -> dict[str, JSONValue]:
    """Build the JSON document of one cohort (all strata, zero counts included)."""
    strata: dict[str, JSONValue] = {
        key_text(key): counts.get(key, 0) for key in grouped
    }
    members_json: list[JSONValue] = list(sorted(members))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": version,
        "cohort": cohort,
        "size": len(members),
        "members": members_json,
        "strata": strata,
        "generated_at": generated_at,
    }


__all__ = [
    "cohort_document",
    "key_text",
    "largest_remainder",
    "member_counts",
    "select_members",
    "selection_key",
    "strata_index",
]
