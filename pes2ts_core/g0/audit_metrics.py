"""Bounded bookkeeping for the cross-split near-duplicate audit.

While the audit scans probe/reference blocks it must keep two things in sync:
an EXACT count of every similarity at or above the threshold, and a bounded
list of the best ``near_dup.max_pairs`` offenses sorted by similarity
descending then reaction IDs ascending. :class:`OffenseCollector` owns both.
Its heap ordering is inverted so the least preferred offense is always the
heap root and can be replaced in O(log n) without ever storing the unbounded
pair list.

The module also loads the duplicate ledger's known train<->held-out pairs and
owns the audit manifest contract through :class:`AuditProgress`, so the
finished and the budget-aborted documents can never drift apart:
:mod:`pes2ts_core.g0.neardup` just calls :meth:`AuditProgress.manifest`.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.dedup import DUPLICATE_LEDGER_FILENAME
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.fingerprints import FP_RADIUS, FP_RINGS
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import read_json

logger = logging.getLogger(__name__)

#: Probe splits: the held-out reactions that must not leak from train.
PROBE_SPLITS: Final[frozenset[str]] = frozenset({"valid", "test"})
#: Reference split: every train reaction is compared against every probe.
REFERENCE_SPLIT: Final[str] = "train"


@dataclass(frozen=True, slots=True)
class Offense:
    """One over-threshold probe/reference pair.

    Ordering is inverted on purpose so the heap root is the LEAST preferred
    offense: a smaller similarity is worse, and among equal similarities larger
    reaction IDs are worse (the manifest sorts by similarity descending, then
    reaction IDs ascending). ``heapq`` then keeps exactly the best records.
    """

    similarity: float
    probe_reaction_id: str
    reference_reaction_id: str

    def __lt__(self, other: Offense) -> bool:
        """Return whether *self* is the less preferred (heap-root) offense."""
        if self.similarity != other.similarity:
            return self.similarity < other.similarity
        return (self.probe_reaction_id, self.reference_reaction_id) > (
            other.probe_reaction_id,
            other.reference_reaction_id,
        )


@final
class OffenseCollector:
    """Exact over-threshold count plus the best ``limit`` offense records.

    ``count`` increments once per :meth:`observe` call, so the caller controls
    exactly which pairs are offenses (already-recorded cross-split duplicates
    are excluded by simply not observing them). The stored records stay capped
    at ``limit`` regardless of how many offenses are observed.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(0, limit)
        self.count = 0
        self._heap: list[Offense] = []

    def observe(
        self, similarity: float, probe_reaction_id: str, reference_reaction_id: str
    ) -> None:
        """Count one offense and keep it when it belongs to the best records."""
        self.count += 1
        if self._limit == 0:
            return
        offense = Offense(similarity, probe_reaction_id, reference_reaction_id)
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, offense)
        elif self._heap[0] < offense:
            _ = heapq.heapreplace(self._heap, offense)

    def records(self) -> list[JSONValue]:
        """Return the stored offenses sorted by similarity descending, then IDs."""
        ordered = sorted(
            self._heap,
            key=lambda offense: (
                -offense.similarity,
                offense.probe_reaction_id,
                offense.reference_reaction_id,
            ),
        )
        records: list[JSONValue] = []
        for offense in ordered:
            records.append(
                {
                    "probe_reaction_id": offense.probe_reaction_id,
                    "reference_reaction_id": offense.reference_reaction_id,
                    "similarity": offense.similarity,
                }
            )
        return records


def load_known_cross_pairs(
    manifests_dir: Path, assignment: Mapping[str, str]
) -> set[frozenset[str]]:
    """Return the frozenset key of every recorded train<->held-out pair.

    The duplicate ledger's classified pairs are the dedup stage's authoritative
    record of identity-equal reactions. A missing ledger yields an empty set;
    a malformed one raises :class:`ValueError` rather than silently ignoring
    already-recorded duplicates. Pairs whose members are not both assigned to
    the split, or that do not straddle train and valid/test, do not apply.
    """
    ledger_path = Path(manifests_dir) / DUPLICATE_LEDGER_FILENAME
    if not ledger_path.is_file():
        return set()
    document = read_json(ledger_path)
    if not isinstance(document, dict):
        msg = f"Duplicate ledger {ledger_path} is not a JSON object"
        raise ValueError(msg)
    records = document.get("pairs")
    if not isinstance(records, list):
        msg = f"Duplicate ledger {ledger_path} has no 'pairs' list"
        raise ValueError(msg)
    known: set[frozenset[str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            msg = f"Duplicate ledger {ledger_path} pair {index} is not an object"
            raise ValueError(msg)
        first = record.get("canonical_reaction_id")
        second = record.get("other_reaction_id")
        if not isinstance(first, str) or not isinstance(second, str):
            msg = (
                f"Duplicate ledger {ledger_path} pair {index} lacks string "
                "reaction IDs"
            )
            raise ValueError(msg)
        first_split = assignment.get(first)
        second_split = assignment.get(second)
        if first_split is None or second_split is None:
            continue
        if (first_split == REFERENCE_SPLIT) == (second_split == REFERENCE_SPLIT):
            continue
        other_split = second_split if first_split == REFERENCE_SPLIT else first_split
        if other_split in PROBE_SPLITS:
            known.add(frozenset((first, second)))
    return known


@dataclass(slots=True)
class AuditProgress:
    """Mutable scan progress behind the single audit manifest contract.

    Both the finished and the budget-aborted manifest are produced by
    :meth:`manifest`, so a new manifest key (or a changed count) can never land
    in one outcome only. ``comparisons_done`` and ``probes_completed`` describe
    the partial scan and are recorded only on an aborted manifest; the offense
    count in ``offenses`` is always exact.
    """

    config: Mapping[str, object]
    threshold: float
    fp_size: int
    probe_count: int
    reference_count: int
    offenses: OffenseCollector
    known_cross_duplicates: int
    max_similarity: float = 0.0
    comparisons_done: int = 0
    probes_completed: int = 0

    @property
    def total_comparisons(self) -> int:
        """Return the exact planned number of probe x reference comparisons."""
        return self.probe_count * self.reference_count

    def manifest(
        self, *, complete: bool, reason: str | None, duration: float
    ) -> dict[str, JSONValue]:
        """Return the audit manifest for the current progress state."""
        params: dict[str, JSONValue] = {
            "fp_size": self.fp_size,
            "radius": FP_RADIUS,
            "rings": FP_RINGS,
            "threshold": self.threshold,
        }
        document: dict[str, JSONValue] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset_version": dataset_version(self.config),
            "method": "DRFP",
            "params": params,
            "complete": complete,
            "n_probes": self.probe_count,
            "n_references": self.reference_count,
            "n_comparisons": self.total_comparisons,
            "n_pairs_over_threshold": self.offenses.count,
            "max_similarity": self.max_similarity,
            "top_offenses": self.offenses.records(),
            "n_cross_split_known_duplicates": self.known_cross_duplicates,
            "duration_seconds": duration,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if not complete:
            document["reason"] = reason
            document["n_probes_completed"] = self.probes_completed
            document["n_comparisons_completed"] = self.comparisons_done
        return document


__all__ = [
    "AuditProgress",
    "Offense",
    "OffenseCollector",
    "PROBE_SPLITS",
    "REFERENCE_SPLIT",
    "load_known_cross_pairs",
]
