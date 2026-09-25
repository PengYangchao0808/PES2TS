"""Leak decision-policy internals for the split freeze stage.

This module owns the pieces :mod:`pes2ts_core.g0.split_freeze` reasons with:
the completeness gate over ``leak_audit.json``, the normalized offending-tuple
records (near-duplicate and known cross-split duplicate), and the victim
selection that removes the held-out member of every offending pair. The policy
constants live here so the freeze stage and the CLI surface can never drift.

An audit with ``complete=false`` is never a verdict: :func:`require_complete_audit`
raises :class:`AuditIncompleteError` before any freeze logic can run. Offending
tuples keep only IDs and one similarity per pair (never the similarity matrix),
and the manifest-bound list stays capped by ``near_dup.max_pairs``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from pes2ts_core.g0.audit_metrics import PROBE_SPLITS, REFERENCE_SPLIT
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import read_json

logger = logging.getLogger(__name__)

#: Final verdict when no cross-split leakage was found.
LEAK_STATUS_CLEAN: Final[str] = "clean"
#: Final verdict when leakage was found and left unremediated (default halt).
LEAK_STATUS_LEAK_FOUND: Final[str] = "LEAK_FOUND"
#: Remediation value that never touches the assignment.
REMEDIATION_NONE: Final[str] = "none"
#: Remediation that drops the held-out members of offending pairs.
REMEDIATION_EXCLUDE_LEAKY: Final[str] = "exclude-leaky"
#: Remediation that rebuilds the whole split from Butina clusters.
REMEDIATION_REBUILD: Final[str] = "rebuild"
#: Allowed ``--remediation`` values, in CLI help order.
REMEDIATION_CHOICES: Final[tuple[str, ...]] = (
    REMEDIATION_NONE,
    REMEDIATION_EXCLUDE_LEAKY,
    REMEDIATION_REBUILD,
)
#: Strategy recorded in the manifest after a successful rebuild.
REBUILT_STRATEGY: Final[str] = "rebuilt_butina_v1"
#: Kind of an over-threshold DRFP near-duplicate pair.
KIND_NEAR_DUP: Final[str] = "near_dup"
#: Kind of a duplicate-ledger pair that straddles train and valid/test.
KIND_KNOWN_DUPLICATE: Final[str] = "known_duplicate"
#: Stage name stamped into every rejection emitted by the freeze stage.
FREEZE_STAGE: Final[str] = "freeze_split"
#: Actionable hint attached to a missing leak-audit error.
AUDIT_HINT: Final[str] = "run `g0 audit` first"


class AuditIncompleteError(RuntimeError):
    """Raised when ``leak_audit.json`` is missing ``complete=true``.

    The freeze stage refuses to derive any verdict from a partial scan, so an
    interrupted audit can never be mistaken for a clean dataset.
    """


class RemediationFailedError(RuntimeError):
    """Raised when a remediation candidate still leaks after its re-audit.

    The pre-remediation assignment and audit manifest have already been
    restored when this leaves the stage, and the freeze manifest records
    ``LEAK_FOUND``; the split is never left in a mixed state.
    """


@dataclass(frozen=True, slots=True)
class OffenseTuple:
    """One offending probe/reference pair recorded in the frozen manifest."""

    similarity: float
    probe_reaction_id: str
    reference_reaction_id: str
    kind: str

    def to_record(self) -> dict[str, JSONValue]:
        """Return the JSON manifest record of this offense."""
        return {
            "probe_reaction_id": self.probe_reaction_id,
            "reference_reaction_id": self.reference_reaction_id,
            "similarity": self.similarity,
            "kind": self.kind,
        }


def require_complete_audit(path: Path) -> dict[str, JSONValue]:
    """Return the parsed audit manifest, refusing an incomplete scan.

    A missing file raises :class:`FileNotFoundError` with the ``g0 audit``
    hint; a non-object document or ``complete != true`` raises
    :class:`ValueError` / :class:`AuditIncompleteError` respectively.
    """
    if not Path(path).is_file():
        msg = f"Missing leak audit {path}; {AUDIT_HINT}"
        raise FileNotFoundError(msg)
    document = read_json(path)
    if not isinstance(document, dict):
        msg = f"Leak audit {path} is not a JSON object"
        raise ValueError(msg)
    if document.get("complete") is not True:
        msg = (
            f"Leak audit {path} is not complete "
            f"(complete={document.get('complete')!r}); an incomplete audit "
            "can never be frozen"
        )
        raise AuditIncompleteError(msg)
    return document


def _near_duplicate_tuples(
    audit_document: Mapping[str, JSONValue],
) -> list[OffenseTuple]:
    """Return the capped near-duplicate offenses recorded in the audit."""
    records = audit_document.get("top_offenses")
    if records is None:
        return []
    if not isinstance(records, list):
        msg = "leak_audit.json top_offenses is not a list"
        raise ValueError(msg)
    tuples: list[OffenseTuple] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            msg = f"leak_audit.json top_offenses[{index}] is not an object"
            raise ValueError(msg)
        probe = record.get("probe_reaction_id")
        reference = record.get("reference_reaction_id")
        similarity = record.get("similarity")
        if (
            not isinstance(probe, str)
            or not isinstance(reference, str)
            or not isinstance(similarity, (int, float))
        ):
            msg = f"leak_audit.json top_offenses[{index}] is malformed"
            raise ValueError(msg)
        tuples.append(
            OffenseTuple(float(similarity), probe, reference, KIND_NEAR_DUP)
        )
    return tuples


def _known_duplicate_tuples(
    known_pairs: set[frozenset[str]],
    assignment: Mapping[str, str],
    vectors: Mapping[str, ExplicitBitVect],
    limit: int,
) -> list[OffenseTuple]:
    """Return the known cross-split duplicates with a measured similarity."""
    tuples: list[OffenseTuple] = []
    for pair in sorted(known_pairs, key=lambda members: tuple(sorted(members))):
        members = sorted(pair)
        probe_ids = [member for member in members if assignment.get(member) in PROBE_SPLITS]
        reference_ids = [
            member for member in members if assignment.get(member) == REFERENCE_SPLIT
        ]
        if not probe_ids or not reference_ids:
            continue
        probe_id, reference_id = probe_ids[0], reference_ids[0]
        similarity = float(
            DataStructs.BulkTanimotoSimilarity(
                vectors[probe_id], [vectors[reference_id]]
            )[0]
        )
        tuples.append(
            OffenseTuple(similarity, probe_id, reference_id, KIND_KNOWN_DUPLICATE)
        )
    tuples.sort(
        key=lambda entry: (
            -entry.similarity,
            entry.probe_reaction_id,
            entry.reference_reaction_id,
        )
    )
    return tuples[:limit]


def offending_tuples(
    audit_document: Mapping[str, JSONValue],
    known_pairs: set[frozenset[str]],
    assignment: Mapping[str, str],
    vectors: Mapping[str, ExplicitBitVect],
    limit: int,
) -> tuple[OffenseTuple, ...]:
    """Return the manifest's capped offending-tuple list.

    Near-duplicate offenses come first, then known cross-split duplicates; the
    combined list is truncated to *limit* (``near_dup.max_pairs``). Remediation
    itself uses the uncapped audit pair list, never this capped view.
    """
    if limit <= 0:
        return ()
    near = _near_duplicate_tuples(audit_document)
    known = _known_duplicate_tuples(known_pairs, assignment, vectors, limit)
    return tuple((near + known)[:limit])


def select_removals(
    near_pairs: Sequence[tuple[str, str]],
    known_pairs: set[frozenset[str]],
    assignment: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    """Return ``removed_id -> (twin_id, kind)`` for every offending pair.

    Every near-duplicate pair is a held-out probe against a train reference and
    every known duplicate pair straddles train and valid/test, so the held-out
    member is the victim and the train member is the twin. The defensive
    both-held-out branch removes the higher reaction ID and logs a warning.
    Pairs whose members are not both assigned do not apply.
    """
    removals: dict[str, tuple[str, str]] = {}

    def consider(members: Sequence[str], kind: str) -> None:
        probes = sorted(
            member for member in members if assignment.get(member) in PROBE_SPLITS
        )
        references = sorted(
            member for member in members if assignment.get(member) == REFERENCE_SPLIT
        )
        if probes and references:
            victim, twin = probes[0], references[0]
        elif len(probes) >= 2:
            victim, twin = probes[-1], probes[0]
            logger.warning(
                "Both members of a %s pair are held out (%s); excluding the "
                "higher reaction ID",
                kind,
                probes,
            )
        else:
            return
        removals.setdefault(victim, (twin, kind))

    for probe_id, reference_id in near_pairs:
        consider((probe_id, reference_id), KIND_NEAR_DUP)
    for pair in sorted(known_pairs, key=lambda members: tuple(sorted(members))):
        consider(sorted(pair), KIND_KNOWN_DUPLICATE)
    return removals


def exclusion_detail(victim: str, twin: str, kind: str) -> str:
    """Return the ledger detail naming the victim and its train twin."""
    if kind == KIND_NEAR_DUP:
        return (
            f"Reaction {victim} excluded from the split: DRFP near-duplicate "
            f"at/above the leak threshold of train reaction {twin}"
        )
    return (
        f"Reaction {victim} excluded from the split: known cross-split "
        f"duplicate of train reaction {twin}"
    )


__all__ = [
    "AUDIT_HINT",
    "AuditIncompleteError",
    "FREEZE_STAGE",
    "KIND_KNOWN_DUPLICATE",
    "KIND_NEAR_DUP",
    "LEAK_STATUS_CLEAN",
    "LEAK_STATUS_LEAK_FOUND",
    "OffenseTuple",
    "REBUILT_STRATEGY",
    "REMEDIATION_CHOICES",
    "REMEDIATION_EXCLUDE_LEAKY",
    "REMEDIATION_NONE",
    "REMEDIATION_REBUILD",
    "RemediationFailedError",
    "exclusion_detail",
    "offending_tuples",
    "require_complete_audit",
    "select_removals",
]
