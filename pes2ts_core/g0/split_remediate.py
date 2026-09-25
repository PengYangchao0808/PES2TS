"""Candidate application for the two split leak remediations.

Both remediations work on a CANDIDATE assignment: it is written over the
assignment Parquet, the audit is re-run against it, and only a clean re-audit
commits the candidate. When the candidate still leaks — or the re-audit itself
fails — the pre-remediation assignment bytes and the pre-remediation audit
manifest are restored before :class:`RemediationFailedError` leaves the module,
so the split is never left in a mixed state. Ledger rejections are written only
after a clean re-audit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pes2ts_core.g0.audit_metrics import load_known_cross_pairs
from pes2ts_core.g0.neardup import LEAK_AUDIT_FILENAME, cross_split_leak_audit
from pes2ts_core.g0.rejections import Rejection, RejectionCode, RejectionLedger
from pes2ts_core.g0.split_freeze_inputs import FreezeInputs
from pes2ts_core.g0.split_manifest import (
    assignment_coverage,
    assignment_counts,
    restore_bytes,
    write_assignment,
)
from pes2ts_core.g0.split_policy import (
    FREEZE_STAGE,
    REBUILT_STRATEGY,
    REMEDIATION_EXCLUDE_LEAKY,
    REMEDIATION_REBUILD,
    RemediationFailedError,
    exclusion_detail,
    select_removals,
)
from pes2ts_core.g0.split_rebuild import rebuild_assignment
from pes2ts_core.g0.split_sources import SPLIT_ASSIGNMENT_FILENAME
from pes2ts_core.utils.hashing import JSONValue

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RemediationOutcome:
    """A committed remediation and the facts the final manifest needs."""

    strategy: str
    counts: dict[str, int]
    covered: list[str]
    split_only: list[str]
    excluded_ids: tuple[str, ...]
    remediation: dict[str, JSONValue]
    audit: dict[str, JSONValue]
    n_pairs: int
    n_known: int
    max_similarity: float


def apply_remediation(
    config: Mapping[str, Any],
    inputs: FreezeInputs,
    remediation: str,
) -> RemediationOutcome:
    """Apply *remediation* to a candidate assignment and commit it when clean.

    ``exclude-leaky`` drops the held-out member of every offending pair;
    ``rebuild`` replaces the assignment with the cluster-preserving Butina
    rebuild. A candidate that still leaks restores both artifacts and raises
    :class:`RemediationFailedError`; the caller owns the ``LEAK_FOUND``
    manifest.
    """
    manifests_dir = Path(config["paths"]["manifests"])
    assignment_path = Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    audit_path = manifests_dir / LEAK_AUDIT_FILENAME

    excluded_ids: tuple[str, ...] = ()
    rebuilt: dict[str, JSONValue] | None = None
    removals: dict[str, tuple[str, str]] = {}
    strategy = inputs.strategy
    if remediation == REMEDIATION_EXCLUDE_LEAKY:
        removals = select_removals(
            inputs.near_pairs, inputs.known_pairs, inputs.assignment
        )
        candidate = {
            reaction_id: label
            for reaction_id, label in inputs.assignment.items()
            if reaction_id not in removals
        }
        excluded_ids = tuple(sorted(removals))
    elif remediation == REMEDIATION_REBUILD:
        rebuild = rebuild_assignment(config, inputs.fingerprints)
        candidate = rebuild.assignment
        strategy = REBUILT_STRATEGY
        rebuilt = {
            "cluster_cutoff": rebuild.cluster_cutoff,
            "n_clusters": rebuild.n_clusters,
            "seed": rebuild.seed,
        }
    else:
        msg = f"Unsupported remediation {remediation!r}"
        raise ValueError(msg)
    if not candidate:
        msg = "Remediation removed every assigned reaction; refusing an empty split"
        raise RemediationFailedError(msg)

    old_assignment = assignment_path.read_bytes()
    old_audit = audit_path.read_bytes()
    write_assignment(assignment_path, candidate)
    try:
        reaudit = cross_split_leak_audit(config, inputs.fingerprints)
    except Exception:
        restore_bytes(assignment_path, old_assignment)
        restore_bytes(audit_path, old_audit)
        raise
    re_known = len(load_known_cross_pairs(manifests_dir, candidate))
    if reaudit.n_pairs_over_threshold > 0 or re_known > 0:
        restore_bytes(assignment_path, old_assignment)
        restore_bytes(audit_path, old_audit)
        msg = (
            f"Remediation {remediation!r} failed: the candidate split still has "
            f"{reaudit.n_pairs_over_threshold} near-duplicate pair(s) and "
            f"{re_known} known cross-split duplicate pair(s); the "
            "pre-remediation assignment was restored"
        )
        logger.error("%s", msg)
        raise RemediationFailedError(msg)

    if remediation == REMEDIATION_EXCLUDE_LEAKY:
        ledger = RejectionLedger.load(manifests_dir)
        for victim in excluded_ids:
            twin, kind = removals[victim]
            ledger.add(
                Rejection(
                    reaction_id=victim,
                    stage=FREEZE_STAGE,
                    code=RejectionCode.LEAK_EXCLUDED,
                    detail=exclusion_detail(victim, twin, kind),
                    source_pointer=f"{audit_path}:{victim}",
                )
            )
        ledger.write()

    covered, split_only, _ = assignment_coverage(candidate, inputs.inventory_ids)
    remediation_record: dict[str, JSONValue] = (
        {"applied": REMEDIATION_EXCLUDE_LEAKY, "excluded_ids": list(excluded_ids)}
        if remediation == REMEDIATION_EXCLUDE_LEAKY
        else {"applied": REMEDIATION_REBUILD, "rebuilt": rebuilt}
    )
    logger.info(
        "Remediation %s committed: covered=%d/%d excluded=%d re-audit "
        "pairs=%d known=%d",
        remediation,
        len(covered),
        inputs.n_inventory,
        len(excluded_ids),
        reaudit.n_pairs_over_threshold,
        re_known,
    )
    return RemediationOutcome(
        strategy=strategy,
        counts=assignment_counts(candidate),
        covered=covered,
        split_only=split_only,
        excluded_ids=excluded_ids,
        remediation=remediation_record,
        audit={
            "complete": True,
            "n_pairs_over_threshold": reaudit.n_pairs_over_threshold,
            "n_cross_split_known_duplicates": re_known,
            "max_similarity": reaudit.max_similarity,
        },
        n_pairs=reaudit.n_pairs_over_threshold,
        n_known=re_known,
        max_similarity=reaudit.max_similarity,
    )


__all__ = ["RemediationOutcome", "apply_remediation"]
