"""Input discovery for the split freeze stage.

:func:`load_freeze_inputs` assembles everything the freeze decision needs in
one pass: the completeness gate over the leak audit, the assignment and
inventory coverage state, the fingerprint table, a fresh full audit run that
also collects the COMPLETE offending-pair set (the manifest only stores the
capped best records), and the known cross-split duplicate pairs from the
duplicate ledger. The audit gate runs before the re-audit so a stale
``complete=false`` manifest is never silently repaired into a clean verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit.DataStructs import ExplicitBitVect

from pes2ts_core.g0.audit_metrics import load_known_cross_pairs
from pes2ts_core.g0.fingerprints import load_fingerprints, read_split_assignment
from pes2ts_core.g0.neardup import LEAK_AUDIT_FILENAME, cross_split_leak_audit
from pes2ts_core.g0.split_manifest import (
    SPLIT_HINT,
    assignment_coverage,
    read_manifest,
)
from pes2ts_core.g0.split_policy import (
    OffenseTuple,
    offending_tuples,
    require_complete_audit,
)
from pes2ts_core.g0.split_sources import read_inventory_ids
from pes2ts_core.utils.hashing import JSONValue

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FreezeInputs:
    """Everything the freeze decision and its remediations reason with."""

    previous: dict[str, JSONValue]
    strategy: str
    assignment: dict[str, str]
    inventory_ids: list[str]
    covered: list[str]
    split_only: list[str]
    n_inventory: int
    fingerprints: tuple[tuple[str, ExplicitBitVect], ...]
    known_pairs: set[frozenset[str]]
    near_pairs: tuple[tuple[str, str], ...]
    offending: tuple[OffenseTuple, ...]
    n_pairs: int
    n_known: int
    max_similarity: float

    @property
    def leak(self) -> bool:
        """Return whether the fresh audit found any leakage."""
        return self.n_pairs > 0 or self.n_known > 0


def load_freeze_inputs(
    config: Mapping[str, Any], previous: Mapping[str, JSONValue]
) -> FreezeInputs:
    """Load and audit every input of one freeze decision.

    Raises
    ------
    FileNotFoundError
        When the audit, assignment, inventory, or fingerprint table is missing.
    AuditIncompleteError
        When the stored audit is not ``complete=true``.
    ValueError
        When a stored artifact is malformed or the manifest lacks a strategy.
    """
    manifests_dir = Path(config["paths"]["manifests"])
    audit_path = manifests_dir / LEAK_AUDIT_FILENAME
    _ = require_complete_audit(audit_path)

    strategy = previous.get("strategy")
    if not isinstance(strategy, str):
        msg = f"Split manifest lacks a strategy; {SPLIT_HINT}"
        raise ValueError(msg)
    assignment = read_split_assignment(config)
    _inventory_path, inventory_ids = read_inventory_ids(config)
    covered, split_only, n_inventory = assignment_coverage(assignment, inventory_ids)
    max_pairs = int(config["near_dup"]["max_pairs"])
    fingerprints = load_fingerprints(config)
    vectors: dict[str, ExplicitBitVect] = dict(fingerprints)

    rerun = cross_split_leak_audit(config, fingerprints, collect_all_pairs=True)
    if len(rerun.all_offending_pairs) != rerun.n_pairs_over_threshold:
        msg = (
            f"Leak audit collected {len(rerun.all_offending_pairs)} pair(s) but "
            f"counted {rerun.n_pairs_over_threshold} offense(s)"
        )
        raise RuntimeError(msg)
    audit_document = read_manifest(audit_path)
    known_pairs = load_known_cross_pairs(manifests_dir, assignment)
    offending = offending_tuples(
        audit_document, known_pairs, assignment, vectors, max_pairs
    )
    logger.info(
        "Split freeze inputs: covered=%d/%d near_pairs=%d "
        "known_cross_duplicates=%d max_similarity=%.4f",
        len(covered),
        n_inventory,
        rerun.n_pairs_over_threshold,
        len(known_pairs),
        rerun.max_similarity,
    )
    return FreezeInputs(
        previous=dict(previous),
        strategy=strategy,
        assignment=assignment,
        inventory_ids=inventory_ids,
        covered=covered,
        split_only=split_only,
        n_inventory=n_inventory,
        fingerprints=fingerprints,
        known_pairs=known_pairs,
        near_pairs=rerun.all_offending_pairs,
        offending=offending,
        n_pairs=rerun.n_pairs_over_threshold,
        n_known=len(known_pairs),
        max_similarity=rerun.max_similarity,
    )


__all__ = ["FreezeInputs", "load_freeze_inputs"]
