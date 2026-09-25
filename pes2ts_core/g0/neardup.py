"""Mandatory full near-duplicate cross-split leakage audit (DRFP + Tanimoto).

:func:`cross_split_leak_audit` compares EVERY valid/test probe reaction against
EVERY train reference reaction using folded DRFP fingerprints
(:func:`~pes2ts_core.g0.fingerprints.compute_fingerprints`, 0/1 strings decoded
back through :func:`rdkit.DataStructs.CreateFromBitString`) and blocked
:func:`rdkit.DataStructs.BulkTanimotoSimilarity` calls of at most
``near_dup.block_size`` references. The full pairwise matrix is never
materialized; only a running maximum, an exact over-threshold count, and the
best ``near_dup.max_pairs`` offense records are kept. There is no sampling
shortcut anywhere in this module: ``n_comparisons`` in the manifest is exactly
``n_probes * n_references``.

Budget and honesty
------------------
Elapsed time is checked with :func:`time.monotonic` before every block. When
``near_dup.max_seconds`` is exceeded the manifest is written with
``complete=false`` and a ``reason``, an ``AUDIT_BUDGET_EXCEEDED`` rejection is
appended to the unified ledger, and :class:`AuditBudgetExceeded` is raised. A
partial audit therefore can never be mistaken for a clean dataset; this module
never writes a leakage verdict (no ``leak_status``), that decision belongs to
the split freeze stage.

Known duplicates
----------------
Exact duplicates and written reverses already recorded in the duplicate ledger
are identity-equal by construction and would otherwise reappear as Tanimoto
1.0 offenses. Pairs whose members straddle train and valid/test are excluded
from ``n_pairs_over_threshold`` and ``top_offenses`` (no double counting with
the dedup stage) and reported separately as
``n_cross_split_known_duplicates``. ``max_similarity`` remains the true overall
maximum over all comparisons, excluded pairs included.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never

import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from pes2ts_core.g0.audit_metrics import (
    PROBE_SPLITS,
    REFERENCE_SPLIT,
    AuditProgress,
    OffenseCollector,
    load_known_cross_pairs,
)
from pes2ts_core.g0.fingerprints import (
    FingerprintResult,
    compute_fingerprints,
    load_fingerprints,
    read_split_assignment,
    resolve_fingerprint_table,
)
from pes2ts_core.g0.rejections import Rejection, RejectionCode, RejectionLedger
from pes2ts_core.utils.jsonio import write_json

logger = logging.getLogger(__name__)

#: Audit manifest filename written under ``config["paths"]["manifests"]``.
LEAK_AUDIT_FILENAME: Final[str] = "leak_audit.json"
#: Stage name stamped into every rejection emitted by this module.
AUDIT_STAGE: Final[str] = "cross_split_leak_audit"
#: Fingerprint method recorded in the manifest.
AUDIT_METHOD: Final[str] = "DRFP"
#: ``reason`` value of a budget-aborted manifest.
BUDGET_EXCEEDED_REASON: Final[str] = "budget exceeded"
#: CLI exit code for an audit that exceeded ``near_dup.max_seconds``.
EXIT_AUDIT_BUDGET_EXCEEDED: Final[int] = 6


class AuditBudgetExceeded(RuntimeError):
    """Raised when the audit exceeds ``near_dup.max_seconds`` before finishing.

    The manifest has already been written with ``complete=false`` and the
    rejection ledger already carries the ``AUDIT_BUDGET_EXCEEDED`` entry when
    this exception leaves the audit, so callers must never treat an aborted run
    as evidence of a clean dataset.
    """


@dataclass(frozen=True, slots=True)
class LeakAuditResult:
    """Outcome of one :func:`cross_split_leak_audit` run."""

    n_pairs_over_threshold: int
    max_similarity: float
    complete: bool
    manifest_path: Path


def cross_split_leak_audit(
    config: Mapping[str, Any],
    fingerprints: Sequence[tuple[str, ExplicitBitVect]] | None = None,
) -> LeakAuditResult:
    """Run the mandatory full cross-split near-duplicate audit.

    Probes are every valid/test reaction and references are EVERY train
    reaction in the split assignment; each probe is compared against the train
    references in blocks of ``near_dup.block_size`` (never the full pairwise
    matrix). Similarities at or above ``near_dup.threshold`` are counted
    exactly and the best ``near_dup.max_pairs`` are recorded in the manifest.
    Pairs already recorded in the duplicate ledger that straddle train and
    valid/test are excluded from the offense count and reported as
    ``n_cross_split_known_duplicates``.

    Parameters
    ----------
    config:
        Loaded configuration; ``paths.interim``, ``paths.manifests``, and the
        ``near_dup`` block are required.
    fingerprints:
        Optional in-memory table from
        :func:`~pes2ts_core.g0.fingerprints.compute_fingerprints`; when omitted
        the table is rebuilt from ``fingerprints.parquet``.

    Raises
    ------
    FileNotFoundError
        When the split assignment or fingerprint table is missing.
    ValueError
        When an input table is malformed or a split reaction has no fingerprint.
    AuditBudgetExceeded
        When ``near_dup.max_seconds`` is exceeded; the manifest (with
        ``complete=false``) and the ledger entry are written before raising.
    """
    near_dup = config["near_dup"]
    threshold = float(near_dup["threshold"])
    fp_size = int(near_dup["fp_size"])
    block_size = int(near_dup["block_size"])
    max_seconds = float(near_dup["max_seconds"])
    max_pairs = int(near_dup["max_pairs"])
    if block_size < 1:
        msg = f"near_dup.block_size must be >= 1, got {block_size}"
        raise ValueError(msg)
    if max_pairs < 0:
        msg = f"near_dup.max_pairs must be >= 0, got {max_pairs}"
        raise ValueError(msg)

    assignment = read_split_assignment(config)
    by_id = resolve_fingerprint_table(config, fingerprints)
    probe_ids = sorted(
        reaction_id
        for reaction_id, label in assignment.items()
        if label in PROBE_SPLITS
    )
    reference_ids = sorted(
        reaction_id
        for reaction_id, label in assignment.items()
        if label == REFERENCE_SPLIT
    )
    missing = [
        reaction_id for reaction_id in (*probe_ids, *reference_ids) if reaction_id not in by_id
    ]
    if missing:
        msg = f"Fingerprints missing for {len(missing)} split reaction(s), e.g. {missing[:10]}"
        raise ValueError(msg)
    reference_vectors = [by_id[reaction_id] for reaction_id in reference_ids]

    manifest_path = Path(config["paths"]["manifests"]) / LEAK_AUDIT_FILENAME
    known_cross_pairs = load_known_cross_pairs(manifest_path.parent, assignment)
    progress = AuditProgress(
        config=config,
        threshold=threshold,
        fp_size=fp_size,
        probe_count=len(probe_ids),
        reference_count=len(reference_ids),
        offenses=OffenseCollector(max_pairs),
        known_cross_duplicates=len(known_cross_pairs),
    )
    total_comparisons = progress.total_comparisons
    start = time.monotonic()

    def _abort(current_probe_id: str) -> Never:
        """Persist the incomplete manifest and ledger entry, then raise."""
        elapsed = time.monotonic() - start
        write_json(
            manifest_path,
            progress.manifest(
                complete=False, reason=BUDGET_EXCEEDED_REASON, duration=elapsed
            ),
        )
        ledger = RejectionLedger.load(manifest_path.parent)
        ledger.add(
            Rejection(
                reaction_id=current_probe_id,
                stage=AUDIT_STAGE,
                code=RejectionCode.AUDIT_BUDGET_EXCEEDED,
                detail=(
                    f"Leak audit exceeded near_dup.max_seconds={max_seconds} after "
                    f"{elapsed:.3f}s ({progress.probes_completed}/{len(probe_ids)} "
                    f"probes, {progress.comparisons_done}/{total_comparisons} comparisons)"
                ),
                source_pointer=str(manifest_path),
            )
        )
        ledger.write()
        message = (
            f"Leak audit budget exceeded after {elapsed:.3f}s (limit "
            f"{max_seconds}s, probe {current_probe_id}); manifest marked "
            f"complete=false and rejection recorded"
        )
        logger.error("%s", message)
        raise AuditBudgetExceeded(message)

    for probe_id in probe_ids:
        if time.monotonic() - start >= max_seconds:
            _abort(probe_id)
        probe_vector = by_id[probe_id]
        for block_start in range(0, len(reference_ids), block_size):
            if time.monotonic() - start >= max_seconds:
                _abort(probe_id)
            block_ids = reference_ids[block_start : block_start + block_size]
            block_vectors = reference_vectors[block_start : block_start + block_size]
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(probe_vector, block_vectors),
                dtype=np.float64,
            )
            progress.comparisons_done += len(block_ids)
            if similarities.size:
                progress.max_similarity = max(
                    progress.max_similarity, float(similarities.max())
                )
            for index in np.flatnonzero(similarities >= threshold).tolist():
                reference_id = block_ids[index]
                if frozenset((probe_id, reference_id)) in known_cross_pairs:
                    continue
                progress.offenses.observe(
                    float(similarities[index]), probe_id, reference_id
                )
        progress.probes_completed += 1

    duration = time.monotonic() - start
    write_json(
        manifest_path,
        progress.manifest(complete=True, reason=None, duration=duration),
    )
    logger.info(
        "Leak audit: probes=%d references=%d comparisons=%d "
        "pairs_over_threshold=%d known_cross_duplicates=%d max_similarity=%.4f -> %s",
        len(probe_ids),
        len(reference_ids),
        total_comparisons,
        progress.offenses.count,
        len(known_cross_pairs),
        progress.max_similarity,
        manifest_path,
    )
    return LeakAuditResult(
        n_pairs_over_threshold=progress.offenses.count,
        max_similarity=progress.max_similarity,
        complete=True,
        manifest_path=manifest_path,
    )


__all__ = [
    "AUDIT_METHOD",
    "AUDIT_STAGE",
    "BUDGET_EXCEEDED_REASON",
    "EXIT_AUDIT_BUDGET_EXCEEDED",
    "LEAK_AUDIT_FILENAME",
    "AuditBudgetExceeded",
    "FingerprintResult",
    "LeakAuditResult",
    "compute_fingerprints",
    "cross_split_leak_audit",
    "load_fingerprints",
    "read_split_assignment",
]
