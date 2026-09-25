"""Freeze the adopted split under an explicit leak decision policy.

This stage is the only writer of the final ``split_manifest.json``: it combines
the adopted assignment with the mandatory near-duplicate audit and enforces a
decision-complete policy. An incomplete audit can never be frozen; a leak
verdict halts by default with ``leak_status="LEAK_FOUND"`` and never touches
the assignment; ``exclude-leaky`` and ``rebuild`` are applied by
:mod:`pes2ts_core.g0.split_remediate`, which commits a candidate only after a
clean re-audit and otherwise restores the pre-remediation state. A clean
original audit ignores the remediation flag (``applied="none"`` plus a
``no-remediation-needed`` note). Manifest files from two identical runs differ
only in the keys listed in :data:`pes2ts_core.g0.VOLATILE_KEYS`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pes2ts_core.g0.split_freeze_inputs import FreezeInputs, load_freeze_inputs
from pes2ts_core.g0.split_manifest import (
    FreezeManifestState,
    assignment_counts,
    audit_summary,
    manifest_document,
    read_manifest,
    stored_counts,
    stored_int,
)
from pes2ts_core.g0.split_policy import (
    LEAK_STATUS_CLEAN,
    LEAK_STATUS_LEAK_FOUND,
    REMEDIATION_CHOICES,
    REMEDIATION_NONE,
    AuditIncompleteError,
    OffenseTuple,
    RemediationFailedError,
)
from pes2ts_core.g0.split_remediate import apply_remediation
from pes2ts_core.g0.split_sources import (
    SPLIT_ASSIGNMENT_FILENAME,
    SPLIT_MANIFEST_FILENAME,
)
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import write_json

logger = logging.getLogger(__name__)

#: CLI exit code for a halted (unremediated) leak verdict.
EXIT_LEAK_FOUND: Final[int] = 7
#: CLI exit code for an audit that is not ``complete=true``.
EXIT_AUDIT_INCOMPLETE: Final[int] = 8
#: CLI exit code for a remediation whose candidate still leaks.
EXIT_REMEDIATION_FAILED: Final[int] = 9


@dataclass(frozen=True, slots=True)
class FreezeResult:
    """Outcome of one :func:`freeze_split` run.

    ``skipped`` marks the idempotent re-run; ``halted`` the default leak halt.
    """

    leak_status: str
    halted: bool
    skipped: bool
    manifest_path: Path
    assignment_path: Path
    counts: dict[str, int]
    excluded_ids: tuple[str, ...] = ()
    n_pairs_over_threshold: int = 0
    n_cross_split_known_duplicates: int = 0
    max_similarity: float = 0.0


def _write_halted_manifest(
    config: Mapping[str, Any],
    inputs: FreezeInputs,
    remediation: dict[str, JSONValue],
) -> None:
    """Write the ``LEAK_FOUND`` manifest over the untouched assignment facts."""
    state = FreezeManifestState(
        strategy=inputs.strategy,
        counts=assignment_counts(inputs.assignment),
        covered=len(inputs.covered),
        n_inventory=inputs.n_inventory,
        n_not_in_split=inputs.n_inventory - len(inputs.covered),
        n_split_only=stored_int(
            inputs.previous, "n_split_only", len(inputs.split_only)
        ),
        leak_status=LEAK_STATUS_LEAK_FOUND,
        offending=inputs.offending,
        remediation=remediation,
        audit=audit_summary(inputs.n_pairs, inputs.n_known, inputs.max_similarity),
    )
    manifest_path = Path(config["paths"]["manifests"]) / SPLIT_MANIFEST_FILENAME
    write_json(manifest_path, manifest_document(config, inputs.previous, state))


def freeze_split(
    config: Mapping[str, Any],
    remediation: Literal["none", "exclude-leaky", "rebuild"] = "none",
) -> FreezeResult:
    """Freeze the split manifest under the leak decision policy.

    Reads the adoption manifest, rejects an already-frozen-clean repeat as an
    idempotent skip, then loads and re-audits every input, and finally freezes
    clean, halts with ``LEAK_FOUND``, or applies the requested remediation and
    freezes only after its clean re-audit.

    Raises
    ------
    FileNotFoundError
        When a required artifact (manifest, assignment, inventory, audit, or
        fingerprint table) is missing.
    AuditIncompleteError
        When ``leak_audit.json`` does not carry ``complete=true``.
    RemediationFailedError
        When a remediation candidate still leaks (state restored first).
    ValueError
        When the remediation name or a stored artifact is invalid.
    """
    if remediation not in REMEDIATION_CHOICES:
        msg = (
            f"Unknown remediation {remediation!r}; expected one of "
            f"{REMEDIATION_CHOICES}"
        )
        raise ValueError(msg)
    manifest_path = Path(config["paths"]["manifests"]) / SPLIT_MANIFEST_FILENAME
    assignment_path = Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    if not manifest_path.is_file():
        msg = f"Missing {manifest_path}; run `g0 split` first"
        raise FileNotFoundError(msg)
    previous = read_manifest(manifest_path)
    if previous.get("frozen") is True and previous.get("leak_status") == LEAK_STATUS_CLEAN:
        logger.info("Split manifest already frozen clean; skipping %s", manifest_path)
        return FreezeResult(
            leak_status=LEAK_STATUS_CLEAN,
            halted=False,
            skipped=True,
            manifest_path=manifest_path,
            assignment_path=assignment_path,
            counts=stored_counts(previous),
        )

    inputs = load_freeze_inputs(config, previous)
    logger.info("Split freeze: leak=%s remediation=%s", inputs.leak, remediation)
    if not inputs.leak:
        remediation_record: dict[str, JSONValue] = {"applied": REMEDIATION_NONE}
        if remediation != REMEDIATION_NONE:
            remediation_record["note"] = "no-remediation-needed"
        state = FreezeManifestState(
            strategy=inputs.strategy,
            counts=assignment_counts(inputs.assignment),
            covered=len(inputs.covered),
            n_inventory=inputs.n_inventory,
            n_not_in_split=inputs.n_inventory - len(inputs.covered),
            n_split_only=stored_int(
                previous, "n_split_only", len(inputs.split_only)
            ),
            leak_status=LEAK_STATUS_CLEAN,
            offending=(),
            remediation=remediation_record,
            audit=audit_summary(
                inputs.n_pairs, inputs.n_known, inputs.max_similarity
            ),
        )
        write_json(manifest_path, manifest_document(config, previous, state))
        logger.info("Split frozen clean (no leakage): %s", manifest_path)
        return FreezeResult(
            leak_status=LEAK_STATUS_CLEAN,
            halted=False,
            skipped=False,
            manifest_path=manifest_path,
            assignment_path=assignment_path,
            counts=assignment_counts(inputs.assignment),
            n_pairs_over_threshold=inputs.n_pairs,
            n_cross_split_known_duplicates=inputs.n_known,
            max_similarity=inputs.max_similarity,
        )

    if remediation == REMEDIATION_NONE:
        _write_halted_manifest(
            config,
            inputs,
            {"applied": REMEDIATION_NONE, "reason": "halt-no-flag"},
        )
        logger.error(
            "Split freeze halted: %d offending tuple(s) found and no "
            "remediation requested; %s left untouched",
            len(inputs.offending),
            assignment_path,
        )
        return FreezeResult(
            leak_status=LEAK_STATUS_LEAK_FOUND,
            halted=True,
            skipped=False,
            manifest_path=manifest_path,
            assignment_path=assignment_path,
            counts=assignment_counts(inputs.assignment),
            n_pairs_over_threshold=inputs.n_pairs,
            n_cross_split_known_duplicates=inputs.n_known,
            max_similarity=inputs.max_similarity,
        )

    try:
        outcome = apply_remediation(config, inputs, remediation)
    except RemediationFailedError:
        _write_halted_manifest(
            config,
            inputs,
            {
                "applied": remediation,
                "status": "failed",
                "reason": (
                    "re-audit still found leakage; pre-remediation assignment "
                    "restored"
                ),
            },
        )
        raise
    state = FreezeManifestState(
        strategy=outcome.strategy,
        counts=outcome.counts,
        covered=len(outcome.covered),
        n_inventory=inputs.n_inventory,
        n_not_in_split=(
            inputs.n_inventory - len(outcome.covered) - len(outcome.excluded_ids)
        ),
        n_split_only=stored_int(
            previous, "n_split_only", len(outcome.split_only)
        ),
        leak_status=LEAK_STATUS_CLEAN,
        offending=inputs.offending,
        remediation=outcome.remediation,
        audit=outcome.audit,
    )
    write_json(manifest_path, manifest_document(config, previous, state))
    logger.info(
        "Split frozen clean after %s: covered=%d/%d excluded=%d -> %s",
        remediation,
        len(outcome.covered),
        inputs.n_inventory,
        len(outcome.excluded_ids),
        manifest_path,
    )
    return FreezeResult(
        leak_status=LEAK_STATUS_CLEAN,
        halted=False,
        skipped=False,
        manifest_path=manifest_path,
        assignment_path=assignment_path,
        counts=outcome.counts,
        excluded_ids=outcome.excluded_ids,
        n_pairs_over_threshold=outcome.n_pairs,
        n_cross_split_known_duplicates=outcome.n_known,
        max_similarity=outcome.max_similarity,
    )


__all__ = [
    "EXIT_AUDIT_INCOMPLETE",
    "EXIT_LEAK_FOUND",
    "EXIT_REMEDIATION_FAILED",
    "AuditIncompleteError",
    "FreezeResult",
    "OffenseTuple",
    "RemediationFailedError",
    "freeze_split",
]
