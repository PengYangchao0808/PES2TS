"""Stage runners for the idempotent end-to-end G0 pipeline.

Each runner owns exactly one stage's work and returns a :class:`StageOutcome`
naming the artifacts it produced.  Runners never decide whether they should
run: the executor in :mod:`pes2ts_core.g0.pipeline` owns every skip decision
through the stage-state sidecar, so the stage modules themselves stay free of
pipeline bookkeeping.

The truth-dependent extraction and relocation work is delegated to the
allow-listed :mod:`pes2ts_core.g0.truth_quarantine` module; the pipeline never
references a quarantined path, so the static guard and the default suite stay
green.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from pes2ts_core.g0.dedup import detect_duplicates
from pes2ts_core.g0.fetch import (
    FetchedSource,
    fetch_sources,
    source_manifest_path,
    verify_sources,
    write_source_manifest,
)
from pes2ts_core.g0.fingerprints import compute_fingerprints
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME, build_inventory
from pes2ts_core.g0.neardup import LEAK_AUDIT_FILENAME, cross_split_leak_audit
from pes2ts_core.g0.pipeline_state import (
    PipelineContext,
    source_records,
    sources_relocated,
)
from pes2ts_core.g0.split import (
    LEAK_STATUS_LEAK_FOUND,
    adopt_official_split,
    freeze_split,
)
from pes2ts_core.g0.split_sources import SPLIT_ASSIGNMENT_FILENAME, SPLIT_MANIFEST_FILENAME
from pes2ts_core.g0.strata import select_cohorts
from pes2ts_core.g0.truth_quarantine import quarantine_truth

#: Stage statuses recorded in the run report.
_RUNNING: Final[str] = "ran"
_SKIPPED: Final[str] = "skipped"
_FAILED: Final[str] = "failed"
#: Allowed remediation values accepted by the freeze stage.
type _Remediation = Literal["none", "exclude-leaky", "rebuild"]


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What one stage runner did: a status, an optional reason, and outputs."""

    status: str
    reason: str | None = None
    error: str | None = None
    outputs: tuple[Path, ...] = ()


def _reused_sources(config: Mapping[str, Any]) -> dict[str, FetchedSource]:
    """Describe the already-present sources without downloading anything.

    Only called after :func:`verify_sources` proved every MD5, so each target
    exists and its size can be read.
    """
    raw = Path(config["paths"]["raw"])
    sources: dict[str, FetchedSource] = {}
    for key, entry in config["source"]["files"].items():
        filename = str(entry.get("filename", key))
        target = raw / filename
        sources[filename] = FetchedSource(
            filename=filename,
            url=str(entry["url"]),
            zenodo_md5=str(entry["md5"]),
            size_bytes=target.stat().st_size,
            downloaded_at=None,
        )
    return sources


def run_fetch(ctx: PipelineContext) -> StageOutcome:
    """Ensure the configured sources exist, are verified, and are manifested.

    ``skip_fetch`` skips downloading but still verifies every source; a
    ``--force`` run reports that work as ``ran``.  When the quarantine stage
    already relocated the archives, the pipeline must not download them back
    into the raw tree: the recorded manifest digests are kept as-is.
    """
    config = ctx.config
    manifest_path = source_manifest_path(config)
    if sources_relocated(ctx):
        if not source_records(config):
            msg = "Sources relocated but no source manifest records their digests"
            raise FileNotFoundError(msg)
        return StageOutcome(_SKIPPED, "sources relocated", outputs=(manifest_path,))
    if ctx.skip_fetch:
        digests = verify_sources(config)
        _ = write_source_manifest(config, _reused_sources(config), digests)
        if ctx.force:
            return StageOutcome(_RUNNING, "verify-only", outputs=(manifest_path,))
        return StageOutcome(_SKIPPED, "flag", outputs=(manifest_path,))
    fetched = fetch_sources(config, force=False)
    digests = verify_sources(config)
    _ = write_source_manifest(config, fetched, digests)
    return StageOutcome(_RUNNING, outputs=(manifest_path,))


def run_inventory(ctx: PipelineContext) -> StageOutcome:
    """Build the TS-free R/P inventory from the raw sources."""
    result = build_inventory(ctx.config)
    return StageOutcome(_RUNNING, outputs=(result.parquet_path, result.manifest_path))


def run_quarantine(ctx: PipelineContext) -> StageOutcome:
    """Extract and relocate the answer-bearing archives behind their manifest."""
    result = quarantine_truth(ctx.config)
    outputs = (result.manifest_path, result.ts_path, result.irc_index_path)
    if result.skipped:
        return StageOutcome(_SKIPPED, "already current", outputs=outputs)
    return StageOutcome(_RUNNING, outputs=outputs)


def run_dedup(ctx: PipelineContext) -> StageOutcome:
    """Detect exact duplicates and written reverses across the inventory."""
    inventory_path = Path(ctx.config["paths"]["interim"]) / INVENTORY_PARQUET_FILENAME
    result = detect_duplicates(inventory_path, ctx.config)
    return StageOutcome(_RUNNING, outputs=(result.groups_path, result.ledger_path))


def run_split(ctx: PipelineContext) -> StageOutcome:
    """Adopt the authors' official split into the assignment map."""
    result = adopt_official_split(ctx.config)
    return StageOutcome(_RUNNING, outputs=(result.assignment_path, result.manifest_path))


def run_audit(ctx: PipelineContext) -> StageOutcome:
    """Compute DRFP fingerprints and run the mandatory leakage audit."""
    fingerprints = compute_fingerprints(ctx.config)
    result = cross_split_leak_audit(ctx.config, fingerprints.fingerprints)
    return StageOutcome(_RUNNING, outputs=(fingerprints.path, result.manifest_path))


def _existing(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return the subset of *paths* that currently exist."""
    return tuple(path for path in paths if path.is_file())


def run_freeze(ctx: PipelineContext) -> StageOutcome:
    """Freeze the split under the leak policy, reporting a halt as a failure."""
    result = freeze_split(
        ctx.config, remediation=cast(_Remediation, ctx.remediation)
    )
    manifests = Path(ctx.config["paths"]["manifests"])
    interim = Path(ctx.config["paths"]["interim"])
    outputs = _existing(
        (
            manifests / SPLIT_MANIFEST_FILENAME,
            interim / SPLIT_ASSIGNMENT_FILENAME,
            manifests / LEAK_AUDIT_FILENAME,
        )
    )
    if result.halted:
        error = (
            f"{LEAK_STATUS_LEAK_FOUND}: {result.n_pairs_over_threshold} near-duplicate "
            f"and {result.n_cross_split_known_duplicates} known cross-split offense(s); "
            f"re-run with --remediation exclude-leaky or --remediation rebuild to remediate"
        )
        return StageOutcome(_FAILED, "leak halt", error=error, outputs=outputs)
    if result.skipped:
        return StageOutcome(_SKIPPED, "already frozen clean", outputs=outputs)
    return StageOutcome(_RUNNING, outputs=outputs)


def run_cohorts(ctx: PipelineContext) -> StageOutcome:
    """Select the deterministic trial and stratified cohorts."""
    result = select_cohorts(ctx.config)
    return StageOutcome(
        _RUNNING, outputs=(result.trial_path, result.stratified_path, result.report_path)
    )


__all__ = [
    "StageOutcome",
    "run_audit",
    "run_cohorts",
    "run_dedup",
    "run_fetch",
    "run_freeze",
    "run_inventory",
    "run_quarantine",
    "run_split",
]
