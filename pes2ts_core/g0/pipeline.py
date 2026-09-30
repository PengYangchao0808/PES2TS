"""Idempotent end-to-end G0 pipeline and its run report.

:func:`run_pipeline` runs the eight G0 stages in the plan's fixed order and
owns every skip decision through the ``g0_stage_state.json`` sidecar (see
:mod:`pes2ts_core.g0.pipeline_state`).  The order is physical: inventory reads
the combined archive from the raw tree before quarantine relocates it, so
re-runs skip through recorded digests.  A stage failure stops the run and the
report names the failing stage, its error, and a traceback tail.  Truth work
goes only through the allow-listed :mod:`pes2ts_core.g0.truth_quarantine`; this
module never references a quarantined path.
"""
from __future__ import annotations

import copy
import logging
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Final

from pes2ts_core.g0.pipeline_report import (
    RUN_REPORT_FILENAME,
    build_document,
    current_leak_status,
    run_report_path,
)
from pes2ts_core.g0.pipeline_stages import (
    StageOutcome,
    run_audit,
    run_cohorts,
    run_dedup,
    run_fetch,
    run_freeze,
    run_inventory,
    run_quarantine,
    run_split,
)
from pes2ts_core.g0.pipeline_state import (
    PipelineContext,
    load_stage_state,
    outputs_with_digests,
    record_stage,
    recorded_outputs,
    skip_reason,
    stage_digest,
)
from pes2ts_core.g0.split_policy import REMEDIATION_CHOICES, REMEDIATION_NONE
from pes2ts_core.utils.jsonio import write_json

logger = logging.getLogger(__name__)

#: The plan's fixed stage order; later stages depend on earlier artifacts.
STAGE_ORDER: Final[tuple[str, ...]] = (
    "fetch",
    "inventory",
    "quarantine",
    "dedup",
    "split",
    "audit",
    "freeze",
    "cohorts",
)
#: Exit code returned by the CLI when any stage fails or the run halts.
EXIT_PIPELINE_FAILED: Final[int] = 21
#: Stage status values recorded in the run report.
STATUS_RAN: Final[str] = "ran"
STATUS_SKIPPED: Final[str] = "skipped"
STATUS_FAILED: Final[str] = "failed"

_STAGE_RUNNERS: Final[dict[str, Callable[[PipelineContext], StageOutcome]]] = {
    "fetch": run_fetch, "inventory": run_inventory, "quarantine": run_quarantine,
    "dedup": run_dedup, "split": run_split, "audit": run_audit,
    "freeze": run_freeze, "cohorts": run_cohorts,
}


@dataclass(frozen=True, slots=True)
class StageStatus:
    """Outcome of one pipeline stage."""

    name: str
    status: str
    duration_seconds: float
    reason: str | None = None
    error: str | None = None
    traceback_tail: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def n_outputs(self) -> int:
        """Return the number of artifacts the stage produced."""
        return len(self.outputs)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-ready record of this stage status."""
        record: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "n_outputs": self.n_outputs,
            "outputs": dict(self.outputs),
        }
        if self.reason is not None:
            record["reason"] = self.reason
        if self.error is not None:
            record["error"] = self.error
        if self.traceback_tail is not None:
            record["traceback_tail"] = self.traceback_tail
        return record


@dataclass(frozen=True, slots=True)
class RunReport:
    """Result of one :func:`run_pipeline` invocation."""

    ok: bool
    stages: list[StageStatus]
    leak_status: str | None
    report_path: Path


def _execute_stage(name: str, ctx: PipelineContext) -> StageStatus:
    """Run, skip, or fail stage *name* and return its status record."""
    started = time.monotonic()
    digest = stage_digest(name, ctx.config)
    reason = skip_reason(ctx, name, digest)
    if reason is not None:
        return StageStatus(
            name=name,
            status=STATUS_SKIPPED,
            duration_seconds=time.monotonic() - started,
            reason=reason,
            outputs=recorded_outputs(ctx, name),
        )
    try:
        outcome = _STAGE_RUNNERS[name](ctx)
    except Exception as exc:  # noqa: BLE001 - every stage failure is reported
        traceback_text = traceback.format_exc().strip()
        return StageStatus(
            name=name,
            status=STATUS_FAILED,
            duration_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback_tail="\n".join(traceback_text.splitlines()[-15:]),
        )
    outputs = outputs_with_digests(outcome.outputs)
    if outcome.status == STATUS_FAILED:
        return StageStatus(
            name=name,
            status=STATUS_FAILED,
            duration_seconds=time.monotonic() - started,
            reason=outcome.reason,
            error=outcome.error,
            outputs=outputs,
        )
    try:
        recorded_digest = stage_digest(name, ctx.config)
    except (OSError, ValueError) as exc:
        logger.warning("Could not recompute the %s input digest: %s", name, exc)
        recorded_digest = digest
    record_stage(ctx, name, recorded_digest, outputs)
    return StageStatus(
        name=name,
        status=outcome.status,
        duration_seconds=time.monotonic() - started,
        reason=outcome.reason,
        outputs=outputs,
    )


def run_pipeline(
    config: Mapping[str, Any],
    *,
    skip_fetch: bool = False,
    force: bool = False,
    remediation: str = REMEDIATION_NONE,
) -> RunReport:
    """Run every G0 stage in order and write ``g0_run_report.json``.

    *config* is deep-copied.  *skip_fetch* verifies sources instead of
    downloading; *force* ignores recorded digests and re-runs every stage
    (fetch still reuses verified local files); *remediation* goes to the
    freeze stage, where ``none`` halts on ``LEAK_FOUND``.  Returns a report
    whose ``ok`` is true only when every stage completed.
    """
    if remediation not in REMEDIATION_CHOICES:
        msg = f"Unknown remediation {remediation!r}; expected one of {REMEDIATION_CHOICES}"
        raise ValueError(msg)
    working: dict[str, Any] = copy.deepcopy(dict(config))
    ctx = PipelineContext(
        config=working,
        state=load_stage_state(working),
        force=force,
        skip_fetch=skip_fetch,
        remediation=remediation,
    )
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    start = time.monotonic()
    statuses: list[StageStatus] = []
    for name in STAGE_ORDER:
        status = _execute_stage(name, ctx)
        statuses.append(status)
        if status.status == STATUS_FAILED:
            logger.error(
                "Pipeline stopped at stage %s: %s", name, status.error or status.reason
            )
            break
    duration = time.monotonic() - start
    leak_status = current_leak_status(working)
    document = build_document(
        working,
        [status.to_record() for status in statuses],
        started_at=started_at,
        duration_seconds=duration,
        leak_status=leak_status,
    )
    report_path = run_report_path(working)
    write_json(report_path, document)
    ok = len(statuses) == len(STAGE_ORDER) and all(
        status.status != STATUS_FAILED for status in statuses
    )
    logger.log(
        logging.INFO if ok else logging.ERROR,
        "Pipeline %s: %d stage(s), leak_status=%s -> %s",
        "complete" if ok else "incomplete",
        len(statuses),
        leak_status,
        report_path,
    )
    return RunReport(ok=ok, stages=statuses, leak_status=leak_status, report_path=report_path)


def with_data_root(config: Mapping[str, Any], data_root: str | Path) -> dict[str, Any]:
    """Return a deep copy of *config* with every derived path moved under *data_root*.

    Every ``paths.*`` value starting with the configured ``paths.data_root``
    gets that prefix replaced, so a fixture tree can be addressed with
    ``--data-root`` without editing the YAML.
    """
    updated: dict[str, Any] = copy.deepcopy(dict(config))
    paths = updated.get("paths")
    if not isinstance(paths, dict):
        msg = "Configuration has no `paths` mapping to re-root"
        raise ValueError(msg)
    old_root = paths.get("data_root")
    new_root = str(data_root).rstrip("/") or str(data_root)
    if isinstance(old_root, str) and old_root:
        windows_style = "\\" in old_root or bool(PureWindowsPath(old_root).drive)
        root_type = PureWindowsPath if windows_style else PurePosixPath
        prefix = root_type(old_root)
        new_prefix = root_type(new_root)
        for key, value in paths.items():
            if not isinstance(value, str):
                continue
            try:
                relative = root_type(value).relative_to(prefix)
            except ValueError:
                continue
            paths[key] = str(new_prefix / relative) if str(relative) != "." else str(new_prefix)
    paths["data_root"] = new_root
    return updated


__all__ = [
    "EXIT_PIPELINE_FAILED",
    "RUN_REPORT_FILENAME",
    "STAGE_ORDER",
    "STATUS_FAILED",
    "STATUS_RAN",
    "STATUS_SKIPPED",
    "RunReport",
    "StageStatus",
    "run_pipeline",
    "with_data_root",
]
