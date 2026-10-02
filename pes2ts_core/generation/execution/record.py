"""Canonical trajectory record shared by every PES-generation backend.

Part of the execution seam frozen by ADR-0001 (workstream X0). It is a plain,
chemistry-free data record: backends (``XTB_PATH`` | ``ORCA_SCAN`` |
``ORCA_CONSTRAINTS`` | ``ORCA_NEB`` | ``CONTINUATION``) each project their
native output into this shape, and the result/quality plane projects it into
the v1 ``PathBundle`` / ``SeedProposal`` / ``ValidationResult`` objects.

Determinism note: this record carries no timestamp; provenance is explicit.
Adding a per-candidate ``method`` changes frozen-plan content hashes, so
existing frozen plans are never re-sealed (ADR-0001).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping

#: Schema tag for a :class:`TrajectoryRecord` document.
TRAJECTORY_RECORD_SCHEMA: Final[str] = "pes2ts_trajectory_record_v1"

#: Trajectory status vocabulary.
STATUS_COMPLETED: Final[str] = "completed"
STATUS_PARTIAL: Final[str] = "partial"
STATUS_FAILED: Final[str] = "failed"
TRAJECTORY_STATUSES: Final[tuple[str, ...]] = (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_FAILED)


@dataclass(frozen=True)
class TrajectoryFrame:
    """One frame of a computed path.

    ``targets_by_driver`` / ``actuals_by_driver`` / ``residuals_by_driver`` are
    keyed by driver id; a missing key means "not recorded for this driver",
    never a silent zero (ADR-0001 / typed-rejection discipline).
    """

    frame_index: int
    lambda_value: float | None = None
    energy_hartree: float | None = None
    geometry: tuple[tuple[float, float, float], ...] | None = None
    targets_by_driver: Mapping[str, float] = field(default_factory=dict)
    actuals_by_driver: Mapping[str, float] = field(default_factory=dict)
    residuals_by_driver: Mapping[str, float] = field(default_factory=dict)
    converged: bool | None = None
    incomplete: bool = False


@dataclass(frozen=True)
class TrajectoryRecord:
    """The single result record every generation backend returns.

    ``plan_sha256`` binds the record to the frozen plan it executed.
    ``provenance`` carries method/engine/argv/executable fingerprints (never
    truth-derived fields).
    """

    reaction_id: str
    method: str
    status: str
    frames: tuple[TrajectoryFrame, ...] = ()
    plan_sha256: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TRAJECTORY_RECORD_SCHEMA
