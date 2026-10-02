"""Execution backends for PES generation (ADR-0001).

Public surface of the execution seam. Backends register by ``method`` string
and implement :class:`ExecutionBackend`; each returns a canonical
:class:`TrajectoryRecord`.

Status (X0): the interface is frozen here; concrete backends are added in X1
(``XTB_PATH`` wrapping the existing G2 pipeline, proven by a byte-identical
equivalence certificate) and X4 (ORCA / NEB / continuation).
"""
from __future__ import annotations

from pes2ts_core.generation.execution.protocol import (
    CHEAP_METHODS,
    METHOD_CONTINUATION,
    METHOD_ORCA_CONSTRAINTS,
    METHOD_ORCA_NEB,
    METHOD_ORCA_SCAN,
    METHOD_XTB_PATH,
    METHODS,
    ExecutionBackend,
    validate_method,
)
from pes2ts_core.generation.execution.record import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    TRAJECTORY_RECORD_SCHEMA,
    TRAJECTORY_STATUSES,
    TrajectoryFrame,
    TrajectoryRecord,
)

__all__ = [
    "CHEAP_METHODS",
    "METHOD_CONTINUATION",
    "METHOD_ORCA_CONSTRAINTS",
    "METHOD_ORCA_NEB",
    "METHOD_ORCA_SCAN",
    "METHOD_XTB_PATH",
    "METHODS",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PARTIAL",
    "TRAJECTORY_RECORD_SCHEMA",
    "TRAJECTORY_STATUSES",
    "ExecutionBackend",
    "TrajectoryFrame",
    "TrajectoryRecord",
    "validate_method",
]
