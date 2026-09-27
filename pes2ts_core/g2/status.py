"""G2 status, failure-code, and precedence contracts.

This module centralizes every constant shared by the G2 inspector, pipeline,
artifacts, and CLI, so a status name or failure code is defined exactly once.
The failure codes are members of the unified :class:`RejectionCode` enum (the
single legal definition site in :mod:`pes2ts_core.g0.rejections`); nothing
here reads data.
"""

from __future__ import annotations

from typing import Final

from pes2ts_core.g0.rejections import RejectionCode

#: Terminal status of a G2 path document: the produced path passed every
#: validity predicate.
STATUS_VALID: Final[str] = "valid"
#: Terminal status of a G2 path document: every attempt failed a predicate.
STATUS_FAILED: Final[str] = "failed"

#: Every G2 path-document status, in documentation order.
G2_STATUSES: Final[tuple[str, ...]] = (STATUS_VALID, STATUS_FAILED)

#: Every G2 failure code, in the deterministic ledger order of the plan.
G2_FAILURE_CODES: Final[tuple[RejectionCode, ...]] = (
    RejectionCode.G2_NOT_ELIGIBLE,
    RejectionCode.G2_MISSING_G1_DOC,
    RejectionCode.G2_G1_NOT_VALID,
    RejectionCode.G2_ENDPOINT_MISMATCH,
    RejectionCode.G2_ASSEMBLY_FAILED,
    RejectionCode.G2_ASSEMBLY_COLLISION,
    RejectionCode.G2_XTB_FAILED,
    RejectionCode.G2_ENDPOINT_NOT_REACHED,
    RejectionCode.G2_TOPOLOGY_DRIFT,
    RejectionCode.G2_ENERGY_INCOMPLETE,
    RejectionCode.G2_PATH_DISCONTINUOUS,
    RejectionCode.G2_COLLISION,
)

#: Ordered validity predicates of ``g2.inspect``: the first violated
#: predicate in this tuple decides the ``failure_code`` of an invalid path.
FAILURE_PRECEDENCE: Final[tuple[RejectionCode, ...]] = (
    RejectionCode.G2_XTB_FAILED,
    RejectionCode.G2_ENDPOINT_NOT_REACHED,
    RejectionCode.G2_TOPOLOGY_DRIFT,
    RejectionCode.G2_ENERGY_INCOMPLETE,
    RejectionCode.G2_PATH_DISCONTINUOUS,
    RejectionCode.G2_COLLISION,
)

#: Forward failure codes that trigger a reversed start/end retry.
#: ``G2_XTB_FAILED`` is an infrastructure failure and is deliberately absent.
DEFAULT_REVERSE_RETRY_TRIGGERS: Final[tuple[RejectionCode, ...]] = (
    RejectionCode.G2_ENDPOINT_NOT_REACHED,
    RejectionCode.G2_TOPOLOGY_DRIFT,
    RejectionCode.G2_ENERGY_INCOMPLETE,
    RejectionCode.G2_PATH_DISCONTINUOUS,
    RejectionCode.G2_COLLISION,
)

__all__ = [
    "DEFAULT_REVERSE_RETRY_TRIGGERS",
    "FAILURE_PRECEDENCE",
    "G2_FAILURE_CODES",
    "G2_STATUSES",
    "STATUS_FAILED",
    "STATUS_VALID",
]
