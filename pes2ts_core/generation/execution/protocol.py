"""PES-generation execution-backend seam frozen by ADR-0001 (workstream X0).

One frozen plan, one execution protocol, one canonical trajectory record. The
seam lives between the frozen plan (``g1_generation_plan_v2``) and the backend,
NOT between ``ReactionCase`` and the executor.

This module is chemistry-free and imports only the standard library plus the
sibling :mod:`record` dataclass, so any backend can implement the protocol
without pulling the pipeline in. A backend is selected by its ``method``
string via an explicit dispatch table — no plugin/registry framework.
"""
from __future__ import annotations

from typing import Final, Mapping, Protocol, runtime_checkable

from pes2ts_core.generation.execution.record import TrajectoryRecord

#: Method vocabulary. ``XTB_PATH`` is the cheap whole-trajectory method (the
#: current G2 metadynamics backend); the ORCA methods are the targeted
#: constrained / path methods from the planning plane; ``CONTINUATION`` is the
#: single-point-gradient experiment and is admitted last (ADR-0001).
METHOD_XTB_PATH: Final[str] = "XTB_PATH"
METHOD_ORCA_SCAN: Final[str] = "ORCA_SCAN"
METHOD_ORCA_CONSTRAINTS: Final[str] = "ORCA_CONSTRAINTS"
METHOD_ORCA_NEB: Final[str] = "ORCA_NEB"
METHOD_CONTINUATION: Final[str] = "CONTINUATION"
METHODS: Final[tuple[str, ...]] = (
    METHOD_XTB_PATH,
    METHOD_ORCA_SCAN,
    METHOD_ORCA_CONSTRAINTS,
    METHOD_ORCA_NEB,
    METHOD_CONTINUATION,
)
#: Cheap, whole-dataset default methods (planning plane may still choose a
#: targeted method per reaction via ``run_if``/``fallback_ids``).
CHEAP_METHODS: Final[tuple[str, ...]] = (METHOD_XTB_PATH,)


def validate_method(method: str) -> str:
    """Return *method* unchanged if known, otherwise raise ``ValueError``."""
    if method not in METHODS:
        raise ValueError(f"unknown execution method {method!r}; expected one of {METHODS}")
    return method


@runtime_checkable
class ExecutionBackend(Protocol):
    """The single execution protocol for PES generation.

    ``prepare`` materialises the backend's run directory from the frozen plan
    and materials; ``run`` executes it and returns exactly one canonical
    :class:`TrajectoryRecord`. Implementations must stay truth-free and must
    express failures as typed statuses, never as silent partial success.
    """

    method: str

    def prepare(
        self, plan: Mapping[str, object], materials: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Materialise the run directory; return a preparation record."""
        ...

    def run(
        self, plan: Mapping[str, object], materials: Mapping[str, object]
    ) -> TrajectoryRecord:
        """Execute the plan and return one canonical trajectory record."""
        ...
