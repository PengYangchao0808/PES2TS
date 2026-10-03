"""Scan-strategy v2 package: graph-theoretic PES scan planning contracts.

The public document surface lives in
:mod:`pes2ts_core.generation.planning.contracts_v2` and is re-exported here.
Later todos add the endpoint graph, selector core, capability registry, and
ORCA compiler modules alongside these contracts.
"""

from __future__ import annotations

def __getattr__(name):
    """Load document contracts only when requested, not for numeric execution."""
    if name not in __all__:
        raise AttributeError(name)
    from pes2ts_core.generation.planning import contracts_v2
    result = getattr(contracts_v2, name)
    globals()[name] = result
    return result


__all__ = [
    "BASE_REQUIRED",
    "CANDIDATE_KIND_PATH",
    "CANDIDATE_KIND_SCAN",
    "CANDIDATE_KINDS",
    "CAPABILITY_CHECK_STATUSES",
    "CAPABILITY_MODES",
    "COVERAGE_KINDS",
    "COMPILED_KINDS",
    "ContractError",
    "DIRECTIONS",
    "DRIVER_KINDS",
    "ENDPOINTS",
    "EPISTEMIC_ENDPOINT_HYPOTHESIS",
    "FORBIDDEN_KEYS",
    "MAX_SCAN_DRIVERS",
    "METHOD_NEB",
    "METHOD_XTB_PATH",
    "MODE_COUPLED_1D",
    "MODE_SCHEDULED_1D",
    "MODE_SINGLE_1D",
    "OBJECTS",
    "OBJECT_BACKEND_CAPABILITY",
    "OBJECT_FIELDS",
    "OBJECT_GENERATION_PLAN",
    "OBJECT_STRATEGY_PROPOSAL",
    "OPTIONAL_FIELDS",
    "PATH_CANDIDATE_OPTIONAL",
    "PATH_CANDIDATE_REQUIRED",
    "PATH_METHOD_KINDS",
    "PROPOSAL_CANDIDATE_OPTIONAL",
    "PROPOSAL_CANDIDATE_REQUIRED",
    "SCHEDULE_KINDS",
    "SCAN_CANDIDATE_OPTIONAL",
    "SCAN_CANDIDATE_REQUIRED",
    "SCAN_MODES",
    "SCHEMA_BACKEND_CAPABILITY",
    "SCHEMA_GENERATION_PLAN",
    "SCHEMA_STRATEGY_PROPOSAL",
    "SCHEMA_VERSIONS",
    "SPLITS",
    "STATUSES",
    "dumps_v2_document",
    "loads_v2_document",
    "make_backend_capability",
    "make_generation_plan",
    "make_strategy_proposal",
    "seal_document",
    "validate_v2_document",
]
