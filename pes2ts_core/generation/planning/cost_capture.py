"""R5 attempt-level cost capture from execution records (G2-AB2 WP-1).

Derives ``flywheel.cost_ledger`` records from the artifacts the continuation
stack already produces — the corrector's per-attempt ``backend_evidence``
(n_gradient_evaluations, duration, ACP receipt reuse flags) and the
origin-preparation evidence.  Counting rules frozen here and unit-locked in
``tests/test_cost_capture.py``:

- one gradient evaluation also computes its energy → ``N_energy = N_gradient``
  (ORCA ``EnGrad`` returns both; no separate energy call exists);
- a cached/reused receipt performs no computation → zero counters, zero
  measured seconds, and ``cost_source = "cache_hit:<evaluation_id>"`` (a true
  measured zero, never an unknown written as 0);
- ``cpu_seconds`` prefers the ACP receipt's measured ``wall_seconds``; the
  in-process duration is the fallback; absence is ``null`` + reason (R5);
- ``allocated_core_seconds`` is a separate account (allocation, not
  measurement) and stays ``null`` + reason unless an allocation is known;
- a numerical/curvature probe is its OWN ledger — its internal gradient
  evaluations are counted once, in the probe ledger, and never also in the
  parent attempt's counter (parent/child dedup via ``includes_cost_ids``);
- nothing is dropped: attempts without backend evidence still get a ledger
  with ``null`` costs and a typed reason.
"""
from __future__ import annotations

from typing import Any

from pes2ts_core.flywheel import aggregate_costs, cost_ledger

#: Schema of the per-case cost bundle written by the campaign runner.
CASE_COSTS_SCHEMA = "pes2ts_case_costs_v1"

_CPU_REASON_NO_EVIDENCE = "no_backend_evidence_recorded"
_CPU_REASON_NOT_MEASURED = "backend_evidence_has_no_measured_duration"
_ALLOC_REASON = "no_allocated_core_budget_recorded"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _attempt_ledger(reaction_id: str, attempt: dict[str, Any],
                    *, attempt_index: int) -> dict[str, Any]:
    attempt_id = attempt.get("attempt_id") or f"attempt-{attempt_index:04d}"
    cost_id = f"{reaction_id}:continuation:{attempt_id}"
    evidence = attempt.get("backend_evidence")
    if not isinstance(evidence, dict):
        return cost_ledger(cost_id=cost_id, n_energy=0, n_gradient=0, n_hessian=0,
                           n_optts_trials=0, cpu_seconds=None,
                           cpu_seconds_reason=_CPU_REASON_NO_EVIDENCE,
                           allocated_core_seconds=None,
                           allocated_core_seconds_reason=_ALLOC_REASON,
                           cost_source="missing_backend_evidence")
    acp = evidence.get("acp") if isinstance(evidence.get("acp"), dict) else {}
    n_gradient = int(evidence.get("n_gradient_evaluations") or 0)
    reused_receipt = bool(acp.get("reused"))
    if reused_receipt:
        # The whole corrector receipt was replayed from cache: no QC call ran.
        return cost_ledger(cost_id=cost_id, n_energy=0, n_gradient=0, n_hessian=0,
                           n_optts_trials=0, cpu_seconds=0.0,
                           allocated_core_seconds=0.0,
                           cost_source="cache_hit:" + str(attempt_id))
    cpu = _numeric(acp.get("wall_seconds"))
    if cpu is None:
        cpu = _numeric(evidence.get("duration_seconds"))
    return cost_ledger(cost_id=cost_id, n_energy=n_gradient, n_gradient=n_gradient,
                       n_hessian=0, n_optts_trials=0, cpu_seconds=cpu,
                       cpu_seconds_reason=None if cpu is not None else _CPU_REASON_NOT_MEASURED,
                       allocated_core_seconds=None,
                       allocated_core_seconds_reason=_ALLOC_REASON,
                       cost_source="acp_orca_gradient_attempt")


def _probe_ledgers(reaction_id: str, attempt: dict[str, Any],
                   *, attempt_index: int) -> list[dict[str, Any]]:
    """One ledger per curvature probe; internal gradients counted ONCE here."""
    probes = attempt.get("curvature_probes")
    if not isinstance(probes, list):
        probe = attempt.get("curvature_probe")
        probes = [probe] if isinstance(probe, dict) else []
    ledgers = []
    for order, probe in enumerate(probes):
        if not isinstance(probe, dict):
            continue
        attempt_id = attempt.get("attempt_id") or f"attempt-{attempt_index:04d}"
        cost_id = f"{reaction_id}:probe:{attempt_id}:{order}"
        n_gradient = int(probe.get("n_gradient_evaluations")
                         or probe.get("n_evaluations") or 0)
        cpu = _numeric(probe.get("duration_seconds"))
        if cpu is None:
            cpu = _numeric(probe.get("wall_seconds"))
        ledgers.append(cost_ledger(cost_id=cost_id, n_energy=n_gradient,
                                   n_gradient=n_gradient, n_hessian=0, n_optts_trials=0,
                                   cpu_seconds=cpu,
                                   cpu_seconds_reason=None if cpu is not None
                                   else _CPU_REASON_NOT_MEASURED,
                                   allocated_core_seconds=None,
                                   allocated_core_seconds_reason=_ALLOC_REASON,
                                   cost_source="curvature_probe"))
    return ledgers


def capture_continuation_costs(result: dict[str, Any], *, reaction_id: str) -> dict[str, Any]:
    """Derive per-attempt R5 ledgers from one continuation result snapshot."""
    if not isinstance(result, dict) or not isinstance(result.get("attempts"), list):
        raise ValueError("INVALID_CONTINUATION_RESULT")
    ledgers: list[dict[str, Any]] = []
    for index, attempt in enumerate(result["attempts"]):
        if not isinstance(attempt, dict):
            raise ValueError("INVALID_ATTEMPT_RECORD")
        ledgers.append(_attempt_ledger(reaction_id, attempt, attempt_index=index))
        ledgers.extend(_probe_ledgers(reaction_id, attempt, attempt_index=index))
    return {"schema_version": CASE_COSTS_SCHEMA, "reaction_id": reaction_id,
            "stage": "continuation", "ledgers": ledgers,
            "aggregate": aggregate_costs(ledgers) if ledgers else None}


def capture_origin_cost(evidence: dict[str, Any], *, reaction_id: str) -> dict[str, Any] | None:
    """Ledger for one origin-preparation evaluation (or None when none ran)."""
    if not isinstance(evidence, dict):
        raise ValueError("INVALID_ORIGIN_EVIDENCE")
    if evidence.get("status") == "invalid_input":
        # Overlap screen rejected the input before any QC call: zero-cost typed
        # rejection — an auditable true zero, not an unknown.
        return None
    duration = _numeric(evidence.get("duration_seconds"))
    acp = evidence.get("acp") or {}
    wall = _numeric(acp.get("wall_seconds")) if isinstance(acp, dict) else None
    cpu = wall if wall is not None else duration
    n_calls = 1 if evidence.get("status") in {"prepared", "failed"} else 0
    return cost_ledger(cost_id=f"{reaction_id}:origin:0000", n_energy=n_calls,
                       n_gradient=n_calls, n_hessian=0, n_optts_trials=0,
                       cpu_seconds=cpu,
                       cpu_seconds_reason=None if cpu is not None else _CPU_REASON_NOT_MEASURED,
                       allocated_core_seconds=None,
                       allocated_core_seconds_reason=_ALLOC_REASON,
                       cost_source="origin_free_optimization")


def total_gradient_calls(*cost_bundles: dict[str, Any] | None) -> int:
    """Campaign-level gradient counter (parent/child dedup via aggregate)."""
    total = 0
    for bundle in cost_bundles:
        if not isinstance(bundle, dict):
            continue
        aggregate = bundle.get("aggregate")
        if isinstance(aggregate, dict):
            total += int(aggregate.get("N_gradient") or 0)
            continue
        for ledger in bundle.get("ledgers", []) or []:
            total += int(ledger.get("N_gradient") or 0)
    return total


__all__ = ["CASE_COSTS_SCHEMA", "capture_continuation_costs", "capture_origin_cost",
           "total_gradient_calls"]
