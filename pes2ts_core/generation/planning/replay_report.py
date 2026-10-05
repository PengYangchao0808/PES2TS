"""Pilot replay judge — pure functions over pilot bundles (G2-AB2 WP-2).

Reads the pilot campaign artifacts (terminal state, continuation result,
origin evidence, per-case round2/round3 reference) and emits a frozen
``passed/failed/blocked`` verdict per case plus the aggregate V2-gate
evidence index.  Pure: no I/O, no engine, no truth access — the caller hands
in parsed documents, so the judge is trivially auditable and the criteria
cannot drift behind a CLI flag (plan §10: 禁止位移门).

Frozen criteria (V2 gate, plan §1.3 D2; semantics revision v2, ADR-0005):

- ``identity_conflicts`` — zero ``IDENTITY_CONFLICT`` occurrences (B1-1);
- ``origin_failure_typed`` — WHEN the free optimization loses a connection,
  the failure is exactly ``ORIGIN_CONNECTION_LOST`` with per-pair atoms,
  before/after distances and the source-structure hash (B1-3); a generic
  code fails the criterion;
- ``progressed_beyond_reference`` — the replay accepted at least one frame
  beyond the prepared origin AND advanced past the reference failure λ when
  a round2/round3 reference exists (B1-2).  Applicable only to cases that
  reached the continuation; an origin-failed case records ``None`` (not
  applicable) because for a broken-bond pilot case the D2 criterion IS the
  typed ``ORIGIN_CONNECTION_LOST`` detail (ADR-0005);
- verdict ``blocked`` (with the typed needs list) whenever the campaign
  terminal itself is blocked — an unavailable environment is never a fail.
"""
from __future__ import annotations

from typing import Any

PILOT_REPLAY_SCHEMA = "pes2ts_pilot_replay_report_v2"

CRITERIA_REVISION = "v2_adr0005"

ORIGIN_FAILED_PROGRESS_NOTE = (
    "progressed_beyond_reference is not applicable when the case failed at "
    "origin preparation; the D2 criterion for an origin-failed case is the "
    "typed ORIGIN_CONNECTION_LOST detail (judge v2, ADR-0005)")

MIN_ACCEPTED_FRAMES_BEYOND_ORIGIN = 1


def _origin_failure_detail_complete(origin_evidence: dict[str, Any]) -> bool:
    if origin_evidence.get("failure_code") != "ORIGIN_CONNECTION_LOST":
        return False
    detail = origin_evidence.get("failure_detail") or {}
    lost = detail.get("lost_connections")
    if not isinstance(lost, list) or not lost:
        return False
    for row in lost:
        if not isinstance(row, dict) or "atoms" not in row \
                or "distance_before_angstrom" not in row \
                or "distance_after_angstrom" not in row:
            return False
    return detail.get("source_geometry_hash") == origin_evidence.get(
        "source_geometry_hash")


def judge_pilot_case(case: dict[str, Any]) -> dict[str, Any]:
    """Judge one pilot case from its campaign artifacts.

    ``case`` keys: ``terminal`` (mandatory, ``pes2ts_g2t_terminal_state_v1``),
    optional ``origin_evidence`` / ``result`` (continuation snapshot) /
    ``reference`` (``{"last_accepted_lambda": float|None, ...}``).
    """
    terminal = case.get("terminal")
    if not isinstance(terminal, dict) or terminal.get(
            "schema_version") != "pes2ts_g2t_terminal_state_v1":
        raise ValueError("INVALID_PILOT_TERMINAL")
    reference = case.get("reference") or {}
    reference_lambda = reference.get("last_accepted_lambda")
    origin_evidence = case.get("origin_evidence") or {}
    identity_ok = int(terminal.get("identity_conflict_count") or 0) == 0
    n_beyond = int(terminal.get("n_accepted_frames_beyond_origin") or 0)
    last_lambda = terminal.get("last_accepted_lambda")
    beyond_reference = bool(
        n_beyond >= MIN_ACCEPTED_FRAMES_BEYOND_ORIGIN
        and (reference_lambda is None
             or (isinstance(last_lambda, (int, float))
                 and float(last_lambda) > float(reference_lambda))))
    origin_failed = terminal.get("origin_status") == "failed"
    origin_typed = (bool(_origin_failure_detail_complete(origin_evidence))
                    if origin_failed else True)
    if terminal.get("terminal_class") == "blocked":
        return {"verdict": "blocked", "blocked_reason": terminal.get("blocked_reason"),
                "needs": list(terminal.get("blocked_needs", [])),
                "criteria": {"identity_conflicts": identity_ok,
                             "progressed_beyond_reference": False,
                             "origin_failure_typed": origin_typed},
                "evidence": {"terminal_class": "blocked",
                             "reference_last_accepted_lambda": reference_lambda}}
    evidence = {"terminal_class": terminal.get("terminal_class"),
                "failure_code": terminal.get("failure_code"),
                "n_accepted_frames_beyond_origin": n_beyond,
                "last_accepted_lambda": last_lambda,
                "reference_last_accepted_lambda": reference_lambda,
                "origin_failure_code": terminal.get("origin_failure_code"),
                "identity_conflict_count":
                    terminal.get("identity_conflict_count"),
                "n_gradient": (terminal.get("costs") or {}).get("n_gradient")}
    if origin_failed:
        criteria = {"identity_conflicts": identity_ok,
                    "progressed_beyond_reference": None,
                    "origin_failure_typed": origin_typed}
        passed = identity_ok and origin_typed
        return {"verdict": "passed" if passed else "failed",
                "criteria": criteria,
                "criteria_note": ORIGIN_FAILED_PROGRESS_NOTE,
                "evidence": evidence}
    criteria = {"identity_conflicts": identity_ok,
                "progressed_beyond_reference": beyond_reference,
                "origin_failure_typed": origin_typed}
    passed = all(criteria.values())
    return {"verdict": "passed" if passed else "failed",
            "criteria": criteria,
            "evidence": evidence}


def judge_pilot(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judge over the pilot cases (V2 gate evidence index)."""
    if not cases:
        raise ValueError("EMPTY_PILOT")
    verdicts = {rid: judge_pilot_case(case) for rid, case in sorted(cases.items())}
    counts = {kind: sum(1 for v in verdicts.values() if v["verdict"] == kind)
              for kind in ("passed", "failed", "blocked")}
    identity_conflicts_total = sum(
        int(case["terminal"].get("identity_conflict_count") or 0)
        for case in cases.values())
    return {
        "schema_version": PILOT_REPLAY_SCHEMA,
        "gate": "V2_pilot",
        "criteria_frozen": True,
        "criteria_revision": CRITERIA_REVISION,
        "min_accepted_frames_beyond_origin": MIN_ACCEPTED_FRAMES_BEYOND_ORIGIN,
        "verdicts": verdicts,
        "counts": counts,
        "identity_conflicts_total": identity_conflicts_total,
        "all_passed": counts["passed"] == len(verdicts) and counts["blocked"] == 0,
        "notes": ["a blocked environment is never a fail (typed needs listed)",
                  "passed proves engineering usability, not chemical correctness",
                  "judge v2 (ADR-0005): origin-failed cases are judged on the "
                  "typed ORIGIN_CONNECTION_LOST detail; "
                  "progressed_beyond_reference is null there"],
    }


__all__ = ["CRITERIA_REVISION", "PILOT_REPLAY_SCHEMA", "judge_pilot",
           "judge_pilot_case"]
