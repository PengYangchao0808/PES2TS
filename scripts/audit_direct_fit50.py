#!/usr/bin/env python3
"""Fit50 data-level acceptance audit (WP1, plan v3 §4 checklist).

For every prepared reaction checkpoint, verify per transition:
``X + a = X_next``, logical step decrement, cost 1 per sampled action,
reward components summing to the reward, E/g presence on valid states,
goal truncation (no transitions after the goal terminal), censored masks
(``td_eligible=false``), and vocabulary consistency. Prints a JSON report
with per-check pass/fail counts; exits non-zero on any hard failure.

Reads only prep checkpoints (audit artifacts); no truth, no QC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.contracts.direct import DirectTransition  # noqa: E402

TERMINAL_REASONS = {
    "goal_connected_proxy", "illegal_action", "qc_failure_terminal",
    "budget_exhausted", "no_progress_terminal",
}


def audit_checkpoint(doc: dict) -> dict:
    """Run the transition-level checks for one reaction checkpoint."""
    checks = {
        "x_plus_a_equals_next": 0, "step_decrement": 0, "logical_cost_one": 0,
        "reward_components_sum": 0, "valid_state_has_energy": 0,
        "gradient_shape_ok": 0, "no_after_terminal": 0, "censored_not_td": 0,
        "expert_logprob_null": 0,
    }
    failures: list[str] = []
    transitions = doc.get("transitions", [])
    if not transitions:
        return {"n_transitions": 0, "checks": checks, "failures": failures}
    parsed = []
    for raw in transitions:
        try:
            parsed.append(DirectTransition.from_payload(raw))
        except Exception as exc:
            failures.append(f"payload_invalid:{type(exc).__name__}:{exc}")
            break
    terminated_seen = False
    for tr in parsed:
        label = tr.transition_id
        if terminated_seen:
            failures.append(f"transition_after_terminal:{label}")
        # X + a = X_next for executed actions. Payload serialization rounds
        # coordinates to 1e-6 and actions to 1e-8, so the round-trip bound is
        # ~2e-6 angstrom; anything larger is a real violation.
        if tr.action.executed and tr.next_state is not None:
            expected = tr.state.coordinates + tr.action.proposed_angstrom
            if np.allclose(expected, tr.next_state.coordinates, atol=2e-6):
                checks["x_plus_a_equals_next"] += 1
            else:
                failures.append(
                    f"x_plus_a_mismatch:{label}:"
                    f"{float(np.abs(expected - tr.next_state.coordinates).max()):.3e}")
        if tr.next_state is not None:
            if tr.next_state.steps_remaining == tr.state.steps_remaining - 1:
                checks["step_decrement"] += 1
            else:
                failures.append(f"step_not_decremented:{label}")
        if tr.logical_cost == 1:
            checks["logical_cost_one"] += 1
        else:
            failures.append(f"logical_cost:{label}")
        total = sum(tr.reward_components.values())
        if abs(total - tr.reward) < 1e-9:
            checks["reward_components_sum"] += 1
        else:
            failures.append(f"reward_components_mismatch:{label}:{total}vs{tr.reward}")
        for st, tag in ((tr.state, "s"), (tr.next_state, "s'")):
            if st is not None and st.observation_mask:
                if st.energy_hartree is not None:
                    checks["valid_state_has_energy"] += 1
                else:
                    failures.append(f"missing_energy:{tag}:{label}")
                grad = st.gradient_hartree_per_angstrom
                if grad is not None and grad.shape == st.coordinates.shape:
                    checks["gradient_shape_ok"] += 1
                elif grad is not None:
                    failures.append(f"gradient_shape:{tag}:{label}")
        if tr.termination_reason.startswith("censored") and tr.td_eligible:
            failures.append(f"censored_is_td:{label}")
        else:
            checks["censored_not_td"] += 1
        if tr.action.log_prob_source == "expert_not_probabilistic" \
                and tr.action.log_prob_z is None:
            checks["expert_logprob_null"] += 1
        if tr.terminated:
            if tr.termination_reason not in TERMINAL_REASONS:
                failures.append(f"unknown_terminal:{label}:{tr.termination_reason}")
            terminated_seen = True
    last = parsed[-1]
    if not last.terminated and not last.termination_reason.startswith("censored"):
        # expert path may simply end at the data slice; that is allowed
        if last.termination_reason not in ("continued",) and \
                last.termination_reason != "data_slice_boundary":
            failures.append(f"tail_reason:{last.termination_reason}")
    return {"n_transitions": len(parsed), "checks": checks, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    totals = {key: 0 for key in ("x_plus_a_equals_next", "step_decrement",
                                 "logical_cost_one", "reward_components_sum",
                                 "valid_state_has_energy", "gradient_shape_ok",
                                 "no_after_terminal", "censored_not_td",
                                 "expert_logprob_null")}
    reactions = {}
    all_failures = []
    n_reactions = 0
    for path in sorted(args.prep_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        report = audit_checkpoint(doc)
        reactions[doc["reaction_id"]] = {
            "status": doc["status"], "n_transitions": report["n_transitions"],
            "n_failures": len(report["failures"]),
        }
        n_reactions += 1
        for key, value in report["checks"].items():
            totals[key] += value
        all_failures.extend(f"{doc['reaction_id']}:{f}" for f in report["failures"])

    payload = {
        "prep_dir": str(args.prep_dir),
        "n_reactions": n_reactions,
        "totals": totals,
        "failures": all_failures[:100],
        "n_failures": len(all_failures),
        "verdict": "pass" if not all_failures else "fail",
        "reactions": reactions,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text)
    print(f"[audit] reactions={n_reactions} failures={len(all_failures)} "
          f"verdict={payload['verdict']}")
    if all_failures:
        for f in all_failures[:10]:
            print("  FAIL", f)
    return 0 if not all_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
